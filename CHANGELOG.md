# 更新日志 / CHANGELOG

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。
所有日期取自文件的实际创建/修改时间戳（可由 `git log` 或 `tools/move_log.csv` 复核）。

---

## [v0.3.0] - 2026-11-12 — 目录重构 + 可移植化 + 项目起源文档

### Added
- `README.md`：项目起源总说明（含 mermaid 架构图、指标表、曲线图、目录树）。
- `docs/PIPELINE.md`、`docs/DATASET.md`、`docs/BOM.md`、`docs/MIGRATION.md`：流水线手册 / 数据规范 / 物料清单 / 迁移映射表。
- `software/scripts/paths.py`：唯一的路径、设备、权重解析入口，仓库整体与盘符解耦。
- `software/configs/args.luosi_luomu.yaml`、`args.jiaoma_zhoucheng.yaml`、`dataset.example.yaml`：超参与数据集配置快照。
- `docs/assets/`：从训练产物抽取的 23 张 README 引用图（曲线、混淆矩阵、GT/PRED 对照、样例图）。
- `tools/reorganize.ps1` + `tools/move_log.csv`：可逆重排脚本与逐步日志（支持 `-DryRun` / `-Undo`）。
- `.gitignore`、`.gitattributes`、`requirements.txt`、`LICENSE`（AGPL-3.0，继承上游）。

### Changed
- **目录重新分类**：`ultralytics-main/` 内的自有内容按职责拆到 `software/{scripts,configs,datasets,models,runs,samples,vendor}`，机械文件集中到 `hardware/`。
- **脚本更名（望文生义 + 顺序化）**：`photos.py → collect_photos.py`、`ram.py → shuffle_rename.py`、`predict.py → predict_video.py`、`real.py → predict_camera.py`；`split_dataset.py`、`train.py` 保留原名。
- **消灭绝对路径**：6 个脚本中的 `D:\Windows_Desktop\ultralytics-main\...` 全部替换为 `paths.*` 推导值；训练输出目录改为 `software/runs/<task>/runs/26train`。
- **权重加载策略**：`paths.resolve_weights()` 按 `models/trained/<task>.pt → runs/<task>/.../best.pt → models/base/yolo26n.pt` 顺序解析，推理默认使用**自训练权重**（此前指向 COCO 基座）。
- **设备自适应**：`paths.pick_device()` 有 CUDA 用 GPU、否则回退 `cpu`，可用 `VISIS_DEVICE` 强制。
- `dataset.yaml` 默认省略 `path` 键（改由 yaml 所在目录推导）；`split_dataset.py` 新增开关 `WRITE_ABSOLUTE_PATH = False`。
- 推理输出目录统一：视频 → `software/results/video/exp/`，实时截图 → `software/results/camera/`（原 `camera_results/` 为空目录，已删除）。

### Fixed
- `train.py` / `predict.py` / `real.py` / `split_dataset.py` 里的硬编码路径**早已指向不存在的目录**（实际文件夹是 `视觉检测台\ultralytics-main\`），重构后全部可运行。
- `dataset.yaml` 的 `path: D:/Windows_Desktop/ultralytics-main/output` 指向已不存在的路径，训练会定位失败 —— 已修正为可移植形式。

### Removed
- 去重：`ultralytics-main/yolo26n.pt` 与 `ultralytics-main/weights/yolo26n.pt` SHA256 完全相同，保留后者。
- 清理：153 个 `.pyc` / 25 个 `__pycache__`（2.1 MB）；空的 `weights/`、`camera_results/` 目录。

---

## [v0.2.0] - 2026-10-21 — 角码 / 轴承任务（当前主力模型）

- 新任务：`jiaoma`（角码）+ `zhoucheng`（轴承）。
- 数据集：**105 张** 1280×720 实拍（`collect_photos.py` 连拍 + `shuffle_rename.py` 打乱改名 + 人工标注 300 个框），划分 84/10/11（seed 42）。
- 训练：YOLO26n，300 epoch 目标、`patience=100`，实际 **282 epoch / 846.4 s**，AdamW + cos_lr，冻结 Backbone 0~10 层。
- 结果：**P 1.000 / R 1.000 / mAP@0.5 0.995 / mAP@0.5:0.95 0.801**（较 v0.1 的 0.627 提升 0.174）。
- 产物：`software/runs/jiaoma_zhoucheng/runs/26train/exp/`（含 `weights/best.pt`、`results.csv`、中文训练图表）。
- 2026-10-25：`train.py` 调参与中文图表输出完善（图表字体自动回退、`l1_loss` 列名兼容）。

## [v0.1.0] - 2026-10-19 — 首次全链路跑通（螺丝 / 螺母）

- 首个任务：`luosi`（螺丝）+ `luomu`（螺母）。
- 训练 185 epoch / 541.5 s，结果 P 0.9964 / R 1.000 / mAP@0.5 0.995 / **mAP@0.5:0.95 0.627**。
- 关键结论：**Recall 稳定 1.0**（不漏检）达成，但定位精度（mAP@0.5:0.95）明显不足 → 促使 v0.2 改数据 + 加严训练。
- 同期产出 `predict_video.py`（逐帧检测 + 音轨合并 + 音频时钟调度）与 `predict_camera.py`（实时 + FPS + 截图热键）。

---

## 项目时间线（按文件时间戳还原）

| 日期 | 事件 |
| --- | --- |
| 2026-09-24 | 最早的两个零件建模：`树莓派主机建模.SLDPRT`、`外壳.SLDPRT` —— **项目起点** |
| 2026-10-09 | `双目摄像头建模.SLDPRT` 更新 |
| 2026-10-14 | 下载 `yolo26n.pt`；写下 `train.py`、`predict.py` 第一版 |
| 2026-10-16 | 录制演示视频 `1.mp4`、`2.mp4` |
| 2026-10-17 | `split_dataset.py`（校验 + 8:1:1 划分）、`real.py`（实时检测雏形） |
| 2026-10-18 | `photos.py`（连拍采集）、`ram.py`（打乱改名）—— 数据采集流水线成型 |
| 2026-10-19 | **v0.1** 螺丝/螺母训练完成，best.pt 落盘 |
| 2026-10-21 | 105 张样本重采重标 + **v0.2** 角码/轴承训练完成 |
| 2026-10-25 | `train.py` 训练策略与图表输出定稿 |
| 2026-10-30 | 机械集中定案：斜方管、欧标 3030 下料、直角支架、亚克力托盘 + `视觉检测台.SLDASM` 总装 + `亚克力板.DXF` 出图 |
| 2026-11-01 | 零件库归档整理 |
| 2026-11-09 | `real.py` 实时检测修复定稿（`plot().copy()`、双后端打开摄像头） |
| 2026-11-12 | **v0.3.0** 目录重排 + 路径可移植 + 起源文档 |
