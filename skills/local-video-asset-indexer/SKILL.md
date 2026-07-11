---
name: local-video-asset-indexer
description: 自动扫描、切分、理解、分类和索引本地实拍视频素材。用户提到素材入库、素材目录扫描、视频分类、自动打标签、Obsidian 视频资产库、本地镜头库、牛肉面资产库或更新视频索引时必须使用。通过 Hermes video_library CLI 操作显式配置的素材库，不移动、改名或删除原始视频；爆款参考视频的导演级分析改用 director-lapian。
---

# 本地视频资产入库

## 目标

把配置过的实拍视频目录转换为可供 Video Studio 查询的镜头级资产索引。以 CLI 返回的 JSON、SQLite 状态和落盘关键帧为执行真相，不根据聊天上下文猜测完成度。

## 路由边界

- 自有门店实拍、素材分类、批量入库：执行本 Skill。
- 爆款参考视频、导演意图、逐镜头拉片、风格圣经：使用 `director-lapian`。
- 用户同时提供两类视频：分别执行两套工作流，不把爆款参考片写进自有素材索引。

## 强制规则

1. 只使用 `libraries` 命令返回的资产库 ID。
2. 不得移动、改名或删除原始视频。
3. 不得扫描配置范围以外的目录，也不得跟随逃逸 source root 的符号链接。
4. 先 dry-run，再进行首次真实扫描。
5. 失败文件不阻断整批任务；最终逐项报告失败阶段。
6. 不把长篇导演分析报告当作机器素材索引。
7. 需要解释字段时完整阅读 `references/shot-schema.md`。
8. 需要解释或修改牛肉面标签时完整阅读 `references/beef-noodle-taxonomy.md`。

## 标准流程

### 1. 确认资产库

```bash
python -m capabilities.video_library.cli libraries
```

如果目标资产库未出现，停止并说明需要在当前 Hermes Profile 的 `config.yaml` 中增加 `video_libraries`。不要自行选择相似目录。

### 2. 读取当前状态

```bash
python -m capabilities.video_library.cli status --library beef-noodle
```

### 3. 检查派生素材污染

```bash
python -m capabilities.video_library.cli prune-derived --library beef-noodle
```

该命令默认只预览。如果 `matched` 大于 0，先报告记录，确认它们的来源全部位于资产库派生目录，再执行：

```bash
python -m capabilities.video_library.cli prune-derived --library beef-noodle --execute
```

该操作只删除 SQLite 中错误收录的资产记录，不删除任何原片或派生视频文件。

### 4. 首次执行先预检

```bash
python -m capabilities.video_library.cli scan --library beef-noodle --dry-run
```

确认 `writes_planned` 全部位于目标资产库目录；否则停止。

### 5. 执行全自动扫描

```bash
python -m capabilities.video_library.cli scan --library beef-noodle
```

执行包含：内容指纹、媒体探测、长短素材自适应切镜、关键帧、视觉语义分析、固定标签、自由标签、质量状态、SQLite 入库和 Obsidian Markdown 投影。

### 6. 复核检索

至少选择三条符合真实门店素材的查询：

```bash
python -m capabilities.video_library.cli search --library beef-noodle --query '厨师拉面'
python -m capabilities.video_library.cli search --library beef-noodle --query '牛肉切片'
python -m capabilities.video_library.cli search --library beef-noodle --query '热汤浇入碗中'
```

检查返回记录包含源文件、时间范围、关键帧、描述、标签、质量、置信度和状态。不要只检查结果数量。

### 7. 验证增量性

再次运行扫描。没有变化的素材应计入 `skipped`，不能重新调用视觉分析或创建重复资产。

## 结果汇报

简洁列出：

- 资产库 ID 和根目录。
- 扫描视频总数。
- 新完成、跳过和失败数量。
- 低置信度、不可用和语义失败镜头数量。
- 三条代表查询及最高相关镜头。
- Obsidian 分析页与 SQLite 路径。
- 未解决的具体失败阶段。

不要把“部分完成”写成“全部完成”。
