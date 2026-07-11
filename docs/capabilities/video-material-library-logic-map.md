# 视频素材库逻辑地图

> 状态：2026-07-12 统一具名素材库、语义匹配、来源追溯和 MoneyPrinter 桥接已完成真实验收。
> 产品边界：素材理解与镜头编排属于 Hermes capability；MoneyPrinterTurbo 只负责本地媒体预处理、顺序拼接、配音、字幕、音乐和导出。

## 1. 单一来源模型

一条视频任务必须手动选择一个具名资产库。所有导入、扫描、搜索、确认和时间线请求都携带同一个 `libraryId`。

```text
Video Studio task
  -> libraryId=beef-noodle
  -> assetId
  -> clipId
  -> sourcePath + sourceSha256 + sourceStart/sourceEnd
  -> segmentId + script
  -> hidden MoneyPrinter cache filename
```

规则：

1. 未选择资产库时，导入、目录、扫描、匹配、迁移和时间线按钮全部禁用。
2. 新文件只能从该库配置的 `source_roots` 导入；符号链接逃逸和目录外文件被拒绝。
3. 搜索只打开当前具名库的 SQLite，不查询默认库、其他具名库或网络图库。
4. 时间线 clip 必须存在于当前库，且其 asset `library_id` 必须等于请求 `libraryId`。
5. 找不到镜头时显示“未找到合适镜头”，不自动跨库，也不自动回退 Pexels/Pixabay/Coverr。
6. MoneyPrinter `storage/local_videos` 是隐藏渲染缓存，不是第二个用户素材库。

## 2. 文件与数据库

牛肉面资产库示例：

```text
/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/
├── .hermes-assets/index.sqlite
├── .hermes-assets/managed-assets/
├── 02_精选镜头/<asset_id>/clip-XXXX.mp4
├── 03_关键帧/<asset_id>/clip-XXXX.jpg
├── 04_素材分析/*.md
└── timelines/timeline_<uuid>.json
```

SQLite：

| 表 | 作用 |
| --- | --- |
| `assets` | 原文件、SHA-256、library_id、source/managed path、媒体参数和状态 |
| `clips` | 原始起止时间、物化文件、关键帧、语义描述、质量、置信度和状态 |
| `tags` / `clip_tags` | 分类标签及 confidence/source |
| `analysis_jobs` | 分析器版本、进度、重试和错误 |

## 3. 分析与检索

```text
source roots
  -> SHA-256 去重
  -> FFprobe
  -> scene detection / fallback 分段
  -> 关键帧
  -> 视频分析智能体结构化语义
  -> 技术标签 + 固定分类 + 自由标签
  -> SQLite
  -> Obsidian Markdown 投影
```

检索会将整段文案与 clip 的描述、标签和语义字段进行组合评分。返回候选包含 `score`、质量、置信度、关键帧、原始路径和起止时间。候选必须经过人工确认后才能进入本次时间线。

## 4. shot plan 与来源证明

`timeline.json` 同时保存渲染轨道和 `shotPlan/provenance`：

```json
{
  "segmentId": "segment-2",
  "script": "师傅现场拉面，劲道面条配上红亮辣油，饭点就来吃一碗。",
  "libraryId": "beef-noodle",
  "assetId": "asset_643ec53df2fac4d3e3b68e31",
  "clipId": "clip_f2a3122c703b459e9a82ff885e2e9b1f",
  "sourcePath": "/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/视频测试牛肉面/2026_02_03_15_10_13_IMG_2367.MOV",
  "sourceSha256": "643ec53df2fac4d3e3b68e3126ff5589916d3f41f1bca98ade3e8c2aec262a5e",
  "sourceStart": 0.0,
  "sourceEnd": 5.0,
  "tags": ["动作/拉面", "工序/拉面", "手工面制作", "现场制作"]
}
```

复制到隐藏缓存时采用：

```text
<libraryId>-<assetId>-<clipId>-<sha256前12位>-<clip文件名>
```

同名文件不会覆盖，缓存文件可以反查资产库、asset、clip 和原始内容哈希。

## 5. API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/capabilities/video-library/libraries` | 列出当前 Profile 的具名库 |
| `GET` | `/libraries/{library_id}/status` | 资产、镜头和失败统计 |
| `POST` | `/libraries/{library_id}/source-roots` | 经桌面选择后添加允许扫描目录 |
| `POST` | `/libraries/{library_id}/scan` | `dryRun=true` 预扫描；确认后真实扫描 |
| `POST` | `/libraries/{library_id}/migrate-legacy` | 按 SHA-256 幂等迁移旧默认库，保留旧文件 |
| `POST` | `/assets` | 携带 `libraryId` 导入文件 |
| `POST` | `/assets/{asset_id}/analyze` | 携带 `libraryId` 分析 |
| `GET` | `/clips?library_id=&query=&tag=` | 当前库语义/标签检索 |
| `POST` | `/clips/{clip_id}/tags` | 当前库人工标签写入 |
| `POST` | `/timelines` | 当前库确认镜头生成来源可追溯时间线 |

旧无 `library_id` 接口只为迁移兼容保留，不再由 Video Studio 主流程调用。

## 6. 2026-07-12 牛肉面真实验收

实时库状态：

- library：`beef-noodle / 牛肉面资产库`
- root：`/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库`
- 21 assets、85 clips、0 failed、0 low-confidence、0 unusable
- 桌面初始状态未自动选择资产库，所有管理按钮禁用
- 选择资产库后只显示一个“素材库”入口

文案与命中：

1. `热气腾腾的牛肉汤刚刚出锅，大块牛肉铺满整碗。`
   - clip：`clip_bbbfd6f18a4b455cb115950a2bad37e0`
   - score：`0.975`
   - 标签包括 `主体/汤锅`、`主体/牛肉`、`热气腾腾`、`真材实料`
2. `师傅现场拉面，劲道面条配上红亮辣油，饭点就来吃一碗。`
   - clip：`clip_f2a3122c703b459e9a82ff885e2e9b1f`
   - score：`0.61`
   - 标签包括 `动作/拉面`、`工序/拉面`、`厨师拉面`、`手工面制作`

产物：

- timeline：`/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/timelines/timeline_382de1597f78453aa075ca85b0cf1ce8.json`
- MoneyPrinter task：`db753600-f0ea-47e1-ad1d-5897d4e57cac`
- final MP4：`external/MoneyPrinterTurbo/storage/tasks/db753600-f0ea-47e1-ad1d-5897d4e57cac/final-1.mp4`
- MP4：H.264 + AAC、1080×1920、30fps、15.0 秒、约 11 MB
- 1 秒抽帧为汤锅牛肉画面；7 秒抽帧为厨师拉面画面

## 7. 验证

- `48 passed`：`tests/capabilities/test_video_library*.py`
- `40 passed`：Desktop `src/app/video-studio`
- Desktop TypeScript typecheck：通过
- 真实桌面：单入口、手动选库、匹配、人工确认、创建时间线均通过
- 真实渲染：MoneyPrinter 顺序消费两个来源可追溯缓存镜头并生成可播放 MP4

## 8. 当前边界

当前桥接已经保证镜头选择和拼接顺序，不会被 MoneyPrinter 随机打乱。旁白与镜头切点目前仍按已物化 clip 时长和 `video_clip_duration` 对齐，不是逐字级音频时间戳对齐；若后续需要广播级口播节奏，应增加分句 TTS 时长或字幕时间戳驱动的二次 timeline 调整。
