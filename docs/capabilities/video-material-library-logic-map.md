# 视频素材库逻辑地图

> 状态：2026-07-10 Phase 1 已实现  
> 范围：本地视频导入、镜头切分、技术标签、查询、timeline 和 MoneyPrinter 素材接入  
> 重要边界：当前标签是 FFmpeg/媒体元数据生成的技术标签，不是 AI 语义理解。

## 1. 已完成能力

| ID | 能力 | 实现 |
| --- | --- | --- |
| `CAP-001` | 本地视频导入 | 显式 `sourcePath`、扩展名白名单、SHA-256 去重、复制到 managed root |
| `CAP-002` | 媒体探测 | FFprobe 读取时长、分辨率、FPS |
| `CAP-003` | 镜头切分 | FFmpeg scene detection；最小时长、最大数量和固定时长 fallback |
| `CAP-004` | 片段与关键帧 | H.264/AAC MP4 片段和中点 JPEG 关键帧 |
| `CAP-005` | 技术标签 | 横屏/竖屏/方形、短/中/长镜头、高清 |
| `CAP-006` | 查询 | 按 asset 或精确 tag 查询 clips |
| `CAP-007` | Timeline | 将选中片段写入原子化 `timeline.json` |
| `CAP-008` | MoneyPrinter 接入 | Desktop 可把生成片段复制到 MoneyPrinter local material 并自动选中 |
| `CAP-009` | Agent 接入 | MCP tools 支持导入、分析、查询和 timeline 创建 |

## 2. 数据与文件

默认根目录：`$HERMES_HOME/video-library/`

```text
video-library/
├── video_library.db
├── assets/
├── clips/<asset_id>/
├── keyframes/<asset_id>/
└── timelines/
```

SQLite 表：

| 表 | 用途 |
| --- | --- |
| `assets` | 原视频、内容哈希、managed path、媒体元数据、状态 |
| `clips` | 片段边界、文件、关键帧、描述和状态 |
| `tags` | 去重标签字典 |
| `clip_tags` | clip-tag 关系、confidence、source |
| `analysis_jobs` | analyzer 版本、状态、进度和错误 |

所有生成文件必须位于当前素材库根目录下。timeline 不允许引用根目录外的 clip 路径。

## 3. API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `POST` | `/api/capabilities/video-library/assets` | 导入本地视频 |
| `GET` | `/api/capabilities/video-library/assets` | 列出原视频 |
| `POST` | `/api/capabilities/video-library/assets/{asset_id}/analyze` | 探测、切分、关键帧、标签 |
| `GET` | `/api/capabilities/video-library/clips?asset_id=&tag=` | 查询片段 |
| `POST` | `/api/capabilities/video-library/clips/{clip_id}/tags` | 原子替换片段标签 |
| `POST` | `/api/capabilities/video-library/timelines` | 创建 timeline |

Desktop `hermes serve` 和 API server 都注册相同路径。响应统一为：

```json
{"ok": true, "data": {}, "error": null}
```

## 4. 核心流程

```text
Desktop 选择视频
  -> MoneyPrinter 本地素材上传
  -> video-library import
  -> 用户点击“切分分析”
  -> ffprobe
  -> scene boundaries
  -> clip MP4 + keyframe JPEG
  -> SQLite clips/tags/job
  -> Desktop 显示片段和标签
  -> “加入混剪”复制到 MoneyPrinter local_videos
  -> Video Studio 生成任务
```

Agent 流程：

```text
video_library_import_asset
  -> video_library_analyze_asset
  -> video_library_search_clips
  -> video_library_create_timeline
```

## 5. 规则

1. `HERMES_HOME` 决定数据库和媒体根目录，禁止写入源码目录。
2. 只导入受支持的视频文件；目录、缺失文件和图片会被拒绝。
3. 重复内容按 SHA-256 返回同一 asset，不重复复制。
4. 重新分析先写 staging；完成后交换目录，并在同一 SQLite 事务提交 clips、tags、asset metadata 和 job。失败会恢复旧目录和旧记录。
5. FFmpeg 子进程使用参数数组、超时和受限错误输出，不拼接 shell 字符串。
6. 当前 UI 的“标签”只表示技术标签；不得宣称已识别人、商品、场景或文案含义。

## 6. 尚未完成

| 优先级 | 缺口 | 下一实现 |
| --- | --- | --- |
| P0 | AI 语义标签 | 关键帧视觉描述 + ASR + 结构化标签 schema |
| P0 | 文案镜头匹配 | 文案段落 embedding 与 clip embedding 检索/重排 |
| P0 | Timeline 渲染 | 把 timeline 映射为 MoneyPrinter render task 并回写产物 |
| P1 | 人工编辑 | 片段预览、标签增删、描述编辑、多选和排序 UI |
| P1 | 长任务体验 | 后台 job queue、取消、断点、进度事件和失败重试 |
| P1 | 搜索 | 模糊标签、全文、向量和组合过滤 |
| P2 | 质量评估 | 模糊、抖动、曝光、音频、重复镜头和版权来源标记 |

## 7. 测试

| 范围 | 文件 | 当前结果 |
| --- | --- | --- |
| SQLite schema/事务/去重/job | `tests/capabilities/test_video_library_store.py` | 4 passed |
| scene parser/路径约束/真实 FFmpeg | `tests/capabilities/test_video_library_media.py` | 5 passed |
| service 分析/timeline/失败回滚 | `tests/capabilities/test_video_library_service.py` | 3 passed |
| FastAPI session auth/routes | `tests/capabilities/test_video_library_web_routes.py` | 2 passed |
| MCP 注册/调用 | `tests/capabilities/test_moneyprinter_mcp_tools.py` | 相关测试通过 |
| Desktop client/type/build | `moneyprinter-client.test.ts` + typecheck/build | 通过 |
