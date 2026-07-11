# Video Studio 逻辑地图

> 状态：2026-07-12 统一素材库与来源可追溯混剪已完成自动化及真实桌面验收。
> 边界：Video Studio 是 Hermes Desktop capability；MoneyPrinterTurbo 是独立 sidecar；不增加 Hermes Agent Core 工具。

## 1. 用户主流程

```text
主题/文案
  -> 手动选择一个具名素材库
  -> 添加文件或选择 source root
  -> dry-run 扫描并确认
  -> 视频分析智能体切镜、描述、分类、标签、质量评分
  -> 按整段文案检索当前库候选
  -> 人工确认每段镜头
  -> 生成 shotPlan + provenance + video track
  -> 物化确认镜头
  -> 复制到隐藏 MoneyPrinter local_videos 缓存
  -> 强制 local + sequential + match_materials_to_script
  -> TTS/字幕/BGM/合成
  -> final MP4 预览与下载
```

## 2. 页面地图

| ID | 页面区域 | 用户控件 | 数据边界 |
| --- | --- | --- | --- |
| `PAGE-001` | Video Studio 页面壳 | Health Check、Start Service | Hermes capability health/start |
| `PAGE-002` | API 配置 | Reload、Save | ignored MoneyPrinter `config.toml`，密钥不回显 |
| `PAGE-003` | 主题与文案 | 生成文案、人工改稿 | `VideoGenerationForm.videoScript` |
| `PAGE-004` | 关键词 | 生成关键词、人工编辑 | 作为语义匹配补充词，不决定跨库来源 |
| `PAGE-005` | 唯一“素材库”工作区 | 选库、添加文件、目录、预扫描、迁移、匹配、确认、时间线 | 只能访问当前 `libraryId` |
| `PAGE-006` | 语音/字幕/BGM | TTS、MiniMax、字体、字幕、BGM、自定义音频 | MoneyPrinter 资产与 MiniMax API |
| `PAGE-007` | 任务列表 | 刷新、选择 | MoneyPrinter task state |
| `PAGE-008` | 预览 | 播放、下载 | adapter 鉴权媒体 URL |

已删除的竞争入口：

- 不再显示“本地素材”上传/勾选列表。
- 不再显示默认 Hermes“视频素材库”上传/切分列表。
- 不再单独显示“Obsidian 具名资产库”卡片。
- `storage/local_videos` 只由确认时间线桥接写入，用户不可将其当长期资产库管理。

## 3. `PAGE-005` 控件与状态

| 控件 | 未选库 | 已选库 | 行为 |
| --- | --- | --- | --- |
| 资产库选择 | 可用，值为空 | 显示当前库 | 每次进入必须手动选择，不恢复草稿中的库 |
| 添加素材文件 | 禁用 | 可用 | 每个请求携带 `libraryId`，导入后分析 |
| 选择素材目录 | 禁用 | 可用 | 桌面原生目录选择，后端写入当前库 source roots |
| 扫描新增素材 | 禁用 | 可用 | 先 `dryRun=true`，出现确认扫描后才能真实写入 |
| 迁移旧素材 | 禁用 | 可用 | 显式 ConfirmDialog，按 SHA-256 去重，旧文件保留 |
| 自动匹配全部文案 | 禁用 | 可用 | 逐段查询当前库，部分失败不清空其他成功段 |
| 选用这个镜头 | 无候选 | 可用 | 人工确认 segment → clip |
| 创建素材时间线 | 禁用 | 至少一个确认后可用 | 生成可追溯 timeline 并桥接隐藏缓存 |

切换或清空资产库会清空：候选、确认、scan preview、migration result、timeline 和 `form.localMaterials`。

## 4. 组件职责

| 组件/模块 | 职责 |
| --- | --- |
| `unified-material-library-panel.tsx` | 单一素材库 UI、候选关键帧、本次已选镜头、迁移确认 |
| `use-named-video-library.ts` | 手动选库、刷新、导入/分析、dry-run/confirm scan、迁移、匹配、确认、时间线 |
| `named-library-matching.ts` | 文案分段、候选/错误/确认状态纯函数 |
| `material-cache.ts` | timeline provenance 校验、稳定缓存命名、强制顺序渲染 form patch |
| `capabilities/video_library/management.py` | source-root 持久化和旧库幂等迁移 |
| `capabilities/video_library/service.py` | clip 物化、单库校验、shotPlan/provenance/timeline 原子写入 |
| `capabilities/moneyprinter/adapter.py` | 安全复制到白名单缓存并代理 MoneyPrinter 任务 |

## 5. 来源指向合同

每个确认镜头必须同时存在：

```text
segmentId + script
libraryId + assetId + clipId
sourcePath + sourceSha256
sourceStart + sourceEnd
description + tags + qualityScore + confidence
```

渲染轨道只引用当前库 `02_精选镜头` 下物化后的 MP4。客户端只有在 `tracks.video` 和 `shotPlan` 的 `clipId` 对得上，并且 `libraryId/assetId/sourceSha256` 完整时，才允许复制进 MoneyPrinter 缓存。

## 6. 错误与回退

| 情况 | 行为 |
| --- | --- |
| 未选择库 | 禁止所有素材写入/匹配/时间线操作 |
| 文件不在 source roots | 拒绝，提示先选择素材目录 |
| 跨库 clip | 服务端拒绝，不静默替换 |
| 当前库无候选 | 当前段显示“未找到合适镜头” |
| 部分分析失败 | 保留成功项，记录逐文件错误 |
| cache copy 失败 | 保留 timeline 和人工确认，不提交不完整素材集合 |
| MoneyPrinter sidecar 退出 | 返回 `MONEYPRINTER_UPSTREAM_UNREACHABLE`，恢复服务后可重试同一任务 |
| 旧库迁移部分失败 | 返回 imported/skipped/failed 和逐资产记录，可幂等重试 |

默认不自动调用网络素材补洞。未来若增加网络补充，必须由用户显式开启并在 shot provenance 标明 provider/license。

## 7. API 地图

### Video library

- `GET /api/capabilities/video-library/libraries`
- `GET /libraries/{library_id}/status`
- `POST /libraries/{library_id}/source-roots`
- `POST /libraries/{library_id}/scan`
- `POST /libraries/{library_id}/migrate-legacy`
- `POST /assets`（body 必须含当前 `libraryId`）
- `POST /assets/{asset_id}/analyze`
- `GET /clips?library_id=&query=&tag=&limit=`
- `POST /clips/{clip_id}/tags`
- `POST /timelines`

### MoneyPrinter

- `GET /api/capabilities/moneyprinter/health`
- `POST /service/start`
- `POST /videos`、`/audio`、`/subtitle`
- `GET /tasks`、`/tasks/{task_id}`
- `POST /materials`（内部 timeline bridge 使用）
- `GET /stream/{file}`、`/download/{file}`

## 8. 真实验收记录

2026-07-12 使用正确开发版：

```text
hermes-dev-desktop restart
Electron: apps/desktop/node_modules/electron/dist/Electron.app
HERMES_HOME: /Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home
```

验收结果：

- 页面只存在一个“素材库”标题；旧三个标题均不存在。
- 初始选择为“请选择资产库”，五个管理按钮和匹配/时间线按钮禁用。
- 手动选择 `牛肉面资产库 · beef-noodle` 后加载当前库。
- 实时状态：21 assets、85 clips、0 failed、0 low-confidence、0 unusable。
- 第一段命中汤锅/牛肉/热气镜头，score 0.975。
- 第二段命中现场拉面镜头，带 `动作/拉面`、`工序/拉面` 标签，score 0.61。
- 人工确认两段并生成：
  `/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/timelines/timeline_382de1597f78453aa075ca85b0cf1ce8.json`
- 缓存文件分别包含 `beef-noodle + asset + clip + source hash`。
- MoneyPrinter 任务：`db753600-f0ea-47e1-ad1d-5897d4e57cac`，状态 complete。
- 成片：`external/MoneyPrinterTurbo/storage/tasks/db753600-f0ea-47e1-ad1d-5897d4e57cac/final-1.mp4`。
- 成片参数：H.264/AAC、1080×1920、30fps、15 秒、约 11 MB。
- 抽帧确认前段为牛肉汤锅，后段为厨师拉面。

## 9. 自动化回归

| 范围 | 命令 | 结果 |
| --- | --- | --- |
| 素材库 Python | `.venv/bin/python -m pytest -q tests/capabilities/test_video_library*.py` | 48 passed |
| Video Studio UI/client/hooks | `npm run test:ui -- src/app/video-studio` | 40 passed |
| Desktop 类型 | `npm run typecheck` | 通过 |
| 工作树格式 | `git diff --check` | 通过 |

## 10. 已知边界

已实现的是“文案语义匹配 → 人工确认 → 确定镜头顺序 → 来源可追溯渲染”。MoneyPrinter 目前仍按 clip 时长和最大片段秒数覆盖旁白，不是逐字级音频时间戳驱动。下一阶段若追求导演级口播节奏，应让分句 TTS 或字幕时间戳回写 `voiceStart/voiceEnd`，再对 shot track 做二次裁剪。
