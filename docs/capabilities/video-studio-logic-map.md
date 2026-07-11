# Video Studio 逻辑地图

> 状态：2026-07-12 统一素材库与来源可追溯混剪已完成自动化及真实桌面验收。
> 边界：Video Studio 是 Hermes Desktop capability；MoneyPrinterTurbo 是独立 sidecar；不增加 Hermes Agent Core 工具。

## 1. 用户主流程

```text
主题/文案
  -> 选择素材来源：本地具名库 / Pexels / Pixabay / Coverr
  -> 本地来源时手动指定一个具名素材库
  -> 添加文件或选择 source root
  -> dry-run 扫描并确认
  -> 视频分析智能体切镜、描述、分类、标签、质量评分
  -> 按整段文案检索当前库候选
  -> AI 按匹配分、质量、置信度与素材多样性自动选镜头
  -> 生成 shotPlan + provenance + video track
  -> 物化 AI 已选镜头
  -> 复制到隐藏 MoneyPrinter local_videos 缓存
  -> 强制 local + sequential + match_materials_to_script
  -> TTS/字幕/BGM/合成
  -> final MP4 预览与下载

在线来源路径由 MoneyPrinter 按 `video_terms` 自动检索、下载和混剪，不要求选择本地具名库。
人工选镜头降级为成片后的可选精修，不再是生成前置条件。
```

## 2. 页面地图

| ID | 页面区域 | 用户控件 | 数据边界 |
| --- | --- | --- | --- |
| `PAGE-001` | Video Studio 页面壳 | Health Check、Start Service | Hermes capability health/start |
| `PAGE-002` | API 配置 | Reload、Save | ignored MoneyPrinter `config.toml`，密钥不回显 |
| `PAGE-003` | 主题与文案 | 生成文案、人工改稿 | `VideoGenerationForm.videoScript` |
| `PAGE-004` | 关键词 | 生成关键词、人工编辑 | 作为语义匹配补充词，不决定跨库来源 |
| `PAGE-005` | 唯一“素材库”工作区 | 来源选择、一键自动成片；本地模式含选库、导入、扫描、候选精修 | 本地严格限制当前 `libraryId`；在线明确记录 provider |
| `PAGE-006` | 语音/字幕/BGM | TTS、MiniMax、字体、字幕、BGM、自定义音频 | MoneyPrinter 资产与 MiniMax API |
| `PAGE-007` | 任务列表 | 刷新、选择 | MoneyPrinter task state |
| `PAGE-008` | 预览 | 播放、下载 | adapter 鉴权媒体 URL |

已删除的竞争入口：

- 不再显示“本地素材”上传/勾选列表。
- 不再显示默认 Hermes“视频素材库”上传/切分列表。
- 不再单独显示“Obsidian 具名资产库”卡片。
- Pexels / Pixabay / Coverr 没有被删除，统一收进“素材来源”下拉框。
- `storage/local_videos` 只由 AI 时间线桥接写入，用户不可将其当长期资产库管理。

## 3. `PAGE-005` 控件与状态

| 控件 | 未选库 | 已选库 | 行为 |
| --- | --- | --- | --- |
| 素材来源 | 可用 | 可用 | `local / pexels / pixabay / coverr`；在线来源不展示本地库管理区 |
| 资产库选择 | 可用，值为空 | 显示当前库 | 每次进入必须手动选择，不恢复草稿中的库 |
| 添加素材文件 | 禁用 | 可用 | 每个请求携带 `libraryId`，导入后分析 |
| 选择素材目录 | 禁用 | 可用 | 桌面原生目录选择，后端写入当前库 source roots |
| 扫描新增素材 | 禁用 | 可用 | 先 `dryRun=true`，出现确认扫描后才能真实写入 |
| 迁移旧素材 | 禁用 | 可用 | 显式 ConfirmDialog，按 SHA-256 去重，旧文件保留 |
| AI 自动匹配并生成视频 | 禁用 | 可用 | 逐段查询、自动避重、生成 timeline、桥接缓存并立即创建 MoneyPrinter 任务 |
| 自动匹配全部文案 | 禁用 | 可用 | 精修辅助入口，展示每段候选 |
| 替换为此镜头 | 无候选 | 可用 | 可选精修 segment → clip，不是生成前置条件 |
| 创建素材时间线 | 禁用 | AI 已选/精修后可用 | 仅用于单独重建时间线 |

切换或清空资产库会清空：候选、确认、scan preview、migration result、timeline 和 `form.localMaterials`。

## 4. 组件职责

| 组件/模块 | 职责 |
| --- | --- |
| `unified-material-library-panel.tsx` | 统一来源选择、一键自动成片、本地库管理、候选精修 |
| `use-named-video-library.ts` | 选库、管理、逐段检索、低匹配回退、自动选镜头和时间线 |
| `named-library-matching.ts` | 文案分段、候选状态、综合评分与素材去重选择 |
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
| 当前查询无候选 | 回退当前库通用可用镜头并继续，绝不跨库 |
| 当前库完全无可用镜头 | 停止本地自动成片并明确报错 |
| 部分分析失败 | 保留成功项，记录逐文件错误 |
| cache copy 失败 | 保留 timeline 和 AI 选择结果，不提交不完整素材集合 |
| MoneyPrinter sidecar 退出 | 返回 `MONEYPRINTER_UPSTREAM_UNREACHABLE`，恢复服务后可重试同一任务 |
| 旧库迁移部分失败 | 返回 imported/skipped/failed 和逐资产记录，可幂等重试 |

本地具名库模式不使用网络素材补洞；用户只有显式切换到 Pexels / Pixabay / Coverr 时才走在线来源。

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
| Video Studio UI/client/hooks | `npm run test:ui -- src/app/video-studio` | 44 passed |
| Desktop 类型 | `npm run typecheck` | 通过 |
| 工作树格式 | `git diff --check` | 通过 |

## 10. 2026-07-12 自动成片升级

- 单一“素材来源”选择器保留 `local / pexels / pixabay / coverr`。
- 本地模式一键执行：匹配全部文案 → 综合排序 → 避免重复 asset → 创建可追溯时间线 → 缓存桥接 → 创建视频任务。
- 在线模式直接创建对应 provider 的 MoneyPrinter 任务，不再错误要求选择本地库。
- 候选列表和“替换为此镜头”仍保留，但只用于可选精修。
- 正确开发版 `hermes-dev-desktop` 实机检查：来源菜单显示本地、Pexels、Pixabay、Coverr；Pexels 模式隐藏本地库管理且一键按钮可用；切回本地并选择 `beef-noodle` 后一键按钮可用。
- 升级后 Video Studio 自动化回归为 44 passed；素材库后端回归仍为 48 passed。

## 11. 已知边界

已实现的是“文案语义匹配 → AI 自动选镜头 → 来源可追溯时间线 → 自动渲染”，人工选择仅用于精修。MoneyPrinter 目前仍按 clip 时长和最大片段秒数覆盖旁白，不是逐字级音频时间戳驱动。下一阶段若追求导演级口播节奏，应让分句 TTS 或字幕时间戳回写 `voiceStart/voiceEnd`，再对 shot track 做二次裁剪。
