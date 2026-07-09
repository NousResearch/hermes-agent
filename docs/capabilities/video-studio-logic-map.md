# Video Studio 逻辑地图

> 状态：精细化控制实施版
> 范围：Hermes Desktop 原生 Video Studio 页面 + MoneyPrinterTurbo adapter API + vendored MoneyPrinterTurbo sidecar
> 约束：Video Studio 是 Desktop capability，不是 Hermes Agent Core 的一部分；所有媒体/密钥/任务状态都必须经过 capability adapter 或 sidecar 的安全边界。

## 0. 编号体系

| 前缀 | 类型 | 说明 |
| --- | --- | --- |
| `CAP-*` | 能力 | 用户可感知的产品能力 |
| `PAGE-*` | 页面 | Desktop 页面/区域 |
| `BTN-*` | 按钮 | 用户可点击控件 |
| `ACT-*` | 动作 | UI 或 API 触发的动作 |
| `FLOW-*` | 流程 | 多动作组成的业务链路 |
| `DATA-*` | 数据 | 表单、任务、输出、配置、资产等数据对象 |
| `API-*` | 接口 | Hermes adapter API 或上游 MoneyPrinter API |
| `RULE-*` | 规则 | 架构、安全、交互和边界规则 |
| `TEST-*` | 测试 | 自动/手动回归测试项 |

---

## 1. 能力地图

| ID | 能力 | 描述 | 页面 | 流程 | API | 测试 |
| --- | --- | --- | --- | --- | --- | --- |
| `CAP-001` | Video Studio 工作台 | 在 Hermes Desktop 内提供原生 React 视频生成页面 | `PAGE-001` | `FLOW-001` | `API-001`~`API-015` | `TEST-001`~`TEST-009` |
| `CAP-002` | MoneyPrinterTurbo sidecar | 将 `external/MoneyPrinterTurbo` 作为独立服务运行 | `PAGE-001` | `FLOW-002` | `API-001`, `API-002` | `TEST-003`, `TEST-006`, `TEST-009` |
| `CAP-003` | 高频视频生成链路 | 复刻主题、文案、关键词、语音、素材、字幕、合成、预览主链路 | `PAGE-002`~`PAGE-006` | `FLOW-003` | `API-003`~`API-007` | `TEST-001`, `TEST-002`, `TEST-004` |
| `CAP-004` | 任务状态管理 | 展示任务 id、状态、进度、错误和输出 | `PAGE-007` | `FLOW-008` | `API-004`, `API-005`, `API-006` | `TEST-004`, `TEST-007` |
| `CAP-005` | 输出预览 | 展示脚本、视频预览、下载入口 | `PAGE-008` | `FLOW-009` | `API-007` | `TEST-002`, `TEST-009` |
| `CAP-006` | API Key / provider 配置 | Desktop 表单同步 `config.toml`，密钥只写入 sidecar ignored config，不回显明文 | `PAGE-002` | `FLOW-004` | `API-008` | `TEST-004`, `TEST-009` |
| `CAP-007` | 本地素材上传/选择 | 上传/列出/选择 `storage/local_videos` 白名单素材 | `PAGE-005` | `FLOW-005` | `API-009`, `API-010` | `TEST-004`, `TEST-009` |
| `CAP-008` | BGM 资产上传/指定 | 上传 MP3 到 `resource/songs` 并可通过 `bgm_file` 指定 | `PAGE-006` | `FLOW-006` | `API-011`, `API-012` | `TEST-004`, `TEST-009` |
| `CAP-009` | 自定义音频 | 上传 audio 到 adapter 管理目录，并通过 `custom_audio_file` 跳过 TTS | `PAGE-006` | `FLOW-006`, `FLOW-007` | `API-011`, `API-013` | `TEST-004`, `TEST-009` |
| `CAP-010` | 独立音频/字幕任务 | 不合成视频，单独创建 `/audio` 或 `/subtitle` 任务 | `PAGE-006` | `FLOW-007` | `API-014`, `API-015` | `TEST-004`, `TEST-009` |
| `CAP-011` | 字体/voice 候选 | 从 sidecar/资源目录聚合 voice 与 font 列表，作为 datalist 候选 | `PAGE-006` | `FLOW-006` | `API-011` | `TEST-009` |

---

## 2. 页面地图

| ID | 页面/区域 | 文件 | 包含按钮 | 读写数据 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `PAGE-001` | Video Studio 页面壳 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-001`, `BTN-002` | `DATA-005` | 页面标题、流程说明、服务状态入口 |
| `PAGE-002` | API 配置面板 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-003`, `BTN-004` | `DATA-006` | provider/model/baseUrl/API keys，多 key 文本可写入 config.toml |
| `PAGE-003` | 主题 → 文案 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-005` | `DATA-001`, `DATA-007` | 主题、语言、段落数、文案 prompt、system prompt、可编辑文案 |
| `PAGE-004` | 文案 → 素材关键词 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-006` | `DATA-001`, `DATA-008` | 关键词数量、顺序匹配、可编辑逐行 terms |
| `PAGE-005` | 素材参数/本地素材 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-007`, `BTN-008` | `DATA-001`, `DATA-009` | 比例、数量、素材来源、clip 秒数、concat/transition、本地素材上传/选择 |
| `PAGE-006` | 语音、字幕、BGM、合成细节 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-009`~`BTN-014` | `DATA-001`, `DATA-010` | voice/font 候选、音量/语速、字幕样式、BGM、自定义音频、音频/字幕/视频任务按钮 |
| `PAGE-007` | 任务列表 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-015`, `BTN-016` | `DATA-003` | 展示和选择 MoneyPrinter 任务 |
| `PAGE-008` | 预览面板 | `apps/desktop/src/app/video-studio/index.tsx` | `BTN-017` | `DATA-003`, `DATA-004` | 展示脚本、状态、错误、视频输出 |
| `PAGE-009` | 左侧导航入口 | `apps/desktop/src/app/chat/sidebar/index.tsx` | `BTN-018` | `DATA-011` | Video Studio nav item |

---

## 3. 按钮地图

| ID | 按钮 | 所在页面 | 触发动作 | 接口 | 规则 |
| --- | --- | --- | --- | --- | --- |
| `BTN-001` | Health Check | `PAGE-001` | `ACT-001` | `API-001` | `RULE-001`, `RULE-004` |
| `BTN-002` | Start Service | `PAGE-001` | `ACT-002` | `API-002` | `RULE-002`, `RULE-004` |
| `BTN-003` | Reload config | `PAGE-002` | `ACT-003` | `API-008` | `RULE-007`, `RULE-008` |
| `BTN-004` | Save config | `PAGE-002` | `ACT-004` | `API-008` | `RULE-007`, `RULE-008` |
| `BTN-005` | 生成文案 | `PAGE-003` | `ACT-005` | `API-016` | `RULE-003`, `RULE-004` |
| `BTN-006` | 生成关键词 | `PAGE-004` | `ACT-006` | `API-017` | `RULE-003`, `RULE-004` |
| `BTN-007` | Refresh local materials | `PAGE-005` | `ACT-007` | `API-009` | `RULE-006` |
| `BTN-008` | Upload local material | `PAGE-005` | `ACT-008` | `API-010` | `RULE-005`, `RULE-006` |
| `BTN-009` | Refresh assets | `PAGE-006` | `ACT-009` | `API-011` | `RULE-006` |
| `BTN-010` | Upload MP3 | `PAGE-006` | `ACT-010` | `API-012` | `RULE-005`, `RULE-006` |
| `BTN-011` | Upload audio | `PAGE-006` | `ACT-011` | `API-013` | `RULE-005`, `RULE-006` |
| `BTN-012` | Generate Audio | `PAGE-006` | `ACT-012` | `API-014` | `RULE-003`, `RULE-004` |
| `BTN-013` | Generate Subtitle | `PAGE-006` | `ACT-013` | `API-015` | `RULE-003`, `RULE-004` |
| `BTN-014` | Generate Video | `PAGE-006` | `ACT-014` | `API-003` | `RULE-003`, `RULE-004`, `RULE-006` |
| `BTN-015` | Refresh tasks | `PAGE-007` | `ACT-015` | `API-004` | `RULE-004` |
| `BTN-016` | Task Row | `PAGE-007` | `ACT-016` | 无 | `RULE-009` |
| `BTN-017` | Download | `PAGE-008` | `ACT-017` | adapter download URL | `RULE-006` |
| `BTN-018` | Sidebar Video Studio | `PAGE-009` | `ACT-018` | 无 | `RULE-001` |

---

## 4. 动作地图

| ID | 动作 | 输入 | 输出 | 触发方 | 后续流程 |
| --- | --- | --- | --- | --- | --- |
| `ACT-001` | 查询 MoneyPrinter 健康状态 | 无 | `DATA-005` | `BTN-001` | `FLOW-002` |
| `ACT-002` | 请求启动 MoneyPrinter 服务 | 无 | `DATA-005` | `BTN-002` | `FLOW-002` |
| `ACT-003` | 读取 config 摘要 | 无 | `DATA-006` | `BTN-003` / 页面加载 | `FLOW-004` |
| `ACT-004` | 保存 config | `DATA-006` | updated `DATA-006` | `BTN-004` | `FLOW-004` |
| `ACT-005` | 生成文案草稿 | `DATA-007` | `video_script` 回填 `DATA-001` | `BTN-005` | `FLOW-003` |
| `ACT-006` | 生成素材关键词 | `DATA-008` | `video_terms` 回填 `DATA-001` | `BTN-006` | `FLOW-003` |
| `ACT-007` | 刷新本地素材 | 无 | `DATA-009[]` | `BTN-007` | `FLOW-005` |
| `ACT-008` | 上传本地素材 | File/sourcePath/dataURL | `DATA-009` | `BTN-008` | `FLOW-005` |
| `ACT-009` | 刷新资产 | 无 | `DATA-010` | `BTN-009` / 页面加载 | `FLOW-006` |
| `ACT-010` | 上传 BGM | MP3 File/sourcePath/dataURL | BGM asset | `BTN-010` | `FLOW-006` |
| `ACT-011` | 上传自定义音频 | audio File/sourcePath/dataURL | custom audio asset | `BTN-011` | `FLOW-006` |
| `ACT-012` | 创建 audio-only 任务 | `DATA-001` | `DATA-003` | `BTN-012` | `FLOW-007`, `FLOW-008` |
| `ACT-013` | 创建 subtitle-only 任务 | `DATA-001` | `DATA-003` | `BTN-013` | `FLOW-007`, `FLOW-008` |
| `ACT-014` | 创建完整视频任务 | `DATA-001` | `DATA-003` | `BTN-014` | `FLOW-003`, `FLOW-008` |
| `ACT-015` | 刷新任务列表 | 无 | `DATA-003[]` | `BTN-015` | `FLOW-008` |
| `ACT-016` | 选择任务 | task id | selected task | `BTN-016` | `FLOW-009` |
| `ACT-017` | 下载输出 | output url | media file | `BTN-017` | `FLOW-009` |
| `ACT-018` | 打开 Video Studio 页面 | route | `/video-studio` | `BTN-018` | `FLOW-001` |

---

## 5. 流程地图

### `FLOW-001` 页面进入流程

```text
BTN-018 → ACT-018 → routes.ts 解析 /video-studio → DesktopController lazy load VideoStudioView → PAGE-001
```

相关对象：`DATA-011`
规则：`RULE-001`

### `FLOW-002` 服务健康/启动流程

```text
BTN-001/BTN-002 → ACT-001/ACT-002 → API-001/API-002 → capabilities.moneyprinter.adapter → external/MoneyPrinterTurbo
```

相关对象：`DATA-005`
规则：`RULE-002`, `RULE-004`

### `FLOW-003` 完整视频生成流程

```text
PAGE-003 主题/文案 → BTN-005 可选生成文案 → 人工改稿 → BTN-006 可选生成 terms → PAGE-005/PAGE-006 精调参数 → BTN-014 → API-003 → MoneyPrinter /api/v1/videos → task id → DATA-003
```

MoneyPrinter 内部链路：

```text
主题/最终文案 → terms → TTS 或 custom_audio_file → 素材下载/本地素材 → 字幕 → BGM → moviepy/ffmpeg 合成 → outputs
```

相关对象：`DATA-001`, `DATA-002`, `DATA-003`, `DATA-004`, `DATA-009`, `DATA-010`
规则：`RULE-003`, `RULE-005`, `RULE-006`

### `FLOW-004` 配置读写流程

```text
页面加载/Reload → API-008 GET → config summary → PAGE-002
Save → API-008 POST → external/MoneyPrinterTurbo/config.toml → summary 回填；secret 输入框清空但保存状态保留
```

相关对象：`DATA-006`
规则：`RULE-007`, `RULE-008`

### `FLOW-005` 本地素材流程

```text
Refresh local materials → API-009 → storage/local_videos 列表
Upload local material → API-010 → 白名单扩展名 + 目录归一化 → storage/local_videos → 自动切换 video_source=local 并选中新素材
```

相关对象：`DATA-009`, `DATA-001.video_materials`
规则：`RULE-005`, `RULE-006`

### `FLOW-006` BGM / custom audio / assets 流程

```text
页面加载/Refresh assets → API-011 → voices + fonts + resource/songs bgms + storage/custom_audio 列表
Upload MP3 → API-012 → resource/songs → bgm_file 回填 + bgm_type=custom
Upload audio → API-013 → storage/custom_audio → custom_audio_file 回填
```

相关对象：`DATA-010`, `DATA-001.bgm_file`, `DATA-001.custom_audio_file`
规则：`RULE-005`, `RULE-006`

### `FLOW-007` 独立音频/字幕流程

```text
Generate Audio → API-014 → MoneyPrinter /api/v1/audio → task id
Generate Subtitle → API-015 → MoneyPrinter /api/v1/subtitle → task id
```

说明：`custom_audio_file` 可用于 audio/subtitle-only 请求；使用自定义音频时 TTS 被跳过，Whisper subtitle provider 可从音频转写字幕。
相关对象：`DATA-001`, `DATA-003`
规则：`RULE-003`, `RULE-004`, `RULE-006`

### `FLOW-008` 任务刷新流程

```text
BTN-015 → ACT-015 → API-004 → MoneyPrinter /api/v1/tasks → adapter normalize → DATA-003[] → PAGE-007
```

相关对象：`DATA-003`
规则：`RULE-004`, `RULE-009`

### `FLOW-009` 预览流程

```text
BTN-016 → ACT-016 → selectedTask → PAGE-008 → video.streamUrl/downloadUrl → resolveMoneyPrinterMediaUrl → video preview / BTN-017
```

相关对象：`DATA-003`, `DATA-004`
规则：`RULE-006`

---

## 6. 数据地图

| ID | 数据 | TypeScript/Python 位置 | 生产方 | 消费方 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `DATA-001` | `VideoGenerationForm` | `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `PAGE-003`~`PAGE-006` | `ACT-012`~`ACT-014` | Hermes UI 表单模型，camelCase，包含 BGM/custom audio/local material |
| `DATA-002` | `CreateVideoPayload` | `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `toCreateVideoPayload` | `API-003`, `API-014`, `API-015` | MoneyPrinter 请求模型，snake_case |
| `DATA-003` | `MoneyPrinterTask` | `moneyprinter-client.ts` / `capabilities/moneyprinter/adapter.py` | adapter | `PAGE-007`, `PAGE-008` | UI 稳定任务模型 |
| `DATA-004` | `MoneyPrinterVideoOutput` | `moneyprinter-client.ts` / `adapter.py` | adapter | `PAGE-008` | 视频文件、stream/download URL |
| `DATA-005` | `MoneyPrinterHealth` | `moneyprinter-client.ts` / `adapter.py` | `API-001`, `API-002` | `PAGE-001`, `PAGE-008` | 服务安装、运行、storage、upstream commit |
| `DATA-006` | `MoneyPrinterConfigInput` / summary | `moneyprinter-client.ts` / `adapter.py` | `PAGE-002`, `API-008` | sidecar config | provider/model/baseUrl/key configured flags |
| `DATA-007` | `GenerateScriptPayload` | `moneyprinter-client.ts` | `PAGE-003` | `API-016` | 主题/语言/段落/prompt/system prompt |
| `DATA-008` | `GenerateTermsPayload` | `moneyprinter-client.ts` | `PAGE-004` | `API-017` | 关键词数量、match_materials_to_script、文案、主题 |
| `DATA-009` | `MoneyPrinterLocalMaterial` | `moneyprinter-client.ts` / `adapter.py` | `API-009`, `API-010` | `PAGE-005`, `DATA-002.video_materials` | local video/image asset |
| `DATA-010` | `MoneyPrinterAssets` | `moneyprinter-client.ts` / `adapter.py` | `API-011`~`API-013` | `PAGE-006` | voices/fonts/bgms/customAudio |
| `DATA-011` | route metadata | `apps/desktop/src/app/routes.ts` | Desktop route config | Sidebar / Controller | `/video-studio` 路由、`video-studio` view id |

---

## 7. 接口地图

| ID | Hermes Adapter API | 方法 | 上游/后端 | 使用方 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `API-001` | `/api/capabilities/moneyprinter/health` | `GET` | MoneyPrinter `/docs` reachability + local checks | `BTN-001` | 已实现 |
| `API-002` | `/api/capabilities/moneyprinter/service/start` | `POST` | local process start | `BTN-002` | 已实现 |
| `API-003` | `/api/capabilities/moneyprinter/videos` | `POST` | `/api/v1/videos` | `BTN-014` | 已实现 |
| `API-004` | `/api/capabilities/moneyprinter/tasks` | `GET` | `/api/v1/tasks` + disk fallback | `BTN-015` | 已实现 |
| `API-005` | `/api/capabilities/moneyprinter/tasks/{task_id}` | `GET` | `/api/v1/tasks/{task_id}` + disk fallback | task polling/detail | 已实现 |
| `API-006` | `/api/capabilities/moneyprinter/tasks/{task_id}` | `DELETE` | `/api/v1/tasks/{task_id}` | 后续删除按钮/MCP | 已实现 |
| `API-007` | `/api/capabilities/moneyprinter/{stream|download}/{file_path}` | `GET` | `/api/v1/stream`, `/api/v1/download` | `BTN-017` | 已实现代理 |
| `API-008` | `/api/capabilities/moneyprinter/config` | `GET/POST` | ignored `config.toml` | `BTN-003`, `BTN-004` | 已实现；密钥不回传 |
| `API-009` | `/api/capabilities/moneyprinter/materials` | `GET` | `storage/local_videos` | `BTN-007` | 已实现 |
| `API-010` | `/api/capabilities/moneyprinter/materials` | `POST` | `storage/local_videos` | `BTN-008` | 已实现；扩展名/路径白名单 |
| `API-011` | `/api/capabilities/moneyprinter/assets` | `GET` | voices/fonts/songs/custom_audio | `BTN-009` | 已实现 |
| `API-012` | `/api/capabilities/moneyprinter/bgms` | `POST` | `resource/songs` | `BTN-010` | 已实现；MP3 白名单 |
| `API-013` | `/api/capabilities/moneyprinter/custom-audio` | `POST` | `storage/custom_audio` | `BTN-011` | 已实现；audio 扩展名白名单 |
| `API-014` | `/api/capabilities/moneyprinter/audio` | `POST` | `/api/v1/audio` | `BTN-012` | 已实现 |
| `API-015` | `/api/capabilities/moneyprinter/subtitle` | `POST` | `/api/v1/subtitle` | `BTN-013` | 已实现 |
| `API-016` | `/api/capabilities/moneyprinter/scripts` | `POST` | `/api/v1/scripts` | `BTN-005` | 已实现 |
| `API-017` | `/api/capabilities/moneyprinter/terms` | `POST` | `/api/v1/terms` | `BTN-006` | 已实现 |

---

## 8. 规则地图

| ID | 规则 | 说明 | 覆盖对象 |
| --- | --- | --- | --- |
| `RULE-001` | 不触碰 Hermes Agent Core | Video Studio 只新增 Desktop route/page/API adapter，不改 Agent 主循环和核心 tool-call 架构 | `CAP-001`, `PAGE-009` |
| `RULE-002` | MoneyPrinterTurbo 独立 sidecar | 上游项目运行在独立 Python 进程，Hermes 只代理 API | `CAP-002`, `API-002` |
| `RULE-003` | UI 模型与上游 payload 分离 | `VideoGenerationForm` 使用 UI 友好字段，提交前转换为 MoneyPrinter snake_case | `DATA-001`, `DATA-002`, `TEST-001` |
| `RULE-004` | adapter 统一错误 envelope | Desktop 页面只消费 `{ok,data,error}`，不直接暴露上游错误结构 | `API-001`~`API-017` |
| `RULE-005` | 上传文件必须白名单 | local material、BGM、custom audio 只允许支持扩展名，不接受任意文件 | `API-010`, `API-012`, `API-013` |
| `RULE-006` | 路径必须目录归一化 | renderer 不直接读任意本地路径；server-side file 先复制到白名单目录或返回 adapter URL | `DATA-004`, `DATA-009`, `DATA-010`, `API-007`, `API-010`~`API-013` |
| `RULE-007` | 密钥不进入 repo/聊天 | API keys 只写入 ignored `config.toml`；summary 只返回 configured/missing，不回传明文 | `DATA-006`, `API-008` |
| `RULE-008` | 空密钥不清空已保存密钥 | Desktop 密码输入为空表示沿用旧值，不覆盖 config.toml 中已有 secret | `PAGE-002`, `API-008` |
| `RULE-009` | 任务状态前端只展示 adapter 稳定模型 | 上游 task shape 变化由 adapter 吸收 | `DATA-003` |
| `RULE-010` | 自定义音频语义 | 设置 `custom_audio_file` 时跳过 TTS；字幕若需从音频转写应使用 Whisper provider | `CAP-009`, `CAP-010` |

---

## 9. 测试地图

| ID | 测试 | 命令/方式 | 覆盖能力 | 当前结果 |
| --- | --- | --- | --- | --- |
| `TEST-001` | UI 表单到 MoneyPrinter payload 映射 | `npm run --workspace apps/desktop test:ui -- src/app/video-studio/moneyprinter-client.test.ts` | `DATA-001`, `DATA-002`, `RULE-003` | 通过：8 tests |
| `TEST-002` | capability API path / media URL namespace | 同上 | `API-001`~`API-017`, `RULE-004`, `RULE-006` | 通过 |
| `TEST-003` | Python adapter targeted tests | `scripts/run_tests.sh tests/capabilities/test_moneyprinter_adapter.py` | adapter defaults、assets、upload、安全路径 | 通过：9 tests |
| `TEST-004` | Desktop typecheck | `npm run --workspace apps/desktop typecheck` | `PAGE-001`~`PAGE-009`, TS model | 通过 |
| `TEST-005` | Desktop targeted lint | `npm exec --workspace apps/desktop eslint -- src/app/video-studio/index.tsx src/app/video-studio/moneyprinter-client.ts src/app/video-studio/moneyprinter-client.test.ts` | 新 TS/TSX 风格 | 通过 |
| `TEST-006` | Python py_compile | `python3 -m py_compile capabilities/moneyprinter/adapter.py gateway/platforms/api_server.py hermes_cli/web_server.py external/MoneyPrinterTurbo/app/models/schema.py` | Python syntax / imports | 通过 |
| `TEST-007` | Gateway/API server route smoke | curl or in-process API server | `API-001`~`API-017` | 待运行 |
| `TEST-008` | Desktop build | `npm run --workspace apps/desktop build` | bundle/type integration | 通过；dirty-tree/CSS/chunk-size warning 为既有构建警告 |
| `TEST-009` | 人工点击式 Desktop QA | `computer_use` 操作 Desktop app：导航、输入、下拉、上传、按钮、任务/预览 | `CAP-001`~`CAP-011` | 待运行 |

---

## 10. 当前实现文件清单

| 文件 | 编号覆盖 | 说明 |
| --- | --- | --- |
| `external/MoneyPrinterTurbo/` | `CAP-002`, `CAP-009`, `CAP-010` | vendored upstream source；`AudioRequest`/`SubtitleRequest` 支持 `custom_audio_file` |
| `capabilities/moneyprinter/upstream.json` | `CAP-002`, `RULE-002` | upstream 元数据 |
| `capabilities/moneyprinter/adapter.py` | `API-001`~`API-017`, `RULE-004`~`RULE-010` | Hermes adapter API、assets、upload、安全路径、audio/subtitle/video 代理 |
| `hermes_cli/web_server.py` | `API-001`~`API-017` | Desktop `hermes serve` / dashboard backend route 注册 |
| `gateway/platforms/api_server.py` | `API-001`~`API-017` | 注册 capability API routes |
| `apps/desktop/src/app/routes.ts` | `DATA-011`, `PAGE-009` | `/video-studio` route/view |
| `apps/desktop/src/app/chat/sidebar/index.tsx` | `BTN-018`, `PAGE-009` | sidebar 入口 |
| `apps/desktop/src/app/desktop-controller.tsx` | `FLOW-001` | lazy load Video Studio page |
| `apps/desktop/src/app/video-studio/moneyprinter-client.ts` | `DATA-001`~`DATA-010` | 前端 API client 和模型转换 |
| `apps/desktop/src/app/video-studio/index.tsx` | `PAGE-001`~`PAGE-008` | React 页面、表单、上传、任务、预览 |
| `apps/desktop/src/app/video-studio/moneyprinter-client.test.ts` | `TEST-001`, `TEST-002` | Vitest 回归测试 |
| `tests/capabilities/test_moneyprinter_adapter.py` | `TEST-003` | Python adapter 回归测试 |
