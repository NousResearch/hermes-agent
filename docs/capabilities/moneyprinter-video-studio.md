# MoneyPrinterTurbo Video Studio 集成专项设计文档

> 状态：Phase 0–3 已落地（adapter API + Desktop + MCP/skill）；高级功能仍待补齐  
> 目标阶段：持续完善  
> 目标读者：Hermes Desktop / Agent / Capability 开发者  
> 关联原则：`docs/hermes-desktop-capability-architecture.md`

## 1. 背景与目标

我们计划把开源项目 [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) 内嵌为 Hermes Desktop 的一个视频生成能力页面，并同时暴露给 Hermes 智能体调用。

用户确认采用以下方向：

1. 新增 Hermes Desktop 原生 React 页面：`Video Studio`。
2. 第三方项目放在仓库的 `external/MoneyPrinterTurbo/` 下。
3. 采用方案 B：用 Hermes 原生 React 页面复刻 MoneyPrinterTurbo 的高频功能。
4. 首期复刻约 70% 高频链路：
   - 主题
   - 文案
   - 语音
   - 素材
   - 字幕
   - 合成
   - 预览
5. 后续逐步补齐高级能力：
   - 自定义 prompt
   - 本地素材
   - BGM 上传
   - 字幕样式
   - 社媒元数据
   - cross-post
6. MoneyPrinterTurbo 应作为独立 capability 内嵌，而不是变成 Hermes Agent Core 的一部分。

最终产品形态：

```text
Hermes Desktop Video Studio 页面
        ↓
Hermes MoneyPrinter Adapter API / service manager
        ↓
external/MoneyPrinterTurbo FastAPI service
        ↓
MoneyPrinterTurbo 视频生成流水线
        ↓
Hermes-managed outputs / artifacts
```

智能体调用形态：

```text
Hermes Agent Core
        ↓
Hermes MCP Client
        ↓
moneyprinter-mcp server
        ↓
Hermes MoneyPrinter Adapter API
        ↓
external/MoneyPrinterTurbo FastAPI service
```

关键约束：**不修改 Hermes Agent 主循环、不扩大核心 tool schema、不把 MoneyPrinterTurbo 业务逻辑塞进 Agent Core。**

---

## 2. MoneyPrinterTurbo 当前架构分析

基于已拉取的 MoneyPrinterTurbo 仓库版本：

```text
commit: 63113a3
```

MoneyPrinterTurbo 是一个 Python 视频生成项目，包含三层：

```text
MoneyPrinterTurbo
├── FastAPI API 服务
├── Streamlit WebUI
└── 视频生成业务流水线
```

### 2.1 关键入口

| 能力 | 文件 | 说明 |
|---|---|---|
| FastAPI 服务入口 | `main.py` | 启动 MoneyPrinterTurbo API 服务 |
| ASGI app | `app/asgi.py` | 创建 FastAPI app、挂载路由 |
| 路由聚合 | `app/router.py` | 注册 v1 controllers |
| 视频 API | `app/controllers/v1/video.py` | 创建视频任务、查询任务、上传素材、下载/流式播放产物 |
| LLM API | `app/controllers/v1/llm.py` | 生成脚本、关键词、社媒元数据 |
| 请求/响应模型 | `app/models/schema.py` | Pydantic schema |
| 核心任务流水线 | `app/services/task.py` | 文案、关键词、音频、字幕、素材、合成 |
| Streamlit UI | `webui/Main.py` | 原始人工操作界面 |
| 配置样例 | `config.example.toml` | LLM、素材、TTS、字幕等配置 |
| Docker | `Dockerfile`, `docker-compose.yml` | API/WebUI 容器启动方式 |

### 2.2 现有 API 能力

MoneyPrinterTurbo 已经自带 FastAPI 接口，可直接作为 Hermes adapter 的下游服务。

#### 视频任务

```http
POST /api/v1/videos
GET  /api/v1/tasks
GET  /api/v1/tasks/{task_id}
DELETE /api/v1/tasks/{task_id}
```

#### 分阶段生成

```http
POST /api/v1/audio
POST /api/v1/subtitle
POST /api/v1/scripts
POST /api/v1/terms
POST /api/v1/social-metadata
```

#### 素材 / BGM 管理

```http
GET  /api/v1/musics
POST /api/v1/musics
GET  /api/v1/video_materials
POST /api/v1/video_materials
```

#### 输出播放 / 下载

```http
GET /api/v1/stream/{file_path}
GET /api/v1/download/{file_path}
```

### 2.3 视频生成流水线

核心入口：`app/services/task.py:start(...)`

流水线：

```text
1. generate_script
   根据主题和语言生成视频文案，或使用用户提供的文案。

2. generate_terms
   根据主题和文案生成素材搜索关键词。

3. generate_audio
   使用 TTS 生成语音，或使用自定义音频。

4. generate_subtitle
   使用 Edge subtitle timeline 或 Whisper 生成字幕。

5. get_video_materials
   使用 Pexels / Pixabay / Coverr 下载素材，或使用本地素材。

6. generate_final_videos
   拼接素材、合成音频、字幕、BGM，输出 final mp4。

7. optional cross_post
   可选发布到社交平台。
```

### 2.4 主要依赖

Python 依赖包括：

```text
moviepy
streamlit
edge_tts
fastapi
uvicorn
openai
faster-whisper
loguru
google.generativeai
dashscope
azure-cognitiveservices-speech
redis
python-multipart
requests
pydub
litellm
```

系统依赖包括：

```text
ffmpeg
imagemagick
```

### 2.5 License

MoneyPrinterTurbo 当前使用 MIT License。内嵌和分发时需要：

1. 保留原始 `LICENSE`。
2. 在 Hermes 的 third-party notices 或 capability 页面中标注来源。
3. 如果 vendor 到 `external/MoneyPrinterTurbo/`，保留 upstream commit 信息。
4. 不把上游项目的真实 API key、用户配置或本地生成文件提交进仓库。

---

## 3. Hermes 集成边界

### 3.1 不做的事情

以下内容明确禁止：

```text
不修改 run_agent.py 主循环。
不修改 Agent 消息循环、tool-call 分发协议或系统 prompt 构造逻辑。
不把 MoneyPrinterTurbo 注册成 Hermes core tool。
不把 MoneyPrinterTurbo 的大量参数塞进默认系统 prompt。
不让 Agent 直接操作 moviepy / ffmpeg / imagemagick 细节。
不让 Desktop renderer 直接读写任意本地路径。
不把第三方 API key 写入源码、文档示例真实值或日志。
```

### 3.2 可以做的事情

```text
新增 external/MoneyPrinterTurbo/。
新增 capabilities/moneyprinter/ 作为 Hermes adapter/service/mcp 层。
新增 apps/desktop/src/app/video-studio/ 原生 React 页面。
新增 /video-studio 路由和侧边栏入口。
新增 moneyprinter service manager，负责启动、停止、健康检查。
新增 adapter API，统一转发 MoneyPrinterTurbo FastAPI。
新增 MCP server，供 Hermes Agent 调用。
新增 skill，教 Agent 如何使用视频能力。
新增输出目录和 artifact 映射。
新增配置 UI，用于设置 provider、素材 API、TTS、路径等。
```

### 3.3 架构原则

```text
MoneyPrinterTurbo 是 Desktop capability，不是 Agent Core。

Agent 调 MCP。
Desktop 页面调 Adapter API。
Adapter API 调 MoneyPrinterTurbo FastAPI。
MoneyPrinterTurbo 独立进程负责真实视频生成。
Hermes 负责页面、配置、生命周期、权限和输出管理。
```

---

## 4. 目标目录结构

建议新增结构：

```text
hermes-agent/
├── external/
│   └── MoneyPrinterTurbo/
│       ├── README-en.md
│       ├── LICENSE
│       ├── main.py
│       ├── app/
│       ├── webui/
│       ├── config.example.toml
│       └── ...
│
├── capabilities/
│   └── moneyprinter/
│       ├── README.md
│       ├── upstream.json
│       ├── config/
│       │   ├── default-config.toml
│       │   └── schema.json
│       ├── service/
│       │   ├── moneyprinter_service.py
│       │   ├── config_manager.py
│       │   ├── health.py
│       │   └── paths.py
│       ├── adapter/
│       │   ├── api.py
│       │   ├── client.py
│       │   └── models.py
│       ├── mcp/
│       │   ├── server.py
│       │   └── tools.py
│       └── tests/
│           ├── test_config_manager.py
│           ├── test_adapter_models.py
│           └── test_mcp_tools.py
│
├── apps/desktop/src/app/video-studio/
│   ├── index.tsx
│   ├── video-studio-route.tsx
│   ├── moneyprinter-client.ts
│   ├── types.ts
│   ├── components/
│   │   ├── generation-form.tsx
│   │   ├── task-list.tsx
│   │   ├── video-preview.tsx
│   │   ├── service-status-card.tsx
│   │   ├── script-panel.tsx
│   │   ├── voice-panel.tsx
│   │   ├── material-panel.tsx
│   │   └── subtitle-panel.tsx
│   └── hooks/
│       ├── use-moneyprinter-service.ts
│       ├── use-moneyprinter-tasks.ts
│       └── use-video-generation.ts
│
├── skills/
│   └── moneyprinter-video/
│       └── SKILL.md
│
└── docs/capabilities/
    └── moneyprinter-video-studio.md
```

说明：

- `external/MoneyPrinterTurbo/`：上游源码，尽量减少本地 patch。
- `capabilities/moneyprinter/`：Hermes 自己维护的薄集成层。
- `apps/desktop/src/app/video-studio/`：Hermes 原生 React 页面。
- `skills/moneyprinter-video/`：让 Agent 知道如何使用 MCP 工具生成视频。
- `docs/capabilities/moneyprinter-video-studio.md`：本设计文档。

---

## 5. Desktop 页面设计

### 5.1 路由

当前 Desktop 路由文件：

```text
apps/desktop/src/app/routes.ts
```

新增：

```ts
export const VIDEO_STUDIO_ROUTE = '/video-studio'
```

扩展 `AppView`：

```ts
| 'video-studio'
```

扩展 `AppRouteId`：

```ts
| 'video-studio'
```

扩展 `APP_ROUTES`：

```ts
{ id: 'video-studio', path: VIDEO_STUDIO_ROUTE, view: 'video-studio' }
```

`Video Studio` 是主页面，不是 overlay。它应和 `skills`、`messaging`、`artifacts` 一样作为主 PaneMain route 渲染。

### 5.2 Desktop Controller 挂载点

当前路由渲染位置：

```text
apps/desktop/src/app/desktop-controller.tsx
```

在 `<Routes>` 中新增：

```tsx
<Route
  element={
    <Suspense fallback={null}>
      <VideoStudioView setStatusbarItemGroup={setStatusbarItemGroup} />
    </Suspense>
  }
  path="video-studio"
/>
```

### 5.3 侧边栏入口

当前侧边栏导航定义：

```text
apps/desktop/src/app/chat/sidebar/index.tsx
```

当前 `SIDEBAR_NAV` 包含：

```text
new-session
skills
messaging
artifacts
```

新增：

```ts
{
  id: 'video-studio',
  label: '',
  icon: props => <Codicon name="device-camera-video" {...props} />,
  route: VIDEO_STUDIO_ROUTE
}
```

如果 Codicon 没有 `device-camera-video`，备选：

```text
play-circle
record
file-media
symbol-event
```

### 5.4 页面布局

首期页面采用三栏布局：

```text
┌──────────────────────────────────────────────────────────────┐
│ Video Studio                                                  │
├───────────────────┬──────────────────────┬───────────────────┤
│ Generation Form   │ Task Queue / Logs     │ Preview / Outputs │
│                   │                      │                   │
│ 主题              │ 当前任务              │ 视频播放器         │
│ 文案              │ 历史任务              │ 下载 / 打开目录    │
│ 语言              │ 状态 / 进度           │ 复制路径           │
│ 比例              │ 错误信息              │ 发给 Agent         │
│ 语音              │                      │                   │
│ 素材源            │                      │                   │
│ 字幕开关          │                      │                   │
│ 生成按钮          │                      │                   │
└───────────────────┴──────────────────────┴───────────────────┘
```

### 5.5 首期字段范围

首期只复刻高频字段。

#### 基础字段

| UI 字段 | MoneyPrinter 参数 | 默认值 |
|---|---|---|
| 视频主题 | `video_subject` | 空 |
| 视频文案 | `video_script` | 空，允许自动生成 |
| 语言 | `video_language` | auto / 空 |
| 视频比例 | `video_aspect` | `9:16` |
| 视频数量 | `video_count` | `1` |
| 单段素材时长 | `video_clip_duration` | `5` |
| 素材来源 | `video_source` | `pexels` |
| 按文案顺序匹配素材 | `match_materials_to_script` | `false` |

#### 语音字段

| UI 字段 | MoneyPrinter 参数 | 默认值 |
|---|---|---|
| 语音 | `voice_name` | `zh-CN-XiaoxiaoNeural-Female` 或服务返回默认 |
| 语速 | `voice_rate` | `1.0` |
| 语音音量 | `voice_volume` | `1.0` |

#### 字幕字段

| UI 字段 | MoneyPrinter 参数 | 默认值 |
|---|---|---|
| 启用字幕 | `subtitle_enabled` | `true` |
| 字幕位置 | `subtitle_position` | `bottom` |
| 字体 | `font_name` | 默认字体 |
| 字号 | `font_size` | `60` |
| 字体颜色 | `text_fore_color` | `#FFFFFF` |
| 描边颜色 | `stroke_color` | `#000000` |
| 描边宽度 | `stroke_width` | `1.5` |

#### BGM 首期简化

首期可以只支持：

```text
bgm_type = "random"
bgm_volume = 0.2
```

高级阶段再做 BGM 上传和选择。

### 5.6 任务状态

MoneyPrinter 返回 task_id 后，Desktop 每 1-3 秒轮询：

```http
GET /capabilities/moneyprinter/tasks/{task_id}
```

UI 状态映射：

| MoneyPrinter state | UI 状态 |
|---|---|
| queued / pending | 排队中 |
| processing | 生成中 |
| complete | 已完成 |
| failed | 失败 |
| unknown | 未知 |

显示内容：

```text
task_id
state
progress
script
terms
audio_file
subtitle_path
materials
videos
combined_videos
error/message
```

### 5.7 预览与输出

完成后优先使用 adapter 返回的 Hermes-safe stream URL：

```text
/capabilities/moneyprinter/stream/{task_id}/{file_name}
```

不要让 renderer 直接使用任意绝对路径。

页面按钮：

```text
播放
下载
打开输出目录
复制输出路径
复制 task_id
发送到当前聊天作为上下文引用（后续阶段）
```

---

## 6. Adapter API 设计

MoneyPrinterTurbo 已有 API，但 Desktop 不应直接裸调 `127.0.0.1:8080/api/v1/...`。

原因：

1. 隔离上游 API 变化。
2. 统一 Hermes 权限和路径策略。
3. 统一错误格式。
4. 输出路径可以映射到 Hermes artifacts。
5. 后续换视频引擎时，Desktop 页面不需要重写。
6. Agent MCP 和 Desktop 页面可以复用同一 adapter。

### 6.1 Adapter API 路径

建议 Hermes adapter 暴露：

```http
GET    /capabilities/moneyprinter/health
POST   /capabilities/moneyprinter/service/start
POST   /capabilities/moneyprinter/service/stop
GET    /capabilities/moneyprinter/config
PUT    /capabilities/moneyprinter/config
POST   /capabilities/moneyprinter/videos
POST   /capabilities/moneyprinter/scripts
POST   /capabilities/moneyprinter/terms
GET    /capabilities/moneyprinter/tasks
GET    /capabilities/moneyprinter/tasks/{task_id}
DELETE /capabilities/moneyprinter/tasks/{task_id}
GET    /capabilities/moneyprinter/outputs
GET    /capabilities/moneyprinter/stream/{task_id}/{file_name}
GET    /capabilities/moneyprinter/download/{task_id}/{file_name}
```

高级阶段新增：

```http
GET    /capabilities/moneyprinter/materials
POST   /capabilities/moneyprinter/materials
GET    /capabilities/moneyprinter/bgms
POST   /capabilities/moneyprinter/bgms
POST   /capabilities/moneyprinter/social-metadata
POST   /capabilities/moneyprinter/cross-post
```

### 6.2 API 响应格式

Hermes adapter 统一返回：

```json
{
  "ok": true,
  "data": {},
  "error": null
}
```

错误：

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "MONEYPRINTER_SERVICE_UNAVAILABLE",
    "message": "MoneyPrinterTurbo service is not running",
    "details": {}
  }
}
```

### 6.3 创建视频请求

Hermes adapter 接收简化 schema：

```json
{
  "video_subject": "上海一日游",
  "video_script": "",
  "video_language": "zh-CN",
  "video_aspect": "9:16",
  "video_count": 1,
  "video_source": "pexels",
  "match_materials_to_script": false,
  "voice_name": "zh-CN-XiaoxiaoNeural-Female",
  "voice_rate": 1.0,
  "voice_volume": 1.0,
  "subtitle_enabled": true,
  "subtitle_position": "bottom",
  "font_name": "STHeitiMedium.ttc",
  "font_size": 60,
  "text_fore_color": "#FFFFFF",
  "stroke_color": "#000000",
  "stroke_width": 1.5,
  "bgm_type": "random",
  "bgm_volume": 0.2
}
```

Adapter 转换为 MoneyPrinterTurbo `TaskVideoRequest`。

### 6.4 输出路径策略

MoneyPrinterTurbo 原生输出在其 `storage/tasks/{task_id}/` 下。

Hermes 集成后建议：

```text
external/MoneyPrinterTurbo/storage/tasks/{task_id}/      # 上游实际输出
~/.hermes/capabilities/moneyprinter/tasks/{task_id}/     # 可选镜像/索引
Hermes artifacts index                                   # 结果登记
```

首期可以不复制大文件，只建立索引：

```json
{
  "task_id": "...",
  "engine": "moneyprinterturbo",
  "source_path": "external/MoneyPrinterTurbo/storage/tasks/.../final-1.mp4",
  "stream_url": "/capabilities/moneyprinter/stream/...",
  "download_url": "/capabilities/moneyprinter/download/..."
}
```

安全要求：

1. 只允许访问 task storage 根目录内的文件。
2. adapter 必须做 path traversal 防护。
3. renderer 不接触未校验绝对路径。
4. 日志不得打印 API key。

---

## 7. Service Manager 设计

MoneyPrinterTurbo 应作为独立本地服务运行。

### 7.1 服务职责

`capabilities/moneyprinter/service/moneyprinter_service.py` 负责：

```text
检查 external/MoneyPrinterTurbo 是否存在
检查 Python/uv 环境
检查依赖是否安装
检查 ffmpeg / imagemagick
生成或更新 config.toml
寻找可用端口
启动 FastAPI 服务
健康检查
停止服务
收集日志
报告错误给 Desktop UI
```

### 7.2 启动方式

首期推荐本地 Python/uv 启动，不强依赖 Docker：

```bash
cd external/MoneyPrinterTurbo
uv run python main.py
```

如果上游没有 uv 环境，则 service manager 可以创建独立 venv：

```bash
cd external/MoneyPrinterTurbo
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

但当前项目环境里 pip 可能不可用，Hermes repo 里也偏向 uv，因此优先设计为 uv。

备选 Docker 模式：

```bash
docker compose up api
```

Docker 模式作为后续可选项，不作为 Phase 1 默认路径。

### 7.3 端口策略

MoneyPrinterTurbo 默认端口：

```text
127.0.0.1:8080
```

Hermes Desktop 内嵌时应避免端口冲突。

策略：

1. 优先使用配置值：`capabilities.moneyprinter.port`。
2. 默认尝试 `18080`，避免和常见 `8080` 冲突。
3. 如端口被占用，自动寻找空闲端口。
4. 将实际端口写入 runtime state。
5. Desktop 和 MCP 都从 adapter/service manager 获取实际 base URL。

### 7.4 健康检查

健康检查包括：

```text
进程是否存活
端口是否监听
FastAPI 是否响应
核心依赖是否存在
config.toml 是否可读
storage 是否可写
```

建议 endpoint：

```http
GET /capabilities/moneyprinter/health
```

响应：

```json
{
  "ok": true,
  "data": {
    "installed": true,
    "service_running": true,
    "api_base_url": "http://127.0.0.1:18080/api/v1",
    "upstream_commit": "63113a3",
    "ffmpeg": true,
    "imagemagick": true,
    "storage_writable": true
  },
  "error": null
}
```

---

## 8. MCP 设计

MCP 是给 Hermes Agent 调用的，不是给 Desktop 页面调用的。

### 8.1 MCP server 位置

```text
capabilities/moneyprinter/mcp/server.py
capabilities/moneyprinter/mcp/tools.py
```

### 8.2 MCP tools 首期范围

首期只做 Agent 生成视频所需最小集合：

```text
moneyprinter_health_check
moneyprinter_generate_video
moneyprinter_get_task
moneyprinter_list_tasks
moneyprinter_list_outputs
```

第二阶段增加：

```text
moneyprinter_generate_script
moneyprinter_generate_terms
moneyprinter_delete_task
moneyprinter_upload_material
moneyprinter_upload_bgm
```

第三阶段增加：

```text
moneyprinter_generate_social_metadata
moneyprinter_cross_post
```

### 8.3 MCP tool 语义

#### moneyprinter_generate_video

输入：

```json
{
  "video_subject": "上海一日游",
  "video_script": "",
  "video_language": "zh-CN",
  "video_aspect": "9:16",
  "video_count": 1,
  "video_source": "pexels",
  "voice_name": "zh-CN-XiaoxiaoNeural-Female",
  "subtitle_enabled": true
}
```

输出：

```json
{
  "task_id": "...",
  "state": "queued",
  "message": "Video generation task created. Poll moneyprinter_get_task for status."
}
```

#### moneyprinter_get_task

输入：

```json
{
  "task_id": "..."
}
```

输出：

```json
{
  "task_id": "...",
  "state": "complete",
  "progress": 100,
  "videos": [
    {
      "name": "final-1.mp4",
      "stream_url": "...",
      "download_url": "...",
      "artifact_id": "..."
    }
  ]
}
```

### 8.4 Agent 使用模式

Agent 不应该一次性等待长任务阻塞到视频完成。推荐模式：

```text
1. 调 moneyprinter_generate_video 创建任务。
2. 告知用户 task_id 和预计耗时。
3. 轮询 moneyprinter_get_task。
4. 完成后返回视频预览/下载链接。
5. 失败时总结失败阶段和下一步建议。
```

---

## 9. Hermes skill 设计

新增 skill：

```text
skills/moneyprinter-video/SKILL.md
```

用途：

1. 告诉 Agent 什么时候使用 MoneyPrinter 视频能力。
2. 指导 Agent 如何拆解用户的视频需求。
3. 指导 Agent 如何选择参数。
4. 指导 Agent 如何处理长任务、失败、输出。
5. 避免 Agent 幻觉出不存在的视频功能。

Skill 内容应包含：

```text
触发场景：生成短视频、视频脚本、短视频素材、字幕视频等。
参数映射：用户自然语言 → MoneyPrinter 参数。
默认策略：9:16、1 个视频、字幕开启、BGM random。
长任务流程：create → poll → report。
安全规则：不要要求用户提供 secrets；不要打印 API key。
失败处理：素材下载失败、TTS 失败、LLM key 缺失、ffmpeg 缺失。
输出规则：返回 artifact/download/stream 链接，不把大文件塞进 prompt。
```

---

## 10. 配置与密钥管理

MoneyPrinterTurbo 需要多类配置：

```text
LLM provider / model / base_url / api_key
Pexels / Pixabay / Coverr API key
TTS provider
Whisper/subtitle provider
Proxy
ffmpeg / imagemagick path
任务并发数
storage 路径
```

### 10.1 Hermes 配置位置

非 secret 行为配置进入 Hermes config：

```yaml
capabilities:
  moneyprinter:
    enabled: true
    install_path: external/MoneyPrinterTurbo
    port: 18080
    auto_start: true
    video_source: pexels
    max_concurrent_tasks: 2
    output_mode: indexed
```

secret 不进入 git，不进入文档真实值，不进入普通日志。

### 10.2 Secret 策略

API key 建议存放：

```text
Hermes secret/env 管理层
或用户本地 MoneyPrinter config.toml，但绝不提交
```

MoneyPrinterTurbo `config.toml` 由 Hermes config manager 生成或更新：

```text
external/MoneyPrinterTurbo/config.toml
```

该文件必须加入 ignore 规则或确认上游已 ignore。

所有文档示例使用：

```text
[REDACTED]
```

不要使用真实 key。

### 10.3 Provider 复用策略

长期目标：复用 Hermes 已配置的 OpenAI-compatible provider。

首期可以允许用户单独配置 MoneyPrinter provider，因为 MoneyPrinterTurbo 支持自己的 provider 配置。

后续优化：

```text
Hermes provider settings
        ↓
MoneyPrinter config manager
        ↓
config.toml openai_base_url / openai_api_key / openai_model_name
```

注意：复用 Hermes provider 时不能把 OAuth token 或内部 credential pool 直接写入第三方 config。需要显式 adapter 或安全代理。

---

## 11. 分阶段开发计划

## Phase 0：准备和上游引入

目标：把 MoneyPrinterTurbo 作为 external 项目纳入工作区，但不改 Agent Core。

### Task 0.1：创建 external 目录

**文件：**

```text
external/MoneyPrinterTurbo/
capabilities/moneyprinter/upstream.json
```

**动作：**

1. 创建 `external/`。
2. 将 MoneyPrinterTurbo clone 到 `external/MoneyPrinterTurbo/`。
3. 记录 upstream URL、commit、license。
4. 确认不提交 `config.toml`、`storage/`、`.venv/`、生成视频。

**验证：**

```bash
git -C external/MoneyPrinterTurbo rev-parse --short HEAD
```

预期：输出固定 upstream commit。

### Task 0.2：确认 license 和 ignore

**文件：**

```text
external/MoneyPrinterTurbo/LICENSE
.gitignore
```

**动作：**

1. 确认 MIT License 存在。
2. 确认忽略以下路径：

```text
external/MoneyPrinterTurbo/config.toml
external/MoneyPrinterTurbo/storage/
external/MoneyPrinterTurbo/.venv/
external/MoneyPrinterTurbo/**/__pycache__/
```

**验证：**

```bash
git status --short external/MoneyPrinterTurbo
```

预期：不会显示 config/storage 产物。

---

## Phase 1：MoneyPrinter service manager + adapter API

目标：Hermes 能启动 MoneyPrinter API，并通过 adapter 查询健康状态、创建视频任务、查询任务、播放输出。

### Task 1.1：新增 capability 基础目录

**创建：**

```text
capabilities/moneyprinter/README.md
capabilities/moneyprinter/service/paths.py
capabilities/moneyprinter/service/health.py
capabilities/moneyprinter/service/config_manager.py
capabilities/moneyprinter/service/moneyprinter_service.py
capabilities/moneyprinter/adapter/models.py
capabilities/moneyprinter/adapter/client.py
```

**要求：**

1. 所有路径从 repo root 推导。
2. 不硬编码用户 home。
3. 不读取 secrets 到日志。
4. 不依赖 Agent Core。

### Task 1.2：实现 health check

**目标：**

检测：

```text
external/MoneyPrinterTurbo 是否存在
main.py 是否存在
requirements.txt 是否存在
ffmpeg 是否可用
imagemagick 是否可用
storage 是否可写
服务端口是否响应
```

**验证：**

```bash
python -m capabilities.moneyprinter.service.health
```

预期：输出 JSON 格式健康状态。

### Task 1.3：实现 config manager

**目标：**

1. 如 `external/MoneyPrinterTurbo/config.toml` 不存在，从 `config.example.toml` 创建。
2. 写入端口、host、storage、并发数等非 secret 配置。
3. 对 secret 字段保留 `[REDACTED]` 或从安全来源注入。
4. 不覆盖用户已有配置，除非字段由 Hermes 管理。

**验证：**

```bash
python -m capabilities.moneyprinter.service.config_manager --dry-run
```

预期：显示将修改的字段，不打印 secret。

### Task 1.4：实现 service start/stop

**目标：**

启动：

```bash
cd external/MoneyPrinterTurbo
uv run python main.py
```

需要：

1. stdout/stderr 写入 Hermes capability log。
2. 记录 PID。
3. 支持 stop。
4. 支持重复 start 时复用现有服务。
5. 支持端口冲突检测。

**验证：**

```bash
python -m capabilities.moneyprinter.service.moneyprinter_service start
python -m capabilities.moneyprinter.service.moneyprinter_service health
python -m capabilities.moneyprinter.service.moneyprinter_service stop
```

### Task 1.5：实现 adapter client

**目标：**

封装 MoneyPrinterTurbo 原生 API：

```text
create_video
get_task
list_tasks
delete_task
stream_output
download_output
generate_script
generate_terms
```

**验证：**

使用 mock 或本地服务 smoke test：

```bash
python -m capabilities.moneyprinter.adapter.client health
```

---

## Phase 2：Desktop Video Studio 页面首版

目标：新增 Hermes 原生 React 页面，完成 70% 高频功能闭环。

### Task 2.1：新增路由常量

**修改：**

```text
apps/desktop/src/app/routes.ts
```

**内容：**

新增：

```ts
export const VIDEO_STUDIO_ROUTE = '/video-studio'
```

扩展：

```text
AppView
AppRouteId
APP_ROUTES
```

**验证：**

```bash
npm --prefix apps/desktop run typecheck
```

或使用项目现有 Desktop typecheck 命令。

### Task 2.2：新增 sidebar 入口

**修改：**

```text
apps/desktop/src/app/chat/sidebar/index.tsx
```

**内容：**

1. import `VIDEO_STUDIO_ROUTE`。
2. 在 `SIDEBAR_NAV` 加入 Video Studio。
3. 使用 Codicon 视频/播放类图标。

**验证：**

启动 Desktop，侧边栏出现新图标，点击进入 `/video-studio`。

### Task 2.3：新增 VideoStudioView skeleton

**创建：**

```text
apps/desktop/src/app/video-studio/index.tsx
apps/desktop/src/app/video-studio/types.ts
apps/desktop/src/app/video-studio/moneyprinter-client.ts
```

**内容：**

1. 页面标题：`Video Studio`。
2. Service status card。
3. Generate button disabled until service healthy。
4. 三栏空布局。

### Task 2.4：挂载 route

**修改：**

```text
apps/desktop/src/app/desktop-controller.tsx
```

**内容：**

1. lazy import `VideoStudioView`。
2. 在 `<Routes>` 中新增 `path="video-studio"`。
3. 确认不是 overlay view。

### Task 2.5：实现 generation form

**创建：**

```text
apps/desktop/src/app/video-studio/components/generation-form.tsx
```

字段：

```text
video_subject
video_script
video_language
video_aspect
video_count
video_source
match_materials_to_script
voice_name
voice_rate
subtitle_enabled
```

提交后调用：

```text
moneyprinterClient.createVideo(payload)
```

### Task 2.6：实现 task polling

**创建：**

```text
apps/desktop/src/app/video-studio/hooks/use-moneyprinter-tasks.ts
apps/desktop/src/app/video-studio/components/task-list.tsx
```

行为：

1. 创建任务后立即加入本地 task list。
2. 每 1-3 秒查询 task 状态。
3. complete/failed 后停止高频轮询。
4. 支持手动刷新。
5. 支持删除任务。

### Task 2.7：实现 preview panel

**创建：**

```text
apps/desktop/src/app/video-studio/components/video-preview.tsx
```

行为：

1. 选中完成任务。
2. 展示 `final-*.mp4`。
3. 使用 adapter stream URL。
4. 提供下载、打开目录、复制路径按钮。

### Task 2.8：Desktop 首版验证

运行：

```bash
npm --prefix apps/desktop run typecheck
npm --prefix apps/desktop run lint
npm --prefix apps/desktop run build
```

如项目脚本名称不同，以 `apps/desktop/package.json` 为准。

手动 smoke test：

1. 打开 Desktop。
2. 点击 Video Studio。
3. Service status 显示可启动/已启动。
4. 输入主题。
5. 创建任务。
6. 任务进入列表。
7. 进度变化。
8. 完成后可以预览 mp4。

---

## Phase 3：MCP server + Agent 调用

目标：Hermes Agent 通过 MCP 使用同一个视频能力，但 Agent Core 不动。

### Task 3.1：新增 MCP server

**创建：**

```text
capabilities/moneyprinter/mcp/server.py
capabilities/moneyprinter/mcp/tools.py
```

工具：

```text
moneyprinter_health_check
moneyprinter_generate_video
moneyprinter_get_task
moneyprinter_list_tasks
moneyprinter_list_outputs
```

### Task 3.2：MCP 本地测试

运行：

```bash
python capabilities/moneyprinter/mcp/server.py
```

使用 Hermes MCP 测试命令：

```bash
hermes mcp add moneyprinter --command "python capabilities/moneyprinter/mcp/server.py"
hermes mcp test moneyprinter
```

### Task 3.3：新增 MoneyPrinter skill

**创建：**

```text
skills/moneyprinter-video/SKILL.md
```

内容：

1. 触发场景。
2. 默认参数。
3. create → poll → result 流程。
4. 失败处理。
5. 输出规范。

### Task 3.4：Agent smoke test

测试 prompt：

```text
帮我生成一个 15 秒中文竖屏短视频，主题是“上海早晨的咖啡店”，字幕打开，语音用默认女声。
```

预期：

1. Agent 调用 `moneyprinter_generate_video`。
2. 返回 task_id。
3. Agent 调用 `moneyprinter_get_task` 轮询。
4. 完成后返回 stream/download/artifact 信息。

---

## Phase 4：高级功能补齐

### 4.1 自定义 prompt

新增字段：

```text
video_script_prompt
custom_system_prompt
paragraph_number
```

UI：高级折叠面板。

### 4.2 本地素材

新增：

```text
GET /capabilities/moneyprinter/materials
POST /capabilities/moneyprinter/materials
```

UI：

```text
素材上传
素材列表
选择素材
顺序拼接
```

### 4.3 BGM 上传和选择

新增：

```text
GET /capabilities/moneyprinter/bgms
POST /capabilities/moneyprinter/bgms
```

UI：

```text
BGM 上传
BGM 选择
BGM 音量
```

### 4.4 字幕样式完整配置

新增：

```text
font_name
text_background_color
rounded_subtitle_background
custom_position
```

UI：字幕样式 panel。

### 4.5 社媒元数据

新增：

```text
POST /capabilities/moneyprinter/social-metadata
```

UI 显示：

```text
title
caption
hashtags
platform
```

### 4.6 Cross-post

谨慎后置。原因：

1. 涉及第三方账号授权。
2. 有外部副作用。
3. 需要明确用户确认。
4. 需要更严格 secret 管理。

必须加确认流程，不允许 Agent 或页面误触发发布。

---

## 12. 测试计划

### 12.1 Python capability tests

目标：测试 adapter/service/mcp，不测 MoneyPrinterTurbo 内部实现。

建议测试：

```text
tests/capabilities/moneyprinter/test_paths.py
tests/capabilities/moneyprinter/test_config_manager.py
tests/capabilities/moneyprinter/test_adapter_models.py
tests/capabilities/moneyprinter/test_adapter_client.py
tests/capabilities/moneyprinter/test_mcp_tools.py
```

重点：

1. 路径不能逃出 allowed root。
2. config manager 不泄露 secret。
3. adapter 错误格式稳定。
4. MCP tools schema 稳定。
5. service health 对缺依赖情况有清晰错误。

### 12.2 Desktop tests

建议：

```text
apps/desktop/src/app/video-studio/*.test.tsx
```

覆盖：

1. 表单默认值。
2. create video payload mapping。
3. task polling 状态变化。
4. failed 状态展示。
5. preview URL 使用 adapter URL，不使用任意绝对路径。

### 12.3 E2E smoke test

本地 smoke：

```text
1. service health
2. start service
3. create script
4. create short video task
5. poll task
6. verify output mp4 exists
7. stream/download endpoint returns video content
8. stop service
```

由于完整视频生成依赖外部网络/API，CI 中应拆分：

```text
unit tests: mock MoneyPrinter API
integration tests: optional，需要本地 env/key
manual smoke: 开发机执行
```

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 上游 MoneyPrinterTurbo API 变化 | Desktop/MCP 失效 | 用 Hermes adapter 隔离，不让 UI 直接依赖上游 schema |
| 外部 API key 缺失 | 生成失败 | Health/config UI 明确提示缺什么，不打印 secret |
| Pexels/Pixabay 网络失败 | 素材下载失败 | 支持本地素材、清晰错误、重试 |
| TTS 失败 | 无音频 | 提示 voice/language/network，后续支持自定义音频 |
| ffmpeg/imagemagick 缺失 | 合成失败 | service health 启动前检查 |
| 视频任务耗时长 | 用户以为卡住 | 任务列表、进度、日志、通知 |
| 大文件进入 prompt | 上下文污染 | Agent 只收路径/URL/artifact id，不收视频二进制 |
| renderer 路径访问风险 | 安全问题 | adapter 做 path whitelist 和 streaming |
| Cross-post 外部副作用 | 误发布 | 后置；必须人工确认；默认关闭 |
| 上游代码维护成本 | 升级复杂 | external 固定 commit；adapter 薄封装；记录 patch |

---

## 14. 验收标准

### Phase 1 验收

```text
MoneyPrinterTurbo 位于 external/MoneyPrinterTurbo/。
Hermes 能检测 MoneyPrinter 安装状态。
Hermes 能启动/停止 MoneyPrinter API service。
Hermes adapter 能 health/create/get/list/delete task。
不会修改 Hermes Agent Core。
```

### Phase 2 验收

```text
Desktop 出现 Video Studio 页面。
页面可以输入主题和基础参数。
页面可以创建视频任务。
页面可以显示任务状态和进度。
页面可以预览/下载完成的视频。
页面不直接访问不安全绝对路径。
Desktop typecheck/lint/build 通过。
```

### Phase 3 验收

```text
moneyprinter-mcp server 可被 Hermes MCP client 发现。
Agent 可以创建视频任务。
Agent 可以查询任务状态。
Agent 完成后返回视频链接或 artifact。
Agent Core 没有新增核心工具。
```

### Phase 4 验收

```text
高级配置逐步补齐。
本地素材、BGM、字幕样式可用。
社媒元数据可生成。
cross-post 默认关闭且需要明确确认。
```

---

## 15. 建议实施顺序

推荐严格按以下顺序：

```text
1. external/MoneyPrinterTurbo 引入和 ignore 规则
2. service health
3. service start/stop
4. adapter client
5. Desktop route + sidebar skeleton
6. Generation form
7. Task polling
8. Preview/download
9. Desktop build/lint/typecheck
10. MCP server
11. MoneyPrinter skill
12. Agent smoke test
13. 高级功能逐步补齐
```

不要一开始就做 MCP、UI、高级设置、cross-post 全部功能。先把最小闭环跑通：

```text
主题 → 创建任务 → 轮询 → 合成 → 预览
```

---

## 16. Open Questions

进入开发前建议确认：

1. `external/MoneyPrinterTurbo/` 是直接 vendor 提交，还是 git submodule？
2. Phase 1 默认服务端口是否使用 `18080`？
3. 是否允许 Desktop 首次启动时自动安装 MoneyPrinter Python 依赖？
4. 如果用户没有配置素材 API key，是否默认支持仅本地素材模式？
5. 输出视频是否要自动登记到 Hermes artifacts？如果是，artifact schema 需再补充。
6. Provider 是否首期单独配置 MoneyPrinter，还是直接复用 Hermes provider？
7. 是否需要保留原 Streamlit WebUI 作为 fallback/advanced 页面？
8. cross-post 是否进入首个产品版本，还是明确后置？

---

## 17. 推荐决策

我的建议：

```text
external/MoneyPrinterTurbo 使用 vendor 或 submodule 均可，但首期建议 vendor 固定 commit，减少开发变量。
默认端口使用 18080。
首期不自动复用 Hermes OAuth/provider credential，只让用户在 Video Studio 设置页配置 MoneyPrinter provider。
首期不做 cross-post。
首期不嵌 Streamlit，只做 Hermes 原生 React 页面。
首期输出先通过 adapter stream/download，后续再接 artifacts。
```

这样风险最低，能最快跑通产品闭环，同时完全符合“不碰 Hermes Agent 架构”的边界。
