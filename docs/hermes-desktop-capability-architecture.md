# Hermes Desktop 内置开源能力：MCP / API 架构与开发规则

> 本文档用于指导在 Hermes Desktop 中内置多个开源项目，并将它们封装为 MCP 工具或本地 API，供 Hermes Agent 调用。  
> 当前协作模式：使用 **Default Hermes Agent** 的主对话框作为整体构建与修改的中控大脑；Default 负责架构决策、任务拆解、子 Agent 调度、结果验收与最终集成。

---

## 1. 目标

我们要构建的是一个可分发给最终用户的 Hermes Desktop 版本，它可以内置一批开源项目，例如视频剪辑、音频处理、文档处理、图像处理、模型推理等能力。

这些能力不应该直接侵入 Hermes Agent Core，而应该被封装成独立、可测试、可替换、可分发的能力模块，并通过标准接口暴露给 Agent。

核心目标：

1. **保持 Agent Core 稳定**：不为了单个开源项目修改 Agent 主循环。
2. **用 MCP 作为 Agent 调用能力的首选接口**。
3. **用本地 API 作为外部程序、桌面 UI、长任务管理的补充接口**。
4. **Desktop 负责打包、启动、配置、健康检查和用户体验**。
5. **所有内置能力都要可独立测试、可关闭、可升级、可替换**。

---

## 2. Hermes 总体架构概览

Hermes 可以理解为：

```text
用户
 │
 ├─ Desktop App / TUI / CLI / Gateway / API
 │
 ▼
Hermes 前端与入口层
 │
 ▼
Hermes Python Agent Core
 │
 ├─ 模型调用层：OpenAI / Anthropic / OpenRouter / Copilot / local 等
 ├─ Prompt 构建层：system prompt、skills、memory、project rules
 ├─ Tool 调度层：terminal / file / browser / image / MCP tools 等
 ├─ Session 层：SQLite 会话、FTS 搜索、历史恢复
 ├─ Memory / Skills 层：长期记忆、技能文档
 ├─ Cron / Delegation / Gateway 后台能力
 │
 ▼
外部能力层
 ├─ 本地 shell / 文件系统 / 浏览器
 ├─ MCP Servers
 ├─ Plugins
 ├─ Local HTTP APIs
 └─ 被封装的开源项目 / 二进制工具 / 模型服务
```

关键源码位置：

| 路径 | 作用 |
|---|---|
| `run_agent.py` | Agent 主循环，处理 LLM 响应、工具调用、多轮执行 |
| `model_tools.py` | 工具发现、工具 schema 组织、工具调用分发 |
| `tools/` | Hermes 内置工具实现 |
| `tools/registry.py` | 工具注册中心，工具通过 `registry.register()` 注册 |
| `toolsets.py` | 工具分组与启用策略，决定哪些工具暴露给模型 |
| `agent/` | prompt、模型路由、memory、压缩、provider 等内部逻辑 |
| `hermes_cli/` | CLI 命令、配置、setup、profile、mcp 命令等 |
| `gateway/` | Telegram、Discord、API Server、Webhook 等消息网关 |
| `tui_gateway/` | TUI/Desktop 与 Python backend 的桥接层 |
| `apps/desktop/` | Electron 桌面端 |
| `plugins/` | 插件系统 |
| `skills/` | Agent 技能与工作流说明 |
| `cron/` | 定时任务系统 |

---

## 3. Desktop 运行架构

Hermes Desktop 是 **Electron + React Renderer + Python Backend** 的架构。

```text
Electron Main Process
 │
 ├─ 启动窗口、菜单、系统能力
 ├─ 启动或连接 Hermes Python backend
 ├─ 处理打包资源、原生依赖、文件访问、系统事件
 ├─ 通过 preload 暴露 `window.hermesDesktop`
 │
 ▼
React Renderer
 │
 ├─ 聊天界面
 ├─ 会话列表
 ├─ 设置页
 ├─ 文件预览
 ├─ 状态展示
 └─ 通过 WebSocket / IPC 与 backend 通信
 │
 ▼
Hermes Python Backend
 │
 ├─ Agent Core
 ├─ Tools
 ├─ MCP Client
 ├─ Gateway
 ├─ Session / Memory / Skills
 └─ Cron / Delegation
```

Desktop 不是一个独立 Agent。Desktop 是 Hermes Agent 的用户界面、进程管理器和本地能力承载层。真正的智能体仍然由 Python Agent Core 驱动。

相关路径：

| 路径 | 作用 |
|---|---|
| `apps/desktop/electron/main.cjs` | Electron 主进程，负责窗口、IPC、backend 启动、打包资源等 |
| `apps/desktop/electron/preload.cjs` | 安全地向 renderer 暴露桌面 API |
| `apps/desktop/src/` | React 前端代码 |
| `apps/desktop/src/store/gateway.ts` | Desktop 与 Hermes backend 的 gateway 连接管理 |
| `apps/desktop/package.json` | Desktop 构建、打包、extraResources 配置 |

---

## 4. Agent 工具调用机制

Hermes Agent 的基本运行循环：

```text
用户消息
 │
 ▼
构建 system prompt
 │
 ├─ 模型配置
 ├─ 当前会话状态
 ├─ project rules / AGENTS.md / skills / memory
 ├─ 可用工具 schema
 │
 ▼
调用 LLM
 │
 ▼
LLM 返回
 ├─ 普通文本：直接展示给用户
 └─ tool_calls：Hermes 调用对应工具
                  │
                  ▼
                工具结果写回模型上下文
                  │
                  ▼
                模型继续推理或输出最终答案
```

工具的来源包括：

1. Hermes 内置工具：`tools/*.py`
2. MCP 工具：由 `mcp_servers` 配置的外部 MCP Server 自动发现
3. 插件暴露的工具或能力
4. Gateway / Desktop 侧提供的能力

工具进入模型上下文需要付出成本，所以 Hermes 的设计原则是：

> **核心 Agent 是窄腰。能力应该尽量放在边缘，通过 MCP、plugin、skill 或 API 接入，而不是持续增大 core tool schema。**

---

## 5. 新能力接入的决策规则

新增能力时，按以下优先级选择实现方式。

### 5.1 首选：Skill

适用场景：

- 不需要新增结构化工具；
- Agent 可以通过已有 terminal/file/browser 工具完成；
- 只是需要告诉 Agent 某个 CLI、API 或工作流怎么用。

例子：

- FFmpeg 常用剪辑命令说明；
- 如何处理字幕；
- 如何把视频剪辑任务拆成 probe、trim、merge、export。

优点：

- 不改代码；
- 不增加 tool schema；
- 成本最低。

### 5.2 推荐：MCP Server

适用场景：

- 能力需要结构化参数和结构化返回；
- 需要让 Agent 自然调用；
- 能力可独立运行；
- 希望以后被其他 MCP host 复用。

这是内置开源项目的默认推荐方式。

例子：

```text
video_editor_mcp
 ├─ probe_video
 ├─ trim_video
 ├─ merge_videos
 ├─ extract_audio
 ├─ add_subtitles
 ├─ render_timeline
 └─ get_job_status
```

Hermes 配置示例：

```yaml
mcp_servers:
  video_editor:
    command: "/path/to/video-editor-mcp"
    args: []
    timeout: 600
    connect_timeout: 60
```

工具注册后会以类似名称出现在 Agent 工具集中：

```text
mcp_video_editor_probe_video
mcp_video_editor_trim_video
mcp_video_editor_merge_videos
```

### 5.3 补充：Local HTTP API

适用场景：

- 桌面 UI 也要调用；
- 外部程序要调用；
- 需要长任务、进度、取消、文件下载；
- MCP tool 不适合承载持续连接或大文件传输。

建议只监听：

```text
127.0.0.1
```

不应默认监听 `0.0.0.0`。

### 5.4 可选：Plugin

适用场景：

- 需要更深地接入 Hermes 配置、dashboard、gateway、profile、生命周期；
- 需要自动安装、启动、升级、健康检查；
- 需要增加 UI 页面或 dashboard 面板；
- 需要把多个 MCP/API 能力组织为一个完整产品模块。

Plugin 可以管理 capability，但 capability 的 Agent 调用面仍建议优先走 MCP。

### 5.5 最后选择：Core Tool

只有在以下条件都满足时才考虑加入 Hermes core tool：

1. 对绝大多数用户都基础且必要；
2. 无法通过 terminal/file/MCP/plugin 实现；
3. 工具 schema 足够稳定；
4. 不会显著增加模型上下文成本；
5. 有完整测试和维护计划。

视频剪辑、音频处理、文档处理等开源项目封装通常不应作为 core tool。

---

## 6. 内置开源项目的标准 Capability 架构

每个内置开源能力应被封装为一个 capability。

```text
capabilities/
 └─ video-editor/
    ├─ manifest.json
    ├─ README.md
    ├─ mcp_server/
    ├─ api_server/
    ├─ runtime/
    ├─ tests/
    └─ licenses/
```

建议的 capability 结构：

```text
Capability
 │
 ├─ Manifest
 │   ├─ name
 │   ├─ version
 │   ├─ description
 │   ├─ license
 │   ├─ bundled binaries
 │   ├─ mcp command
 │   ├─ api command / port
 │   ├─ health check
 │   └─ enabled by default?
 │
 ├─ MCP Server
 │   └─ 给 Agent 调用的结构化工具
 │
 ├─ Local API Server
 │   └─ 给 Desktop UI / 外部程序 / 长任务系统调用
 │
 ├─ Runtime Adapter
 │   └─ 调用底层开源项目、二进制、模型、脚本
 │
 ├─ Job Manager
 │   └─ 长任务、进度、取消、日志、输出文件
 │
 ├─ Tests
 │   └─ 单元测试、集成测试、E2E 测试
 │
 └─ License Notices
     └─ 第三方开源许可证、版权声明
```

原则：

> Agent 不直接依赖具体开源项目。Agent 只依赖 MCP/API contract。底层实现可以被替换。

---

## 7. MCP 与 API 的职责边界

### MCP 负责 Agent 调用

MCP 工具应该：

- 接收结构化参数；
- 返回结构化 JSON；
- 返回文件路径、元数据、job id，而不是大文件内容；
- 避免长时间阻塞；
- 对危险操作进行明确表达；
- 适合被 LLM 规划和组合调用。

示例：

```json
{
  "tool": "trim_video",
  "arguments": {
    "input_path": "/Users/me/input.mp4",
    "start": "00:00:10",
    "end": "00:00:25",
    "output_path": "/Users/me/output.mp4"
  }
}
```

返回：

```json
{
  "success": true,
  "output_path": "/Users/me/output.mp4",
  "duration": 15.0,
  "codec": "h264"
}
```

### API 负责 UI、外部调用和长任务

API 应该：

- 提供 job 创建、查询、取消；
- 提供进度；
- 支持 Desktop 预览、下载、打开输出目录；
- 对外部程序提供稳定接口；
- 只绑定本机回环地址；
- 做鉴权或本地 token 校验，避免被其他本机恶意进程滥用。

示例：

```http
POST /api/video/jobs
GET  /api/video/jobs/{job_id}
POST /api/video/jobs/{job_id}/cancel
GET  /api/video/jobs/{job_id}/logs
GET  /api/video/outputs/{asset_id}
```

---

## 8. 视频剪辑 Capability 建议设计

第一版建议基于：

```text
FFmpeg + FFprobe + Python MCP Server + 可选 FastAPI 本地 API
```

### 8.1 MCP 工具列表

建议第一版工具：

| 工具 | 作用 |
|---|---|
| `probe_video` | 获取视频元数据：时长、分辨率、fps、编码、音轨、字幕轨 |
| `trim_video` | 截取片段 |
| `merge_videos` | 合并多个视频 |
| `extract_audio` | 抽取音频 |
| `transcode_video` | 转码、压缩、改分辨率 |
| `add_subtitles` | 添加或烧录字幕 |
| `generate_thumbnail` | 生成封面图 |
| `render_timeline` | 根据 timeline JSON 渲染成片 |
| `start_job` | 创建长任务 |
| `get_job_status` | 查询任务状态 |
| `cancel_job` | 取消任务 |

### 8.2 Timeline 格式

建议定义统一 `timeline.json`：

```json
{
  "version": "1.0",
  "canvas": {
    "width": 1920,
    "height": 1080,
    "fps": 30
  },
  "tracks": [
    {
      "type": "video",
      "clips": [
        {
          "source": "/path/input.mp4",
          "start": 0,
          "end": 10,
          "timeline_start": 0
        }
      ]
    },
    {
      "type": "audio",
      "clips": []
    },
    {
      "type": "subtitle",
      "clips": []
    }
  ],
  "effects": [],
  "export": {
    "format": "mp4",
    "codec": "h264",
    "preset": "medium"
  }
}
```

### 8.3 长任务返回格式

对于耗时任务，不要让 MCP tool 长时间阻塞。推荐返回 job：

```json
{
  "success": true,
  "job_id": "job_abc123",
  "status": "running",
  "progress": 0.0,
  "message": "Video render started"
}
```

查询结果：

```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "progress": 1.0,
  "output_path": "/path/output.mp4",
  "logs_tail": "..."
}
```

---

## 9. Desktop 打包规则

### 9.1 二进制依赖

内置能力如果依赖二进制工具，例如：

- `ffmpeg`
- `ffprobe`
- `imagemagick`
- `whisper.cpp`
- `node` 工具链
- 本地模型 runtime

应通过 Desktop 的打包资源系统加入安装包。

当前 Desktop 已经使用：

```json
"extraResources": [
  {
    "from": "build/native-deps",
    "to": "native-deps"
  }
]
```

建议把平台相关二进制放入：

```text
apps/desktop/build/native-deps/<capability>/<platform>/<arch>/
```

运行时通过 `process.resourcesPath` 定位。

### 9.2 Python / Node 依赖

原则：

1. 不要求普通用户手动安装复杂依赖；
2. 能随 Desktop 打包的尽量随包；
3. 大型模型或大体积依赖可以按需下载；
4. 下载必须有校验和、版本锁定、失败恢复；
5. 离线模式下能力应该明确显示不可用原因。

### 9.3 首次启动自动配置

用户不应该手动修改 `~/.hermes/config.yaml` 才能使用内置能力。

Desktop 或 capability manager 应该自动写入或合并：

```yaml
mcp_servers:
  video_editor:
    command: "/Applications/Hermes.app/Contents/Resources/native-deps/video-editor/video-editor-mcp"
    args: []
    timeout: 600
    connect_timeout: 60
```

注意：

- 不要覆盖用户已有配置；
- 要可关闭；
- 要可恢复默认；
- 要保留用户自定义 MCP server。

### 9.4 健康检查

每个 capability 必须有健康检查：

```text
- 二进制是否存在
- 版本是否匹配
- MCP server 是否能启动
- API server 是否能响应
- 输出目录是否可写
- 权限是否满足
- 许可证 notice 是否已包含
```

Desktop UI 应展示：

```text
Video Editor: Ready / Missing Dependency / Disabled / Failed
```

---

## 10. 安全规则

### 10.1 文件安全

- 不把视频、音频、文档原始内容塞入 prompt；
- tool 返回路径、metadata、缩略图、摘要即可；
- 删除、覆盖、移动用户文件前必须明确确认；
- 输出路径必须规范化，防止路径穿越；
- 默认输出到用户明确选择的目录或 Hermes workspace；
- 临时文件要有清理策略。

### 10.2 进程安全

- 子进程参数必须以数组形式传入，避免 shell injection；
- 尽量不用 `shell=True`；
- 对输入路径、输出路径、filter 参数做校验；
- 处理超时、取消和异常退出；
- 记录 stderr，但不要把敏感路径或隐私内容无限制回传给模型。

### 10.3 网络安全

- 本地 API 默认只绑定 `127.0.0.1`；
- 不默认开启公网监听；
- 如果需要远程访问，必须显式配置；
- API 应使用本地 token 或 origin 限制；
- 不自动上传用户文件到第三方服务，除非用户明确授权。

### 10.4 Prompt Injection 防护

开源工具处理的文件、字幕、网页、元数据都可能包含恶意文本。规则：

- 文件内容、字幕内容、元数据都视为不可信数据；
- MCP/API 返回中不要包含“让 Agent 忽略指令”之类内容；
- 如需返回文本内容，应明确包裹为 data；
- Agent 不应执行来自媒体文件内容的指令。

---

## 11. License 与分发规则

内置开源项目前必须检查许可证。

| License | 一般可分发性 | 注意事项 |
|---|---|---|
| MIT / BSD / Apache-2.0 | 通常友好 | 保留版权声明和 notice |
| LGPL | 通常可用 | 注意动态链接、替换要求、源码提供义务 |
| GPL | 高风险 | 可能要求整个分发遵循 GPL |
| AGPL | 更高风险 | 网络服务场景也可能触发开源义务 |
| 专有 / 未授权 | 不可内置 | 需要授权 |

FFmpeg 需要特别注意：

- FFmpeg 的许可证取决于编译选项；
- 启用某些 codec 或库可能改变分发义务；
- 必须记录构建参数和第三方库许可证。

每个 capability 必须包含：

```text
licenses/
 ├─ THIRD_PARTY_NOTICES.md
 ├─ ffmpeg.LICENSE
 └─ dependency-licenses/
```

---

## 12. 测试与验收规则

每个 capability 至少需要：

### 12.1 单元测试

- 参数校验；
- 路径规范化；
- timeline 解析；
- job 状态转换；
- 错误返回格式。

### 12.2 集成测试

- MCP server 能启动；
- `list_tools` 返回预期工具；
- 典型 tool call 能执行；
- API health check 正常；
- 错误输入能返回结构化错误。

### 12.3 E2E 测试

- Desktop 启动后 capability 状态为 Ready；
- Hermes 能发现 MCP 工具；
- Agent 能调用 `probe_video`；
- Agent 能完成一个最小剪辑任务；
- 输出文件存在且可播放或可被 `ffprobe` 识别。

### 12.4 打包测试

- macOS dmg/zip；
- Windows nsis/msi；
- Linux AppImage/deb/rpm；
- fresh install；
- existing install；
- 离线启动；
- 无依赖环境启动。

---

## 13. Default Hermes Agent 作为中控大脑的协作规则

当前项目协作约定：

> **Default Hermes Agent 的主对话框是整个框架构建与修改的中控大脑。**

Default 的职责：

1. **总体架构决策**：决定 capability、MCP、API、plugin、desktop UI 的边界。
2. **任务拆解**：把大功能拆成可执行子任务。
3. **子 Agent 调度**：把独立研究、代码审查、实现探索交给子 Agent。
4. **上下文汇总**：收集子 Agent 输出，判断是否可信。
5. **最终集成**：由主对话框确认代码位置、应用补丁、运行测试。
6. **验收把关**：不以子 Agent 自述为准，必须用工具验证文件、测试、构建输出。
7. **记忆与规则维护**：把稳定约定写入 memory，把可复用流程写入 skill 或文档。

子 Agent 的职责：

- 做局部研究；
- 读代码并报告发现；
- 提出实现方案；
- 做独立审查；
- 不直接作为最终事实来源。

重要规则：

- 子 Agent 的结论必须被 Default 验证；
- 有外部副作用的事情必须由 Default 最终确认；
- 子 Agent 不应修改核心架构决策；
- 子 Agent 输出不能替代测试；
- Default 对最终用户和代码库负责。

---

## 14. 推荐实施路线图

### Phase 1：视频 MCP 最小闭环

目标：让 Agent 能调用 MCP 完成最小视频剪辑任务。

任务：

1. 设计 `video-editor-mcp` 目录；
2. 基于 FFmpeg/FFprobe 实现：
   - `probe_video`
   - `trim_video`
   - `merge_videos`
   - `extract_audio`
   - `transcode_video`
3. 写 MCP server；
4. 本地配置 Hermes MCP；
5. 验证 Hermes Desktop 中 Agent 可以调用工具。

验收：

```text
用户给出 input.mp4 和时间范围
Agent 调用 MCP
MCP 调用 FFmpeg
输出 output.mp4
Agent 返回输出路径和 metadata
```

### Phase 2：长任务系统

目标：支持长视频渲染、进度、取消。

任务：

1. 实现 job manager；
2. 增加 `start_job`、`get_job_status`、`cancel_job`；
3. API 暴露 job 状态；
4. Desktop UI 展示进度。

### Phase 3：Desktop 自动集成

目标：用户安装后开箱即用。

任务：

1. 打包 FFmpeg；
2. 打包 MCP server；
3. 首次启动自动写入 `mcp_servers`；
4. 设置页显示 Video capability 状态；
5. 支持启用/禁用。

### Phase 4：Timeline 项目格式

目标：从简单命令升级为多轨剪辑。

任务：

1. 定义 `timeline.json`；
2. 支持多轨 video/audio/subtitle/image；
3. 实现 `render_timeline`；
4. 支持模板与预设。

### Phase 5：多 capability 管理

目标：把视频、音频、文档、图像等能力统一管理。

任务：

1. 设计 capability manifest；
2. 设计 capability manager；
3. 统一健康检查；
4. 统一安装、升级、禁用；
5. 统一 license notice。

---

## 15. 开发时的 Do / Don't

### Do

- 优先 MCP；
- 优先独立 capability；
- 保持 Agent Core 稳定；
- 对长任务使用 job id；
- 对大文件只传路径和 metadata；
- 每个 capability 都写 manifest、health check、tests、license notice；
- Desktop 负责用户体验和自动配置；
- Default 主对话框负责最终架构和验收。

### Don't

- 不要为了一个开源项目改 Agent 主循环；
- 不要把视频、音频大文件塞进 prompt；
- 不要默认开启公网 API；
- 不要没有 license 审查就内置依赖；
- 不要把所有能力都塞进 core toolset；
- 不要让子 Agent 的自述替代真实测试；
- 不要覆盖用户已有 MCP 配置；
- 不要要求普通用户手动安装复杂依赖后才能使用内置能力。

---

## 16. 总结原则

最终原则一句话：

> **Hermes Agent Core 保持窄而稳；开源项目封装为 capability；capability 对 Agent 暴露 MCP，对 Desktop/UI/外部程序暴露本地 API；Desktop 负责打包、启动、配置和健康检查；Default Hermes Agent 主对话框作为所有构建与修改工作的中控大脑。**
