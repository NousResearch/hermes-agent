# WebSocket Server 平台插件

## 概述

`plugins/platforms/ws_server/` 是一个 Hermes Agent 平台插件，为 Agent 添加 **WebSocket 服务端** 能力。外部前端（如 hermes-webui）通过 WebSocket 客户端连接后，可以使用完整的 `_HERMES_CORE_TOOLS` 工具集，包括 `terminal`、`execute_code`、`browser`、`file` 等所有核心工具。

### 文件结构

```
plugins/platforms/ws_server/
├── plugin.yaml      # 插件清单（kind: platform）
├── __init__.py      # 导出 register()
└── adapter.py       # WSServerAdapter 实现 (~550 行)
```

### 工作原理

- 基于 aiohttp `WebSocketResponse` 实现
- 继承 `BasePlatformAdapter`，复用 Hermes Gateway 完整的 agent 管线
- 通过 `resolve_toolset("hermes-ws-server")` 自动继承 `_HERMES_CORE_TOOLS`
- 支持 per-chat 串行锁、审批交互、后台任务推送

---

## 安装

### 前提条件

- Hermes Agent 已安装并可用
- Python 3.10+
- aiohttp（Hermes Agent 自带）

### 手动安装

```bash
# 1. 进入 hermes-agent 仓库
cd /path/to/hermes-agent

# 2. 复制插件文件到 plugins 目录
mkdir -p plugins/platforms/ws_server
cp path/to/ws_server/*.py plugins/platforms/ws_server/
cp path/to/ws_server/plugin.yaml plugins/platforms/ws_server/

# 3. 重启 gateway
supervisorctl restart hermes
# 或
hermes gateway restart
```

### 一键安装

```bash
# 从仓库根目录运行
bash docs/install-ws-server.sh
```

---

## 配置

### 方式一：config.yaml（推荐）

```bash
hermes config set gateway.platforms.ws_server.enabled true
hermes config set gateway.platforms.ws_server.extra.api_key "your-secret-key"
hermes config set gateway.platforms.ws_server.extra.host "0.0.0.0"
hermes config set gateway.platforms.ws_server.extra.port 8765
```

或直接编辑 `~/.hermes/config.yaml`：

```yaml
gateway:
  platforms:
    ws_server:
      enabled: true
      extra:
        api_key: "your-strong-secret-key-here"
        host: "0.0.0.0"
        port: 8765
```

### 方式二：环境变量

```bash
WS_SERVER_API_KEY=your-key
WS_SERVER_HOST=0.0.0.0
WS_SERVER_PORT=8765
```

---

## 使用

### 启动

```bash
supervisorctl restart hermes
```

### 验证

```bash
# 检查端口
ss -tlnp | grep 8765

# 健康检查
curl http://127.0.0.1:8765/health

# 预期输出：
# {"status":"ok","platform":"ws_server","connected_clients":0}

# 查看日志
tail -f ~/.hermes/logs/gateway.log | grep ws_server
# 成功时显示：
# [ws_server] Listening on ws://0.0.0.0:8765/_ws
# ✓ ws_server connected
```

---

## WebSocket 协议

### 端点

```
ws://host:port/_ws
```

### 认证

**客户端 → 服务端**（连接后第一条消息必须为认证消息）：

```json
{"type": "auth", "api_key": "your-key"}
```

**服务端 → 客户端**：

```json
{"type": "auth_ok", "chat_id": "ws_abc123def456"}
```

### 客户端 → 服务端消息

| 类型 | 用途 | 必填字段 |
|------|------|---------|
| `msg` | 发送用户消息 | `chat_id`, `text` |
| `approve` | 审批响应 | `approval_id`, `choice` |
| `stop` | 停止运行 | `run_id` |
| `ping` | 心跳 | 无 |

### 服务端 → 客户端事件

| 类型 | 用途 | 关键字段 |
|------|------|---------|
| `send` | 文本回复 | `chat_id`, `content` |
| `edit` | 编辑消息 | `chat_id`, `message_id`, `content` |
| `approval_card` | 审批请求 | `chat_id`, `approval_id`, `command`, `reason` |
| `typing` | 打字指示 | `chat_id` |
| `send_file` | 文件/图片 | `chat_id`, `file_path`, `caption` |
| `system` | 系统消息 | `chat_id`, `message` |
| `done` | 处理完成 | `chat_id` |
| `ping` | 心跳 | 无 |
| `pong` | 心跳回复 | 无 |

---

## 工具集

该插件自动使用 `hermes-ws-server` 工具集，无需手动配置。通过 `toolsets.py` 的自动回退机制，`resolve_toolset("hermes-ws-server")` 会自动展开为 `_HERMES_CORE_TOOLS`：

- `terminal`, `process` — 终端执行
- `execute_code` — 代码执行
- `read_file`, `write_file`, `patch`, `search_files` — 文件操作
- `browser_*` — 浏览器自动化
- `web_search`, `web_extract` — 网络搜索
- `delegate_task` — 子代理委派
- `todo`, `memory` — 任务和记忆
- `cronjob` — 定时任务
- 等全部核心工具

可以通过 `hermes tools` 命令按平台启用/禁用具体工具集。

---

## 架构

### 类层次

```
BasePlatformAdapter (gateway/platforms/base.py)
  └─ WSServerAdapter (plugins/platforms/ws_server/adapter.py)
       ├─ connect()      → 启动 aiohttp WS 服务端
       ├─ disconnect()   → 停止服务端
       ├─ send()         → 通过 WS 发送文本
       ├─ edit_message() → 编辑消息
       ├─ send_image()   → 发送文件/图片
       ├─ send_typing()  → 打字指示
       ├─ send_exec_approval() → 审批请求
       ├─ on_processing_start() → 开始处理
       └─ on_processing_complete() → 处理完成（发送 done）
```

### 连接生命周期

```
WS 客户端连接
    ↓
_auth_ok（认证 + chat_id 绑定）
    ↓
_dispatch_ws_message（消息分发）
    ↓
_handle_inbound → MessageEvent → handle_message()
    ↓
Agent 管线处理
    ↓
send() / edit_message() / send_exec_approval() → WS 推送
    ↓
on_processing_complete() → {"type": "done"}
```

---

## 与 API Server 的对比

| 特性 | API Server | WS Server |
|------|-----------|-----------|
| 协议 | HTTP/SSE | WebSocket |
| 依赖 | aiohttp | aiohttp |
| 工具集 | `hermes-api-server` | `hermes-ws-server`（同 `_HERMES_CORE_TOOLS`） |
| 全双工 | ❌ | ✅ |
| 后台通知 | ❌ | ✅ |
| 连接持久性 | 无状态 | 有状态 |
| 鉴权 | `API_SERVER_KEY` | `WS_SERVER_API_KEY` |
| 实现方式 | 内置（`gateway/platforms/`） | 插件（`plugins/platforms/`） |
| 核心代码修改 | 无 | 无 |