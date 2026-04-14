[根目录](../CLAUDE.md) > **gateway**

# Gateway 模块 - 消息平台网关

> 最后更新：2026-04-13 16:35:00

## 模块职责

`gateway/` 模块是 Hermes Agent 的消息平台网关，负责将 Hermes 连接到各种消息平台（Telegram、Discord、Slack、WhatsApp、Signal 等）。该模块提供统一的接口，使用户可以通过任何支持的平台与 Hermes 交互。

主要职责：
- **平台适配器**：为每个消息平台提供统一的适配器接口
- **会话管理**：持久化会话状态和历史记录
- **命令调度**：将平台消息映射到 Hermes 斜杠命令
- **多平台支持**：同时支持多个平台实例
- **危险命令审批**：在消息平台上审批危险命令
- **消息格式化**：将 Hermes 响应格式化为平台原生格式
- **文件处理**：支持平台特定的文件上传和下载
- **重启通知**：`/restart` 命令完成后自动通知请求者
- **服务重启**：支持 systemd 服务重启（替代分离的子进程）

## 入口与启动

### 主要入口点
- **`gateway/run.py`**：`start_gateway()` 函数 - 网关主入口
- **`hermes gateway` 命令**：通过 CLI 启动网关

### 启动流程
```
hermes gateway 命令
  ↓
gateway/run.py:start_gateway()
  ↓
加载配置和凭据
  ↓
初始化平台适配器
  ↓
启动平台客户端
  ↓
进入事件循环
```

## 对外接口

### GatewayRunner 类 (run.py)

**主要方法**：
```python
class GatewayRunner:
    def __init__(self, config: dict = None):
        """初始化网关运行器。"""

    async def start(self) -> None:
        """启动所有配置的平台适配器。"""

    async def stop(self) -> None:
        """停止所有平台适配器。"""

    def add_platform(self, platform_name: str, adapter) -> None:
        """添加平台适配器。"""

    def get_platform(self, platform_name: str) -> Any:
        """获取平台适配器。"""
```

### 平台适配器接口

所有平台适配器都实现以下接口：

```python
class BasePlatformAdapter:
    async def start(self) -> None:
        """启动平台客户端。"""

    async def stop(self) -> None:
        """停止平台客户端。"""

    async def send_message(self, session_id: str, message: str) -> None:
        """发送消息到平台。"""

    async def handle_message(self, message: Any) -> None:
        """处理来自平台的消息。"""

    def format_message(self, content: str, metadata: dict = None) -> Any:
        """格式化消息为平台原生格式。"""
```

### 支持的平台

#### Telegram (gateway/platforms/telegram.py)
**特性**：
- 完整的 Telegram Bot API 支持
- 斜杠命令自动注册
- 内联键盘按钮（审批危险命令）
- 文件上传和下载
- Markdown 和 HTML 格式化

**配置**：
```yaml
gateway:
  telegram:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    home_channel: "YOUR_HOME_CHANNEL_ID"
```

**主要函数**：
```python
class TelegramAdapter:
    async def start(self) -> None:
        """启动 Telegram bot。"""

    async def send_message(self, chat_id: str, message: str) -> None:
        """发送消息到 Telegram chat。"""

    async def handle_update(self, update: Update) -> None:
        """处理 Telegram 更新。"""
```

#### Discord (gateway/platforms/discord.py)
**特性**：
- 完整的 Discord Bot API 支持
- 斜杠命令注册
- 嵌入式消息（Embeds）
- 文件附件
- 反应按钮（审批危险命令）

**配置**：
```yaml
gateway:
  discord:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    home_channel: "YOUR_HOME_CHANNEL_ID"
```

**主要函数**：
```python
class DiscordAdapter:
    async def start(self) -> None:
        """启动 Discord bot。"""

    async def send_message(self, channel_id: str, message: str) -> None:
        """发送消息到 Discord channel。"""

    async def handle_message(self, message: Message) -> None:
        """处理 Discord 消息。"""
```

#### Slack (gateway/platforms/slack.py)
**特性**：
- Slack RTM API 支持
- 斜杠命令集成
- 块消息格式化
- 文件上传

**配置**：
```yaml
gateway:
  slack:
    enabled: true
    bot_token: "YOUR_BOT_TOKEN"
    home_channel: "YOUR_HOME_CHANNEL_NAME"
```

**主要函数**：
```python
class SlackAdapter:
    async def start(self) -> None:
        """启动 Slack bot。"""

    async def send_message(self, channel: str, message: str) -> None:
        """发送消息到 Slack channel。"""

    async def handle_event(self, event: dict) -> None:
        """处理 Slack 事件。"""
```

#### WhatsApp (gateway/platforms/whatsapp.py)
**特性**：
- WhatsApp Business API 支持
- 文本和媒体消息
- 消息模板

**配置**：
```yaml
gateway:
  whatsapp:
    enabled: true
    phone_number_id: "YOUR_PHONE_NUMBER_ID"
    access_token: "YOUR_ACCESS_TOKEN"
```

#### Signal (gateway/platforms/signal.py)
**特性**：
- Signal REST API 支持
- 端到端加密
- 群组和私聊

**配置**：
```yaml
gateway:
  signal:
    enabled: true
    http_url: "http://localhost:8080"
    account: "YOUR_SIGNAL_ACCOUNT"
```

### 会话管理 (session.py)

**主要类和函数**：
```python
class GatewaySession:
    def __init__(self, session_id: str, platform: str):
        """初始化会话。"""

    def load_history(self) -> list:
        """加载会话历史。"""

    def save_history(self, messages: list) -> None:
        """保存会话历史。"""

    def get_metadata(self) -> dict:
        """获取会话元数据。"""

class SessionManager:
    def __init__(self, hermes_home: Path):
        """初始化会话管理器。"""

    def get_session(self, session_id: str) -> GatewaySession:
        """获取会话。"""

    def create_session(self, session_id: str, platform: str) -> GatewaySession:
        """创建新会话。"""

    def delete_session(self, session_id: str) -> None:
        """删除会话。"""
```

### 命令调度 (run.py)

**主要函数**：
```python
async def dispatch_message(
    platform: str,
    session_id: str,
    message: str,
    metadata: dict = None,
) -> str:
    """调度消息到 Hermes 代理。"""

async def handle_slash_command(
    platform: str,
    session_id: str,
    command: str,
    args: str,
) -> str:
    """处理斜杠命令。"""

def is_dangerous_command(command: str) -> bool:
    """检查是否为危险命令。"""

async def request_approval(
    platform: str,
    session_id: str,
    command: str,
) -> bool:
    """请求用户审批危险命令。"""
```

### 重启通知机制 (run.py)

**主要函数**：
```python
async def send_restart_notification(
    platform: str,
    chat_id: str,
    thread_id: str = None,
) -> None:
    """发送重启成功通知到请求者。"""

async def handle_restart_command(
    platform: str,
    chat_id: str,
    thread_id: str = None,
) -> None:
    """处理 /restart 命令并保存通知信息。"""

def save_restart_notification(
    platform: str,
    chat_id: str,
    thread_id: str = None,
) -> None:
    """保存重启通知信息到 .restart_notify.json。"""

def load_and_send_restart_notifications() -> None:
    """加载并发送重启通知。"""
```

**特性**：
- **持久化通知信息**：将请求者的路由信息（平台、chat_id、thread_id）保存到 `.restart_notify.json`
- **自动通知**：网关重启后自动发送 "Gateway restarted successfully" 消息
- **线程保留**：保留 Telegram topic 或 Discord thread 的 ID，确保通知发送到正确位置
- **自动清理**：发送通知后自动删除 `.restart_notify.json` 文件

### systemd 服务重启 (run.py)

**主要函数**：
```python
def restart_via_systemd() -> bool:
    """通过 systemd 重启网关服务。"""

def is_running_under_systemd() -> bool:
    """检查是否在 systemd 下运行。"""
```

**特性**：
- **服务重启**：使用 `systemctl restart` 替代分离的子进程
- **改进的响应**：提供更好的重启响应和回退说明
- **venv 符号链接保留**：在重映射路径时保持 venv python 符号链接未解析

## 关键依赖与配置

### 依赖项
- **python-telegram-bot**：Telegram Bot API
- **discord.py**：Discord Bot API
- **slack-bolt**：Slack Bot API
- **aiohttp**：异步 HTTP 客户端
- **websockets**：WebSocket 支持

### 配置文件
- `~/.hermes/config.yaml` - 主配置文件
- `~/.hermes/.env` - API 密钥和令牌
- `~/.hermes/sessions/` - 会话持久化

### 环境变量
- `TELEGRAM_BOT_TOKEN` - Telegram bot 令牌
- `DISCORD_BOT_TOKEN` - Discord bot 令牌
- `SLACK_BOT_TOKEN` - Slack bot 令牌
- `WHATSAPP_PHONE_NUMBER_ID` - WhatsApp 电话号码 ID
- `WHATSAPP_ACCESS_TOKEN` - WhatsApp 访问令牌
- `SIGNAL_HTTP_URL` - Signal HTTP 服务 URL

## 数据模型

### 会话数据格式
```python
{
    "session_id": str,
    "platform": str,
    "user_id": str,
    "channel_id": str,
    "created_at": str,
    "updated_at": str,
    "messages": [
        {"role": "user", "content": str, "timestamp": str},
        {"role": "assistant", "content": str, "timestamp": str},
    ],
    "metadata": {
        "title": str,
        "model": str,
        "toolsets": list[str],
    },
}
```

### 消息格式
```python
{
    "platform": str,
    "session_id": str,
    "user_id": str,
    "channel_id": str,
    "message": str,
    "timestamp": str,
    "metadata": dict,
}
```

### 审批请求格式
```python
{
    "platform": str,
    "session_id": str,
    "command": str,
    "args": dict,
    "requested_at": str,
    "expires_at": str,
}
```

## 测试与质量

### 测试文件
- `tests/gateway/test_session.py` - 会话管理测试
- `tests/gateway/test_telegram.py` - Telegram 适配器测试
- `tests/gateway/test_discord.py` - Discord 适配器测试
- `tests/gateway/test_slack.py` - Slack 适配器测试
- `tests/gateway/test_dispatch.py` - 命令调度测试
- `tests/gateway/test_restart_notification.py` - 重启通知机制测试（约 173 个测试用例）
- `tests/gateway/test_run_progress_topics.py` - 工具进度显示测试（约 63 个测试用例）

### 测试覆盖
- **单元测试**：每个适配器都有对应的测试文件
- **集成测试**：测试与真实平台的交互（需要 API 密钥）
- **约 35 个测试文件**覆盖核心功能

### 质量指标
- 高测试覆盖率（~75%+）
- 统一的错误处理
- 详细的日志记录
- 安全的凭据管理

## 常见问题 (FAQ)

### Q: 如何添加新的平台支持？
A:
1. 在 `gateway/platforms/` 中创建新的适配器文件
2. 实现 `BasePlatformAdapter` 接口
3. 在 `gateway/run.py` 中注册适配器
4. 在配置文件中添加平台配置

### Q: 如何处理平台特定的消息格式？
A: 使用适配器的 `format_message()` 方法将 Hermes 响应转换为平台原生格式。

### Q: 如何持久化会话？
A: 使用 `SessionManager` 类，会话自动保存到 `~/.hermes/sessions/` 目录。

### Q: 如何审批危险命令？
A: 网关会发送审批请求，用户通过平台特定的按钮（如 Telegram 内联键盘）审批。

### Q: 如何调试网关问题？
A: 设置 `HERMES_DEBUG=1` 环境变量，查看详细的网关日志。

### Q: 如何支持多实例？
A: 每个实例有独立的 `~/.hermes/` 目录，可以运行多个网关实例。

## 相关文件清单

### 核心文件
- `gateway/__init__.py` - 模块初始化
- `gateway/run.py` - 网关主入口
- `gateway/session.py` - 会话管理
- `gateway/config.py` - 配置加载

### 平台适配器
- `gateway/platforms/__init__.py` - 适配器基类
- `gateway/platforms/telegram.py` - Telegram 适配器
- `gateway/platforms/discord.py` - Discord 适配器
- `gateway/platforms/slack.py` - Slack 适配器
- `gateway/platforms/whatsapp.py` - WhatsApp 适配器
- `gateway/platforms/signal.py` - Signal 适配器
- `gateway/platforms/matrix.py` - Matrix 适配器
- `gateway/platforms/home_assistant.py` - Home Assistant 适配器

### 辅助文件
- `gateway/message_formatting.py` - 消息格式化
- `gateway/approval.py` - 危险命令审批
- `gateway/file_handler.py` - 文件处理
- `gateway/rate_limit.py` - 速率限制

### 测试文件
- `tests/gateway/` - 完整的测试套件

## 变更记录 (Changelog)

### 2026-04-13 16:35:00 - 重启通知与 systemd 集成 🔄
- ✅ **重启通知机制**：`/restart` 命令完成后自动通知请求者
- 📊 **持久化通知信息**：保存路由信息到 `.restart_notify.json`
- 🔧 **systemd 集成**：支持 systemd 服务重启（替代分离的子进程）
- 🛠️ **改进的响应**：提供更好的重启响应和回退说明
- 🔒 **venv 符号链接保留**：在重映射路径时保持 venv python 符号链接未解析
- 🧪 **测试覆盖**：约 173 个重启通知测试用例、63 个工具进度测试用例
- 📖 **文档更新**：添加重启通知机制和 systemd 服务重启文档

### 2026-04-08 - 初始化模块文档 🚀
- ✅ **创建模块文档**：生成 gateway 模块的完整 CLAUDE.md
- 📊 **架构分析**：详细说明平台适配器和会话管理
- 🔧 **接口文档**：记录主要对外接口
- 📖 **使用指南**：提供添加新平台和调试的步骤
- 🎯 **下一步**：需要补充详细的 API 文章和示例代码
