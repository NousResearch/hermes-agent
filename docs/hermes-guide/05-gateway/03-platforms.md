# 平台适配器

## 5.1 概述

**基类文件**: `gateway/base.py` (~60,000 字符)

**平台实现**: `gateway/platforms/*.py`

---

## 5.2 基类定义

```python
# gateway/base.py
class Platform(ABC):
    """消息平台抽象基类"""

    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """启动平台监听"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """停止平台监听"""
        pass

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        **kwargs
    ) -> str:
        """发送消息"""
        pass

    @abstractmethod
    async def get_updates(self) -> List[dict]:
        """获取消息更新"""
        pass

    async def handle_update(self, update: dict) -> None:
        """处理收到的更新"""
        # 1. 解析消息内容
        message = self.parse_message(update)

        # 2. 获取或创建会话
        session = await self.get_session(message)

        # 3. 投递给 Agent
        await self.deliver_to_agent(session, message)

    def parse_message(self, update: dict) -> Message:
        """解析平台消息为统一格式"""
        raise NotImplementedError
```

---

## 5.3 Telegram 适配器

```python
# gateway/platforms/telegram.py
class Telegram(Platform):
    """Telegram 平台适配器"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = config["telegram_token"]
        self.api = TelegramBotAPI(self.token)

    async def start(self) -> None:
        """启动 Telegram bot"""
        await self.api.start_polling()
        self._running = True

    async def send_message(
        self,
        chat_id: str,
        text: str,
        **kwargs
    ) -> str:
        """发送 Telegram 消息"""
        message = await self.api.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="Markdown",
            **kwargs
        )
        return str(message.message_id)

    def parse_message(self, update: dict) -> Message:
        """解析 Telegram 更新"""
        message = update.get("message", {})
        return Message(
            platform="telegram",
            chat_id=str(message["chat"]["id"]),
            user_id=str(message["from"]["id"]),
            text=message.get("text", ""),
            message_id=str(message["message_id"]),
        )
```

---

## 5.4 Discord 适配器

```python
# gateway/platforms/discord.py
class Discord(Platform):
    """Discord 平台适配器"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = config["discord_token"]
        self.intents = DiscordIntents.default()

    async def start(self) -> None:
        """启动 Discord bot"""
        self.client = DiscordClient(token=self.token, intents=self.intents)
        await self.client.start()
        self._running = True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        **kwargs
    ) -> str:
        """发送 Discord 消息"""
        message = await self.client.send_message(
            channel=channel_id,
            content=text
        )
        return str(message.id)
```

---

## 5.5 平台特性对比

| 平台 | 消息格式 | 特殊功能 | 限制 |
|------|---------|---------|------|
| Telegram | Markdown/HTML | 键盘、InlineQuery | 单消息 4096 |
| Discord | Markdown | Embeds、组件 | 2000/msg |
| Slack | Block Kit | 线程、模态 | 3000/msg |
| 飞书 | Markdown | 卡片、机器人 | 5000/msg |
| 企业微信 | Markdown | 群聊、应用 | 2048/msg |

---

## 5.6 关键源码索引

| 平台 | 文件 | 行数(估) |
|------|------|---------|
| 基类 | `gateway/base.py` | ~60k |
| Telegram | `gateway/platforms/telegram.py` | ~92k |
| Discord | `gateway/platforms/discord.py` | ~98k |
| Slack | `gateway/platforms/slack.py` | ~37k |
| 飞书 | `gateway/platforms/feishu.py` | ~136k |

---

## 5.7 下一步

- [SessionDB 与记忆](../../06-memory-state/01-sessiondb.md) - SQLite FTS5
- [Memory Tool](../../06-memory-state/02-memory-tool.md) - 长期记忆

---

[← 返回总目录](../SUMMARY.md) | [← 会话管理](02-session.md) | [SessionDB →](../../06-memory-state/01-sessiondb.md)
