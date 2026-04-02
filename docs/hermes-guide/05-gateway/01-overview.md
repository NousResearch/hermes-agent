# Gateway 系统架构

## 5.1 概述

**核心文件**: `gateway/run.py` (~287,000 字符)

**功能**: 多平台消息网关，支持 Telegram、Discord、Slack、飞书等 15+ 消息平台的集成。

---

## 5.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gateway 架构                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       gateway/run.py                             │
│                      (Gateway 主循环)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   session.py    │ │   delivery.py   │ │    hooks.py     │
│  会话管理       │ │   消息投递       │ │   钩子系统      │
└─────────────────┘ └─────────────────┘ └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     gateway/base.py                              │
│                    (Platform 基类)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   telegram.py   │ │   discord.py    │ │    slack.py     │
│   (~92k)       │ │   (~98k)        │ │   (~37k)       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   feishu.py    │ │   wecom.py      │ │    email.py     │
│   (~136k)      │ │   (~53k)        │ │   (~23k)       │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 5.3 消息流转

```
┌─────────────────────────────────────────────────────────────────┐
│                      消息流转流程                                 │
└─────────────────────────────────────────────────────────────────┘

用户 ──► Telegram/Discord/... ──► Platform Adapter
                                          │
                                          ▼
                                   gateway/run.py
                                   (消息解析 & 路由)
                                          │
                                          ▼
                                   session.py
                                   (会话关联)
                                          │
                                          ▼
                                   run_agent.py
                                   (AIAgent 处理)
                                          │
                                          ▼
                                   delivery.py
                                   (消息投递)
                                          │
                                          ▼
                                   Platform Adapter
                                          │
                                          ▼
                                   用户
```

---

## 5.4 核心组件

### 5.4.1 Platform 基类

```python
# gateway/base.py
class Platform(ABC):
    """消息平台基类"""

    @abstractmethod
    async def send_message(self, message: Message) -> str:
        """发送消息"""
        pass

    @abstractmethod
    async def get_updates(self) -> List[Update]:
        """获取更新"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动平台"""
        pass

    async def handle_update(self, update: Update) -> None:
        """处理更新"""
        # 1. 解析消息
        # 2. 关联会话
        # 3. 投递给 Agent
        pass
```

### 5.4.2 会话管理

```python
# gateway/session.py
class SessionStore:
    """
    会话存储。

    管理用户与 Agent 之间的对话状态，
    支持跨平台会话恢复。
    """

    def get_or_create_session(
        self,
        platform: str,
        user_id: str
    ) -> Session:
        """获取或创建会话"""
        session_id = f"{platform}:{user_id}"
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                id=session_id,
                platform=platform,
                user_id=user_id,
                messages=[],
                created_at=datetime.now(),
            )
        return self._sessions[session_id]

    def add_message(self, session_id: str, message: dict) -> None:
        """添加消息到会话"""
        session = self._sessions[session_id]
        session.messages.append(message)
```

---

## 5.5 支持的平台

| 平台 | 文件 | 状态 |
|------|------|------|
| Telegram | `gateway/platforms/telegram.py` (~92k) | 活跃 |
| Discord | `gateway/platforms/discord.py` (~98k) | 活跃 |
| Slack | `gateway/platforms/slack.py` (~37k) | 活跃 |
| 飞书 | `gateway/platforms/feishu.py` (~136k) | 活跃 |
| 企业微信 | `gateway/platforms/wecom.py` (~53k) | 活跃 |
| 钉钉 | `gateway/platforms/dingtalk.py` (~12k) | 活跃 |
| WhatsApp | `gateway/platforms/whatsapp.py` (~34k) | 活跃 |
| Email | `gateway/platforms/email.py` (~23k) | 活跃 |
| Signal | `gateway/platforms/signal.py` (~31k) | 活跃 |
| Matrix | `gateway/platforms/matrix.py` (~48k) | 活跃 |
| Mattermost | `gateway/platforms/mattermost.py` (~26k) | 活跃 |
| Webhook | `gateway/platforms/webhook.py` | 活跃 |

---

## 5.6 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 网关主循环 | `gateway/run.py` | `GatewayRunner` |
| Platform 基类 | `gateway/base.py` | `Platform` |
| 会话管理 | `gateway/session.py` | `SessionStore` |
| 消息投递 | `gateway/delivery.py` | `DeliveryService` |
| Telegram | `gateway/platforms/telegram.py` | `Telegram` |
| Discord | `gateway/platforms/discord.py` | `Discord` |

---

## 5.7 下一步

- [会话管理](02-session.md) - session.py 详解
- [平台适配器](03-platforms.md) - 平台实现详解
- [记忆与状态管理](../../06-memory-state/01-sessiondb.md) - SessionDB

---

[← 返回总目录](../SUMMARY.md) | [← CLI 系统](../../04-cli-system/05-skin.md) | [会话管理 →](02-session.md)
