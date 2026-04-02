# 会话管理

## 5.1 概述

**核心文件**: `gateway/session.py` (~40,000 字符)

**功能**: 管理用户与 Agent 之间的对话状态，支持跨平台会话恢复。

---

## 5.2 Session 数据结构

```python
# gateway/session.py
class Session:
    """会话对象"""

    def __init__(
        self,
        id: str,                    # 唯一标识 (platform:user_id)
        platform: str,              # 平台名称
        user_id: str,               # 用户 ID
        messages: List[dict] = None,
        created_at: datetime = None,
        updated_at: datetime = None,
        metadata: dict = None,
    ):
        self.id = id
        self.platform = platform
        self.user_id = user_id
        self.messages = messages or []
        self.created_at = created_at or datetime.now()
        self.updated_at = updated_at or datetime.now()
        self.metadata = metadata or {}

    def add_message(self, role: str, content: str) -> None:
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        self.updated_at = datetime.now()
```

---

## 5.3 SessionStore

```python
# gateway/session.py
class SessionStore:
    """会话存储管理器"""

    def __init__(self, db_path: str = None):
        self._sessions: Dict[str, Session] = {}
        self._db_path = db_path
        self._load_from_disk()

    def get_or_create_session(
        self,
        platform: str,
        user_id: str,
    ) -> Session:
        """获取或创建会话"""
        session_id = f"{platform}:{user_id}"

        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                id=session_id,
                platform=platform,
                user_id=user_id,
            )
            self._save_to_disk(session_id)

        return self._sessions[session_id]

    def list_sessions(self, platform: str = None) -> List[Session]:
        """列出所有会话"""
        if platform:
            return [s for s in self._sessions.values() if s.platform == platform]
        return list(self._sessions.values())
```

---

## 5.4 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 会话类 | `gateway/session.py` | `Session` |
| 会话存储 | `gateway/session.py` | `SessionStore` |
| 消息添加 | `gateway/session.py` | `Session.add_message()` |
| 持久化 | `gateway/session.py` | `_save_to_disk()` |

---

## 5.5 下一步

- [平台适配器](03-platforms.md) - 平台实现详解
- [SessionDB 与记忆](../../06-memory-state/01-sessiondb.md) - SQLite FTS5

---

[← 返回总目录](../SUMMARY.md) | [← Gateway 概述](01-overview.md) | [平台适配器 →](03-platforms.md)
