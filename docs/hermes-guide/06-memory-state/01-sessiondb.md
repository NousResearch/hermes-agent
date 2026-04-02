# SessionDB 与记忆系统

## 6.1 概述

**核心文件**: `hermes_state.py`

**功能**: SQLite + FTS5 实现的会话存储，支持全文搜索和语义搜索。

---

## 6.2 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     记忆层次架构                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: 上下文窗口 (Context Window)                            │
│  - 模型最大上下文长度                                            │
│  - 自动管理                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: SessionDB (hermes_state.py)                            │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  messages   │  │  sessions   │  │  message_fts (FTS5)    │ │
│  │   表        │  │   表        │  │     全文搜索索引         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                  │
│  路径: ~/.hermes/sessions.db                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Memory Tool (tools/memory_tool.py)                     │
│  - Agent 主动读写的长期记忆                                       │
│  - 语义存储                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Honcho (honcho_integration/)                          │
│  - dialectic 用户建模                                           │
│  - 跨会话用户偏好学习                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.3 SessionDB 实现

```python
# hermes_state.py
class SessionDB:
    """
    SQLite + FTS5 会话存储。

    表结构:
    - messages: 消息表
    - sessions: 会话表
    - message_fts: FTS5 全文搜索虚拟表
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path.home() / ".hermes" / "sessions.db"

        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._setup_tables()

    def _setup_tables(self):
        """初始化表结构"""
        # 消息表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """)

        # FTS5 虚拟表
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS message_fts
            USING fts5(content, session_id, tokenize='porter unicode61')
        """)

        # 会话表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                platform TEXT,
                user_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def add_message(self, session_id: str, role: str, content: str, metadata: dict = None):
        """添加消息"""
        cursor = self.conn.execute(
            """INSERT INTO messages (session_id, role, content, metadata)
               VALUES (?, ?, ?, ?)""",
            (session_id, role, content, json.dumps(metadata or {}))
        )

        # 同步到 FTS 表
        self.conn.execute(
            "INSERT INTO message_fts (content, session_id) VALUES (?, ?)",
            (content, session_id)
        )

        self.conn.commit()

    def search(self, query: str, session_id: str = None, limit: int = 10) -> List[dict]:
        """
        全文搜索消息。

        使用 FTS5 的 MATCH 查询。
        """
        if session_id:
            cursor = self.conn.execute(
                """SELECT m.* FROM messages m
                   JOIN message_fts fts ON m.rowid = fts.rowid
                   WHERE message_fts MATCH ? AND m.session_id = ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (query, session_id, limit)
            )
        else:
            cursor = self.conn.execute(
                """SELECT m.* FROM messages m
                   JOIN message_fts fts ON m.rowid = fts.rowid
                   WHERE message_fts MATCH ?
                   ORDER BY m.created_at DESC LIMIT ?""",
                (query, limit)
            )

        return [
            {"id": row[0], "session_id": row[1], "role": row[2],
             "content": row[3], "created_at": row[4]}
            for row in cursor.fetchall()
        ]
```

---

## 6.4 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| SessionDB | `hermes_state.py` | `SessionDB` |
| FTS5 搜索 | `hermes_state.py` | `search()` |
| 消息存储 | `hermes_state.py` | `add_message()` |
| 会话管理 | `hermes_state.py` | `SessionStore` |

---

## 6.5 下一步

- [Memory Tool](02-memory-tool.md) - 长期记忆工具
- [Honcho 集成](03-honcho.md) - 用户建模

---

[← 返回总目录](../SUMMARY.md) | [← Gateway 系统](../../05-gateway/03-platforms.md) | [Memory Tool →](02-memory-tool.md)
