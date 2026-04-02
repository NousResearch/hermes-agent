# Memory Tool

## 6.1 概述

**核心文件**: `tools/memory_tool.py` (~22,000 字符)

**功能**: Agent 主动读写的长期记忆系统，支持语义存储和检索。

---

## 6.2 工具定义

| 工具名 | 功能 | 签名 |
|--------|------|------|
| `MemoryRead` | 读取记忆 | `(query: str) -> str` |
| `MemoryWrite` | 写入记忆 | `(content: str, tags?: list) -> str` |
| `MemoryDelete` | 删除记忆 | `(memory_id: str) -> str` |
| `MemoryList` | 列出记忆 | `(query?: str) -> str` |

---

## 6.3 核心实现

```python
# tools/memory_tool.py
class MemoryStore:
    """记忆存储"""

    def __init__(self, db_path: str = None):
        self.db = SessionDB(db_path)

    def write(self, content: str, tags: List[str] = None) -> str:
        """写入记忆"""
        memory_id = str(uuid.uuid4())

        self.db.conn.execute(
            """INSERT INTO memories (id, content, tags, created_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
            (memory_id, content, json.dumps(tags or []))
        )
        self.db.conn.commit()

        return memory_id

    def search(self, query: str, limit: int = 5) -> List[dict]:
        """语义搜索记忆"""
        # 使用嵌入向量进行相似度搜索
        # 或使用 FTS5 全文搜索
        return self.db.search(query, limit=limit)
```

---

## 6.4 关键源码索引

| 功能 | 文件 | 关键类/函数 |
|------|------|------------|
| 记忆存储 | `tools/memory_tool.py` | `MemoryStore` |
| 读取记忆 | `tools/memory_tool.py` | `MemoryRead` |
| 写入记忆 | `tools/memory_tool.py` | `MemoryWrite` |

---

## 6.5 下一步

- [Honcho 集成](03-honcho.md) - dialectic 用户建模
- [Skills 架构](../../07-skills/01-overview.md) - 技能系统

---

[← 返回总目录](../SUMMARY.md) | [← SessionDB](01-sessiondb.md) | [Honcho →](03-honcho.md)
