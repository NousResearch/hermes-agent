---
name: memory-system-architecture
description: MemPalace 7层记忆系统架构和工具实现详情
category: memory
version: 3.2
---

# MemPalace 7层记忆系统架构

## 核心路径

- **KG**: `/Users/mars/.mempalace_hermes/knowledge_graph.sqlite3`
- **ChromaDB**: `/Users/mars/.mempalace_hermes/`
- **Plugin**: `plugins/memory/mempalace/__init__.py`
- **Identity**: `/Users/mars/.mempalace_hermes/identity.txt`

## 架构层级

```
L-WM  Working Memory  │ In-memory LRU cache, 50 recent turns
L0    Identity Layer  │ identity.txt
L1    Narrative       │ Auto-generated essential story summary
L2    Semantic        │ ChromaDB drawers (facts, preferences)
L3    Episodic        │ Raw conversation logs with speaker/time/topic
L4    Procedural      │ Step sequences, workflows, code patterns
L5    Knowledge Graph │ Temporal, typed, inverse relations
L6    Episode Index   │ ChromaDB episodes collection
L7    Predictive       │ Proactive prediction retrieval
L8    Meta-Memory     │ Reflection, consolidation, aliasing, import/export
```

## 工具列表 (26个)

### 核心记忆工具
| 工具 | 功能 |
|------|------|
| `mempalace_wakeup` | 加载 L0+L1 记忆 |
| `mempalace_search` | ChromaDB 语义搜索 + L0/L1 展示 |
| `mempalace_add_drawer` | 存储记忆 (L0/L1/L2 分层 + simhash去重) |
| `mempalace_status` | 统计 palace 状态 |
| `mempalace_list_wings` | 列出所有 wings |

### Working Memory
| 工具 | 功能 |
|------|------|
| `mempalace_get_working_memory` | 读取 WM |
| `mempalace_search_working_memory` | WM 搜索 |

### KG 工具
| 工具 | 功能 |
|------|------|
| `mempalace_kg_query` | KG 查询 (entity as_of) |
| `mempalace_kg_add` | KG 添加 (自动inverse) |
| `mempalace_kg_add_typed` | 带类型的 KG 添加 |
| `mempalace_kg_invalidate` | 失效 KG 三元组 |
| `mempalace_kg_stats` | KG 统计 |
| `mempalace_kg_query_decomposed` | 复杂查询分解规划 |
| `mempalace_kg_belief_history` | 查看事实演化历史 |

### 跨系统 & 预测
| 工具 | 功能 |
|------|------|
| `mempalace_cross_palace_search` | 跨 palace 搜索 |
| `mempalace_proactive_predict` | 主动预测相关记忆 |

### OpenClaw 只读
| 工具 | 功能 |
|------|------|
| `openclaw_wakeup` | OpenClaw L0+L1 |
| `openclaw_search` | OpenClaw ChromaDB 搜索 |
| `openclaw_status` | OpenClaw 状态 |
| `openclaw_kg_query` | OpenClaw KG 查询 |

### L8 元记忆 (v3.0新增)
| 工具 | 功能 |
|------|------|
| `mempalace_memory_reflection` | 自我审计 (orphaned entities, dangling rels, duplicates, stale) |
| `mempalace_consolidate` | 重要性衰减 + L2 清除 |
| `mempalace_kg_alias_add` | 添加跨 palace 实体别名 |
| `mempalace_kg_alias_resolve` | 解析实体别名 |
| `mempalace_export` | 导出 JSON 备份 |
| `mempalace_import` | 导入 JSON 备份 |

## KG Schema

**triples 表** (15列):
```
id, subject, predicate, object, valid_from, valid_to,
confidence, source_closet, source_file, extracted_at,
inverse_predicate, context, source, subject_type, object_type
```

**belief_history 表** (12列):
```
entity, predicate, old_value, new_value, change_type,
changed_at, source, confidence, context, valid_from, valid_to
```

**entity_aliases 表** (6列):
```
id, entity, alias, entity_type, palace, created_at
```

**change_type**: created, updated, corrected, invalidated, reinstated

**PREDICATE_INVERSES**: 23对 inverse 谓词

## 重要实现细节

### 字符串格式化 — 核心原则
**所有 handler 脚本统一用 `str.replace()` 模式，绝对禁止 f-string 多行脚本。**

原因：Python 在解析时就计算 f-string 的 `{var}` 表达式，从**外层词法作用域**取值，不是在字符串被使用时从局部作用域取值。这导致几乎所有看起来"正常"的 f-string 多行脚本都会在解析时报 `NameError`。

```python
# ❌ 错误 — NameError: name 'subject' is not defined
script = f"SELECT * FROM triples WHERE subject='{subject}'"

# ✅ 正确 — str.replace 模式
script = "SELECT * FROM triples WHERE subject='REPLACEME_S'".replace("REPLACEME_S", subject)
```

### LIKE 通配符转义 (`%%` 问题)
当 SQL `LIKE '%keyword%'` 放在 `%`-format 字符串内时，`%` 会被 Python 的 `%` 格式化消耗。正确做法：

```python
# ❌ 错误 — %k, %w 被 Python %-format 吃掉
script = "WHERE subject LIKE '%%%s%%'" % keyword

# ✅ 正确 — str.replace 模式，完全避免 %
script = "WHERE subject LIKE 'REPLACEME_KW'".replace("REPLACEME_KW", keyword)
```

如果必须用 `%%` 转义：
```python
# 先把 % 替换为 %%，Python %-format 后 %% → %
where_clause_esc = where_clause.replace("%", "%%")
script = "WHERE subject LIKE '%s'" % where_clause_esc  # Python: %% → %
```

### `.format()` 陷阱
`.format()` 也会在解析时扫描 `{identifier}` 模式。**如果 content/L0/L1 包含 `{` 字符**，`.format()` 会报 `KeyError`。解决：

```python
# ❌ 错误 — content 中的 {name} 被 .format() 解析
script = "content='{content}'".format(content=content)

# ✅ 正确 — 先转义，再用 %(name)s 位置参数
l0_esc = l0.replace('"', '\\"')   # 双重转义
script = "l0='%(l0_esc)s'".format(l0_esc=l0_esc)
```

### 模块级 import
Handler 在**插件进程**中运行时的 import 是模块级生效的。但如果 handler 脚本通过 `subprocess.run(['/usr/bin/python3', '-c', script])` 执行（绕过 venv Python 3.14 dataclass bug），子进程只有 Python stdlib，**不会继承模块级 import**。

```python
# ❌ 错误 — _run_python 子进程没有 mempalace 模块的 import
script = "from mempalace import ..."  # 子进程没有 mempalace！

# ✅ 正确 — 子进程只 import stdlib（sqlite3, json, datetime 等）
script = "import sqlite3, json, datetime"
```

### `import zlib` 的位置
`_handle_add_drawer` 调用 `zlib.crc32()` **在构建脚本字符串之前**（用于 deduplication check），此时代码运行在插件进程中，不是子进程。所以 `import zlib` 必须放在**模块顶部**（line 7），不能放在子进程脚本里。

### SQL NULL
SQL 的 `NULL` 必须用 Python 的 `None`（传入 sqlite3 会自动转为 SQL NULL）。使用字符串 `'NULL'` 会变成文本 "NULL"，导致 `IS NULL` 查询失败。

```python
# ❌ 错误
cur.execute("INSERT ... VALUES(?, NULL, ...)", [id, 'NULL', ...])

# ✅ 正确
cur.execute("INSERT ... VALUES(?, ?, ...)", [id, None, ...])
```

### PRIMARY KEY 不能为空字符串
`triples.id` 是 PRIMARY KEY，不能为空字符串 `''`。使用 `uuid.uuid4()` 生成有效 UUID。

```python
# ❌ 错误 — PRIMARY KEY 约束失败
(id='', subject=..., ...)

# ✅ 正确
import uuid
(id=str(uuid.uuid4()), subject=..., ...)
```

### `valid_to` 有效期语义
- `valid_to = None` → SQL NULL → **有效**三元组
- `valid_to = ''` → 空字符串 → **已失效**（`'' IS NULL` 为 FALSE）
- 查询有效三元组：`WHERE valid_to IS NULL OR valid_to = ''`

### `_detect_language` 函数位置
此函数被 `_handle_add_drawer` 和 `_handle_kg_add_typed` 的脚本字符串**在解析时**引用。必须定义在**这两个 handler 之前**，否则报 `NameError`。

### 有效期
- `valid_to = None` (NULL) 表示有效，`''` (空字符串) 在无效化时使用
- 查询条件: `valid_to IS NULL OR valid_to = ''`

### ChromaDB 元数据键
- wing, room, hall, importance, language, source_file
- created_at, last_accessed, access_count
- content_hash (zlib.crc32 32-bit)
- l0 (约30字符摘要), l1 (约300字符摘要)
- original_length, l2_evicted, l2_evicted_at

### 语言检测
- `_detect_language()`: 返回 zh/en/mixed/unknown
- 中文 > 30% → zh，英文 > 30% → en，两者都高 → mixed

### Deduplication
- ChromaDB: `content_hash = zlib.crc32(content)` (32位)
- KG: exact `(subject, predicate, object)` 匹配

### Contradiction Detection
- `_store_triple_with_inverse()` 在插入前检查 `(S, P, old_O)` 是否存在
- 若存在且 new_O ≠ old_O：自动 invalidate 旧三元组，belief_history 记录为 'corrected'
- 也 invalidate 旧 inverse triple

### L0/L1/L2 分层
- L0: ~30 字符，按词边界截断
- L1: ~300 字符，按词边界截断
- L2: 原始全文 (最多8000字符存入 ChromaDB)
- 手动指定: `l0=` / `l1=` 参数优先

### Consolidation
- 30天未访问 → importance -0.2 (>0.7 高优先级保护)
- 90天未访问 → L2 清除，只保留 L0/L1 (>0.7 高优先级保护)

### Reflection 检测类型
1. orphaned_entities (实体无关系)
2. dangling_relations (关系指向不存在实体)
3. duplicate_drawers (simhash diff ≤ 5)
4. stale_high_importance (>0.7 90天未访问)
5. inverse_relation_mismatch (inverse_predicate 值错误)

### Import/Export
- Export: KG triples + belief_history + ChromaDB collections metadata + entity_aliases
- Import: dry-run 模式可用，重复检测通过 content_hash (ChromaDB) 和 exact match (KG)
- ChromaDB 原始向量不导出 (只导出 documents + metadatas)

## 调用方式

```bash
# 通过 Hermes agent
openclaw agent --agent main --message "mempalace_search query=城哥项目" --local

# 直接 Python
cd /Users/mars/hermes-agent
source .venv/bin/activate
python -c "from plugins.memory.mempalace import get_tool_schemas; ..."
```
