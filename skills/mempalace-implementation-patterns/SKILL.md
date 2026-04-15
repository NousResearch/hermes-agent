---
name: mempalace-implementation-patterns
description: MemPalace 插件实现中的字符串脚本构建模式、Python f-string 作用域坑、SQL LIKE 通配符转义
category: devops
version: 1.1
---

# MemPalace 实现模式与坑

## 核心文件
- Plugin: `plugins/memory/mempalace/__init__.py` (~3159 行)
- KG: `/Users/mars/.mempalace_hermes/knowledge_graph.sqlite3`
- ChromaDB: `/Users/mars/.mempalace_hermes/`

## 字符串脚本构建：REPLACEME 模式

所有在 `_run_python()` 子进程中执行的 SQL/ChromaDB 脚本，**必须**用 `str.replace("REPLACEME_TOKEN", value)` 模式，禁止嵌套 `%`-format 或 f-string。

### 正确方式
```python
script = (
    "import sqlite3, json\n"
    "conn = sqlite3.connect('REPLACEME_KG')\n"
    "cur = conn.cursor()\n"
    "cur.execute(\"SELECT * FROM triples WHERE subject=?\", ('REPLACEME_SUBJECT',))\n"
    "rows = cur.fetchall()\n"
    "print(rows)\n"
).replace("REPLACEME_KG", kg_path.replace("'", "\\'")).replace(
    "REPLACEME_SUBJECT", subject.replace("'", "\\'")
)
```

### 错误方式 1：嵌套 f-string
```python
# ❌ Python 在 PARSE TIME 求值 {var}，导致 NameError
script = f"""
import sqlite3
conn = sqlite3.connect('{kg_path}')
cur.execute("SELECT * FROM triples WHERE subject='{subject}'")
"""
# 若 subject 包含单引号，更难调试
```

### 错误方式 2：嵌套 %-format
```python
# ❌ 两个 %-format 嵌套，% 会被消耗
script = """
import sqlite3
conn = sqlite3.connect('%s')
cur.execute("SELECT * FROM triples WHERE subject='%s'" %% (kg_path, subject))
""" % outer_dict
```

## Python F-string 作用域坑

**Python 在解析多行字符串时立即求值 `{var}`**，不是在调用时。

```python
lang = detect_language(content)

script = f"""
import chromadb
col.query(query_texts=['{content[:50]}'])  # ❌ 若 content 含 {lang} 形式的文本会崩溃
"""
```

正确做法：提前把所有变量替换掉，或用 `.replace()`：
```python
script = (
    "import chromadb\n"
    "col.query(query_texts=['REPLACEME_CONTENT'])\n"
).replace("REPLACEME_CONTENT", content[:50].replace("'", "\\'"))
```

## SQL LIKE 通配符与 % 转义

```python
# ❌ '%s' 中的 % 会被外部 %-format 消耗
script = "SELECT * FROM triples WHERE subject LIKE '%%s'" % keyword

# ✅ 用 REPLACEME 模式
script = (
    "SELECT * FROM triples WHERE subject LIKE '%s'"  # 保持原样
).replace("%s", keyword)  # 替换为带通配符的值
```

但注意：如果用 `WHERE subject LIKE 'REPLACEME_KEYWORD'`，然后 `.replace("REPLACEME_KEYWORD", "'%" + keyword + "%'")`，结果是 `WHERE subject LIKE '%keyword%'`，这是正确的。

## 方法链换行语法错误

```python
# ❌ SyntaxError: .replace() 返回字符串，不能换行后再 .replace()
script = "....".replace("A", "B")
             .replace("C", "D")  # ❌

# ✅ 所有 replace 在一行内
script = "....".replace("A", "B").replace("C", "D")

# ✅ 或者用临时变量
script = "...."
script = script.replace("A", "B")
script = script.replace("C", "D")
```

## SQL IN 子句与 tuple 格式化

```python
# ✅ tuple(params) 作为 %s 参数是有效的
params = ('Mars', 'Chen', 'Wei')
script = "SELECT * FROM triples WHERE subject IN %s"
# Python % 格式化: → "SELECT * FROM triples WHERE subject IN ('Mars', 'Chen', 'Wei')"
# SQLite 正确解释
```

## valid_to = NULL vs ''

```python
# ✅ 有效三元组
cur.execute("INSERT INTO triples VALUES(...)", (sid, subject, predicate, obj, valid_from, None, ...))
# None → SQLite NULL

# ✅ 查询有效三元组
cur.execute("SELECT * FROM triples WHERE valid_to IS NULL OR valid_to = ''")

# ✅ 无效化（设置过期时间）
cur.execute("UPDATE triples SET valid_to=? WHERE id=?", (today, triple_id))
# today = datetime.now().isoformat()
```

## ChromaDB 元数据写入

```python
script = (
    "import chromadb\n"
    "client = chromadb.PersistentClient(path='REPLACEME_PALACE')\n"
    "col = client.get_collection('REPLACEME_COL')\n"
    "col.add(\n"
    "    ids=['REPLACEME_ID'],\n"
    "    documents=['REPLACEME_DOC'],\n"
    "    metadatas=[{\n"
    "        'wing': 'REPLACEME_WING',\n"
    "        'importance': REPLACEME_IMP,\n"
    "        'language': 'REPLACEME_LANG',\n"
    "        'l0': 'REPLACEME_L0',\n"
    "        'l1': 'REPLACEME_L1',\n"
    "        'content_hash': 'REPLACEME_HASH',\n"
    "        'original_length': REPLACEME_ORIG_LEN,\n"
    "        'created_at': 'REPLACEME_NOW',\n"
    "        'last_accessed': 'REPLACEME_NOW',\n"
    "        'access_count': 0,\n"
    "    }]\n"
    ")\n"
    "print('ok')\n"
).replace("REPLACEME_PALACE", palace_path.replace("'", "\\'")).replace(
    "REPLACEME_COL", col_name.replace("'", "\\'")
).replace("REPLACEME_ID", drawer_id.replace("'", "\\'")).replace(
    "REPLACEME_DOC", doc.replace("'", "\\'")[:8000]
).replace("REPLACEME_WING", wing.replace("'", "\\'")).replace(
    "REPLACEME_IMP", str(importance)
).replace("REPLACEME_LANG", lang).replace(
    "REPLACEME_L0", l0[:50].replace("'", "\\'")
).replace("REPLACEME_L1", l1[:300].replace("'", "\\'")
).replace("REPLACEME_HASH", content_hash).replace(
    "REPLACEME_ORIG_LEN", str(len(content))
).replace("REPLACEME_NOW", datetime.now().isoformat())
```

## zlib crc32 作为 lightweight dedup hash

```python
import zlib
content_hash = str(zlib.crc32(content.encode("utf-8")) & 0xFFFFFFFF)
# 32位足够用于 ChromaDB 文档去重
```

## Simhash (用于相似度检测)

```python
import hashlib

def simhash(s: str) -> int:
    sig = [0] * 8
    words = s.lower().split()
    for w in words:
        h = hashlib.md5(w.encode()).digest()
        for i in range(8):
            sig[i] += 1 if h[i] > 127 else -1
    return sum((1 << i) if s > 0 else 0 for i, s in enumerate(sig))

# Hamming distance 阈值: diff <= 5 = 非常相似
diff = bin(h1 ^ h2).count('1')
if diff <= 5:
    # duplicate group
```

## _run_python 模式

```python
def _run_python(script: str, timeout: int = 10) -> tuple[int, str, str]:
    import subprocess
    result = subprocess.run(
        ['/usr/bin/python3', '-c', script],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr
```

## 关键陷阱总结

| 坑 | 原因 | 解法 |
|----|------|------|
| `{var}` in multi-line string | Python parse-time evaluation | 用 `.replace()` 或提前格式化 |
| `%%` 转义混乱 | 嵌套 %-format | 用 `str.replace()` 替代 |
| `WHERE x IN %s` 失败 | 以为需要 `%%` | 直接用 `tuple(params)`，不需要转义 |
| `valid_to = ''` 无效化失败 | `'' IS NULL` 为 FALSE | 用 `None` (NULL) |
| 方法链换行 | 语法不支持 | 全部在一行 `.replace()` |
| `NULL` in Python code | `NULL` 不是有效 Python | 改为 Python `None` |
| `''` as PRIMARY KEY id | 空字符串冲突 | 用 `uuid.uuid4()` 生成 |
| `_handle_kg_belief_history` f-string | 内部 WHERE 子句用外部 `where` 变量 | 改用完整脚本字符串传入 |
| ChromaDB 路径 | `~/.mempalace_hermes/palace`（注意 .palace 子目录） | 不能写成 `~/.mempalace_hermes/` |
| MemPalaceMemoryProvider 调用 | instance method，非 module function | `p=Provider(); p.initialize(); p.handle_tool_call()` |
|| `col.get(limit=N)` 返回最老 N 条 | ChromaDB 按插入顺序，返回最旧条目 | 获取全部，按 timestamp 倒序，取前 N 条 ||
|| `_restore_episodes()` daemon 线程 | `initialize()` 返回时 WM 仍为空 | 改为同步调用，不在后台线程 ||
|| `_episode_store(topic=None)` 崩溃 | `topic.replace()` 对 None 调用 | `(topic or "").replace()` 兜底 ||

### Gateway 日志调试：发现 CLI 测不出的问题

CLI 测试正常 ≠ Gateway 进程正常。Gateway 用 Python 3.14 独立进程，加载自己的 MemPalace 实例。

**关键发现**：这次所有严重 bug 都是从 `gateway.error.log` 发现的，CLI 测试全部通过：
- CLI 测试时 `topic` 从未传 None
- Gateway 真实运行时 WorkingMemory 有机会以 `topic=None` 调用 `_episode_store`

**必须同时检查两个日志**：
```bash
# Gateway 错误日志
grep -i 'mempalace\|memory.*error\|episode.*error\|chroma.*error' \
  /Users/mars/.hermes/logs/gateway.error.log

# 搜索 AttributeError
grep "AttributeError" /Users/mars/.hermes/logs/gateway.error.log

# 查看 memory 相关的 Memory updated 确认正常运行
grep "Memory updated" /Users/mars/.hermes/logs/gateway.log

# Gateway 进程是否加载了 MemPalace
ps aux | grep hermes-gateway | grep -v grep
```

### 验证命令
```bash
cd ~/hermes-agent && source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'plugins/memory')
import mempalace
p=mempalace.MemPalaceMemoryProvider()
p.initialize('test')
print(len(p.get_tool_schemas()), 'tools')
"
