# Hermes 向量记忆系统架构

> **版本:** 1.0  
> **日期:** 2026-05-17  
> **状态:** 权威参考文档（整合 lancedb-memory-migration + optimize-lance-memory + lancedb-embed plugin）  
> **目标读者:** 系统维护者、技能开发者

---

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Hermes Agent (run_agent.py)                              │
│  ┌────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐  │
│  │   vec_memory_add   │     │  vec_memory_search │     │   vec_memory_list    │  │
│  │   vec_memory_delete│     │  vec_memory_stats  │     │  (工具层: 5 个 tool) │  │
│  └────────┬───────────┘     └────────┬───────────┘     └──────────┬───────────┘  │
│           │                          │                              │             │
│           └──────────────────────────┼──────────────────────────────┘             │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │              lancedb-embed Plugin (plugins/memory/lancedb-embed/)          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐    │   │
│  │  │  _tool_add()    │  │  _tool_search() │  │  on_session_end()       │    │   │
│  │  │  实时写入入口    │  │  HNSW ANN 检索  │  │  session 结束批量写入   │    │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘    │   │
│  │           │                    │                        │                  │   │
│  │           ▼                    ▼                        ▼                  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐     │   │
│  │  │               Ollama Client (_ollama_embed)                        │     │   │
│  │  │               HTTP POST /api/embed  {model:"bge-m3:567m", input}  │     │   │
│  │  └───────────────────────────────┬──────────────────────────────────┘     │   │
│  └──────────────────────────────────┼──────────────────────────────────────┘   │
└─────────────────────────────────────┼──────────────────────────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │    Ollama Server (localhost:11434)  │
                    │    bge-m3:567m → 1024-dim vector   │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Storage Layer (双系统)                                   │
│                                                                                    │
│  ┌────────────────────────────────┐    ┌─────────────────────────────────────┐   │
│  │  System 2: LanceDB .lance      │    │  System 1: ollama_embed.db (LEGACY) │   │
│  │  ★ 主要系统                    │    │  SQLite, 仅 Ollama embed 插件内部用  │   │
│  │  ★ vec_memory_* tools 读写     │    │  不参与向量搜索                      │   │
│  │  ★ HNSW ANN 索引               │    │  BLOB 存储向量，无 ANN 索引          │   │
│  │  ~/.hermes/lance_memory/       │    │  ~/.hermes/ollama_embed.db           │   │
│  │  Profile: profiles/<n>/        │    │  Profile: profiles/<n>/              │   │
│  │           lance_memory/        │    │           ollama_embed.db            │   │
│  └────────────────────────────────┘    └─────────────────────────────────────┘   │
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  原始数据: state.db (SQLite)                                                 │  │
│  │  表: sessions (id, message_count, started_at) + messages (role, content,      │  │
│  │       timestamp)                                                             │  │
│  │  ★ 迁移脚本的源数据，优化脚本的查询源                                         │  │
│  │  $HERMES_HOME/state.db (default)  /  ~/.hermes/profiles/<n>/state.db          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件一览

| 组件 | 路径 | 作用 |
|------|------|------|
| **Plugin** | `~/.hermes/plugins/memory/lancedb-embed/__init__.py` (765 行) | 5 个 tool 的实现：add/search/list/delete/stats |
| **Ollama 嵌入** | `localhost:11434` / `bge-m3:567m` | 文本 → 1024 维向量 |
| **LanceDB** | `~/.hermes/lance_memory/memories.lance/` | HNSW ANN 索引，向量搜索主力 |
| **state.db** | `$HERMES_HOME/state.db` (default) | 原始 session 消息，迁移/优化源数据 |
| **ollama_embed.db** | `~/.hermes/ollama_embed.db` | ⚠️ 遗留系统，仅插件内部用 |
| **迁移脚本** | `~/.hermes/scripts/migrate_<p>_sessions_to_lancedb.py` | FTS5 → LanceDB 批量迁移 |
| **优化脚本** | `~/.hermes/scripts/optimize_lance_memory.py` | Strip + Twig + 去重 |
| **优化状态** | `~/.hermes/profiles/<p>/.optimization_state.json` | 增量优化进度 |

---

## 2. 双存储系统详解

⚠️ **这是最容易混淆的地方。** 系统中有两个不同的存储，服务于不同目的。

### System 1: ollama_embed.db（遗留 — 仅供插件内部）

```
格式:    SQLite
路径:    ~/.hermes/ollama_embed.db
作用:    仅供 Ollama embed 插件内部记录。不参与向量搜索。
表:      memories (BLOB vector) + sessions
索引:    无 ANN 索引（BLOB 列不可搜索）
```

### System 2: LanceDB .lance（主力 — vec_memory_* 工具读写此系统）

```
格式:    LanceDB .lance 目录 + HNSW 索引
路径:    ~/.hermes/lance_memory/memories.lance/
作用:    所有 vec_memory_add / vec_memory_search / vec_memory_list 工具读写
Schema:
  id              string             UUID
  content         string             存储文本
  role            string             turn | session_end | session_migrated
  session_id      string             来源 session（如 20260513_223106_8757f640）
  vector          list<float32>[1024] bge-m3 嵌入
  created_at      double             Unix timestamp
  metadata        string             JSON（message_timestamps、user_preview 等）
```

**判别规则：**
- 如果路径以 `.lance/` 结尾 → System 2，主力系统
- 如果文件名是 `.db` → System 1，遗留系统
- `vec_memory_*` 工具 100% 走 System 2

---

## 3. 两种写入路径

### 路径 A: 实时写入（Agent 运行时）

```
run_agent.py  turn 完成
    → _sync_external_memory_for_turn(turn_timestamp=time.time())
    → lancedb-embed plugin.sync_turn()
    → Ollama /api/embed
    → LanceDB table.add()

run_agent.py  session 结束
    → memory_manager.on_session_end(messages)
    → lancedb-embed plugin.on_session_end()
    → 每个 message 取 messages[i]["timestamp"]（修复于 2026-05-16）
    → Ollama /api/embed（批量）
    → LanceDB table.add()
```

**关键修复（2026-05-16）：** `on_session_end()` 以前所有行都用写入时刻的 `time.time()`，现已改为读取每条消息自身的 `timestamp`。

### 路径 B: 批量迁移（从 state.db）

```
state.db (SQLite)
    → migrate_<profile>_sessions_to_lancedb.py
    → 读取 sessions + messages
    → process_session() — 过滤、拼接 [user]/[assistant]
    → Ollama /api/embed（批量 4 条）
    → LanceDB table.add()
    → ⚠️ 迁移后必须运行 optimize_lance_memory.py！
```

### 路径 C: 优化覆写（Strip + Twig + 去重）

```
state.db → Strip Pipeline → Twig Split → Deduplication → LanceDB (overwrite)
```

详见第 5 节。

---

## 4. Plugin 内部架构

```
lancedb-embed/__init__.py (765 lines)
│
├── _get_lance_db()          → lancedb.connect()，懒加载
├── _build_schema()          → PyArrow schema（7 字段）
├── _ollama_embed()          → HTTP POST /api/embed（60s 超时）
├── _sanitize_metadata()     → 确保 metadata 是合法 JSON 字符串
│
├── lance_memory 类（继承 MemoryProvider）
│   ├── __init__()           → 从 config.yaml 读配置，连 LanceDB
│   ├── get_tools()          → 注册 5 个 tool schema
│   │
│   ├── sync_turn()          → 每轮 turn 写入（role="turn"）
│   ├── on_session_end()     → session 结束批量写入（role="session_end"）
│   │                          修复: 使用 messages[i].get("timestamp")
│   │
│   ├── _tool_add()          → vec_memory_add 实现
│   ├── _tool_search()       → vec_memory_search 实现（HNSW ANN）
│   ├── _tool_list()         → vec_memory_list 实现
│   ├── _tool_delete()       → vec_memory_delete 实现
│   └── _tool_stats()        → vec_memory_stats 实现
```

### 配置读取

从 `config.yaml` 的 `plugins.lancedb-embed` 块读取，默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `http://localhost:11434` | Ollama 服务 |
| `embedding_model` | `bge-m3:567m` | 嵌入模型 |
| `lance_dir` | `$HERMES_HOME/lance_memory` | LanceDB 目录 |
| `batch_size` | `32` | 每批嵌入数 |
| `search_top_k` | `5` | 搜索返回数 |
| `min_content_len` | `50` | 最小内容长度 |

---

## 5. 迁移流水线

```
state.db ──▶ process_session() ──▶ Ollama embed ──▶ LanceDB
                │
                ├── 过滤：短确认／纯问候
                ├── 拼接：[user]\n...\n[assistant]\n...
                └── ⚠️ 原始内容含 skill 模板前缀（cos≈1.0 问题）

         ⚠️ 迁移后必须优化！
         │
         ▼
state.db ──▶ Strip Pipeline ──▶ Twig Split ──▶ Dedup ──▶ LanceDB (overwrite)
```

### 问题：Naive 迁移导致向量坍塌

```
Session A: [IMPORTANT: skill... 4700 chars] + [assistant] + REAL REPORT 500 chars
Session B: [IMPORTANT: skill... 4700 chars] + [assistant] + REAL REPORT 480 chars
                      ↑── 4700 chars 完全相同 ──↑

Vector A ≈ Vector B  (cos ≈ 1.0 → 搜索失效)
```

**根因：** skill 调用块（`[IMPORTANT: ...]` + YAML frontmatter）占 session 内容的 60-80%，且跨 session 完全相同。

---

## 6. Strip Pipeline（5 阶段）

Strip Pipeline 是 `optimize_lance_memory.py` 的核心，用可插拔的阶段链去除模板噪音。

```
原始 content
  │
  ├── Stage 1: prefix（strip_before）
  │   去除 [IMPORTANT: skill...]---<yaml>--- 前缀块
  │   例：r'\[IMPORTANT:[^\]]*\]\n\n---\n[\s\S]{50,5000}\n---\n+'
  │
  ├── Stage 2: assistant_frontmatter（strip_after_match）
  │   去除 [assistant] 后的 ## Overview / ## When to Use 等节标题
  │   关键：保留 [assistant]\n 前缀，只删除后面的模板标题
  │   Pitfall: 不能用可变宽度 lookbehind，用 strip_after_match action
  │
  ├── Stage 3: trailing（strip_after）
  │   去除文档尾部固定标记：免责声明、尾部 ---、多余空行
  │   ⚠️ 必须用 lookahead (?=[\n\s]*$) 只匹配文档末尾
  │   错误写法: r'^\s*---\s*$.*'  → 会吃掉报告中所有 --- 分割线！
  │   正确写法: r'^\s*---\s*$(?=[\n\s]*$)'
  │
  ├── Stage 4: embedded_meta（replace）
  │   清理内嵌的 UUID、时间戳等元信息
  │
  └── Stage 5: quality_gate（filter）
      过滤过短内容、纯标点、残留模板碎片
```

### Stage 配置格式

```python
{
    "id": "prefix",
    "name": "模板前缀裁剪",
    "enabled": True,
    "type": "regex",           # "regex" | "quality"
    "action": "strip_before",  # strip_before | strip_after_match | strip_after | replace
    "patterns": [...],
    "flags": re.DOTALL | re.IGNORECASE,
    "stop_on_first": True,     # True=命中第一个即停止
}
```

### Actions 说明

| Action | 效果 | 典型用途 |
|--------|------|----------|
| `strip_before` | `text = text[m.end():]` | 去除匹配及之前的所有内容 |
| `strip_after_match` | `text = text[:m.start()] + text[m.end():]` | 去除匹配部分，保留前缀 |
| `strip_after` | `text = text[:last.start()]` | 去除最后匹配位置之后的一切 |
| `replace` | `text = compiled.sub(repl, text)` | 原地替换 |

---

## 7. Twig 分割策略

优化脚本将 session 切分为 **Twig**（最小可检索单元），而非存储整个 session。

| 策略 | 触发条件 | 场景 |
|------|----------|------|
| `pair_uai` | ≥2 对 `[user]/[assistant]` | cron job 中多次独立报告 |
| `section_header` | ≥2 个 `##` 标题 + clean_len > 3000 | 长报告按章节分割 |
| `last_pair_only` | 单对，内容提取完整 | 短 session 只保留最后一对 |
| `no_split` | content < 3000 chars | 短 session 保持完整 |

每个 Twig 携带元数据：`twig_id`、`twig_index`、`twig_count`、`twig_strategy`。

---

## 8. 去重策略

- **阶段**：Strip Pipeline 之后、Twig Split 之后
- **粒度**：Twig 级别（不是 session 级别）
- **阈值**：cos > 0.98（默认）
- **规则**：保留较新的 Twig，删除较旧的

---

## 9. 时间戳完整性链路（v2 修复）

完整链路有 4 个环节，均已于 2026-05-17 修复：

| 环节 | 文件 | 修复内容 |
|------|------|----------|
| ① turn 层 | `run_agent.py` | `_sync_external_memory_for_turn()` 加 `turn_timestamp` 参数 |
| ② sync 层 | `lancedb-embed/__init__.py` `sync_turn()` | metadata 从 `"{}"` 扩展为含 preview 的 JSON |
| ③ session 层 | `lancedb-embed/__init__.py` `on_session_end()` | metadata 含 `user_ts`/`asst_ts`，`created_at` 优先用 user timestamp |
| ④ 迁移/优化层 | `scripts/optimize_lance_memory.py` | 从 `state.db` 读 timestamp → `metadata.message_timestamps[]` |

### Metadata 字段对照表

| 字段 | 类型 | 来源 | 用途 |
|------|------|------|------|
| `created_at` | float (Unix) | 第一条消息 timestamp | 排序 / 范围查询 |
| `message_timestamps[]` | string[] (ISO) | 全部消息 timestamp | "报时间找对话" |
| `user_preview` | string | 第一条 user 摘要 | 快速预览 |
| `asst_preview` | string | 第一条 assistant 摘要 | 快速预览 |
| `user_ts` / `asst_ts` | float | 对应消息 timestamp | Twig 内双时间戳 |
| `topics` | string[] | 分类器 | 主题标签 |
| `recency_weight` | float | 指数衰减计算 | 时间排序 |
| `strip_ratio` | float | Strip Pipeline | 优化质量指标 |

### 核心原则

> **时间戳存 metadata，不存 content。** 这是铁律——content 只放净化后的语义内容，任何时间信息都会污染 bge-m3 向量空间，稀释语义区分度。

---

## 10. 完整文件路径表

```
# ─ 源码 ─
~/.hermes/plugins/memory/lancedb-embed/__init__.py    # Plugin 实现 (765 lines)

# ─ 配置 ─
$HERMES_HOME/config.yaml                                # default agent
$HERMES_HOME/profiles/<name>/config.yaml                # 子 agent
  → memory.provider: lancedb-embed
  → plugins.lancedb-embed: {base_url, embedding_model, lance_dir, ...}

# ─ 数据 ─
$HERMES_HOME/state.db                                     # default session 原始数据
~/.hermes/profiles/<name>/state.db                    # 子 profile session 原始数据
~/.hermes/lance_memory/memories.lance/                 # default 向量数据库
~/.hermes/profiles/<name>/lance_memory/                # 子 profile 向量数据库
~/.hermes/ollama_embed.db                              # 遗留系统（可忽略）

# ─ 脚本 ─
~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py   # 迁移脚本
~/.hermes/scripts/optimize_lance_memory.py                   # 优化脚本（v2.4.1+）

# ─ 状态 ─
~/.hermes/.optimization_state.json                     # default 优化状态
~/.hermes/profiles/<name>/.optimization_state.json    # 子 profile 优化状态

# ─ 技能文档 ─
~/.hermes/skills/lancedb-memory-migration/SKILL.md    # 迁移技能
~/.hermes/skills/lancedb-memory-migration/references/ # 参考文档
  ├── vector-memory-architecture.md                    # ★ 本文档
  ├── lancedb-lance-storage.md                        # LanceDB 存储格式
  ├── actual-storage-format.md                        # ollama_embed.db 格式
  ├── optimization-guide.md                           # 优化指南
  ├── lancedb-embed-plugin-internals.md               # Plugin 内部机制
  └── strip-pipeline-architecture.md                  # Strip Pipeline 详解
~/.hermes/skills/mlops/optimize-lance-memory/SKILL.md # 优化技能

# ─ 部署输出 ─
~/hermes_out/vector-memory-architecture.md            # 用户可访问副本
```

---

## 11. Profile 独立性

每个 profile 的向量记忆系统完全独立，data 和 state.db 互不干扰：

| Agent 类型 | LanceDB 路径 | state.db 路径 |
|-----------|-------------|---------------|
| default agent | `~/.hermes/lance_memory/` | `$HERMES_HOME/state.db` |
| 子 agent（`--profile <name>`） | `~/.hermes/profiles/<name>/lance_memory/` | `~/.hermes/profiles/<name>/state.db` |

> ⚠️ Profile 间无配置继承。每个 profile（包括 default）必须独立配置 `memory.provider: lancedb-embed`。  
> ⚠️ `lancedb-embed` 插件代码（`~/.hermes/plugins/memory/lancedb-embed/__init__.py`）是所有 profile 共用的，修改即全局生效。  
> ⚠️ 如果想多个 profile 共用同一份记忆，需手动 symlink 或统一 `lance_dir` 配置。

---

## 12. 常规操作流程

### 首次设置（新 profile）

```bash
# 1. 安装依赖
uv pip install --python ~/.hermes/venv/bin/python lancedb

# 2. 启动 Ollama
ollama serve &
ollama pull bge-m3:567m

# 3. 配置 config.yaml
hermes config set memory.provider lancedb-embed --profile <name>

# 4. 迁移历史 session
python ~/.hermes/scripts/migrate_<name>_sessions_to_lancedb.py --dry-run
python ~/.hermes/scripts/migrate_<name>_sessions_to_lancedb.py

# 5. 优化向量质量（必须！）
python ~/.hermes/scripts/optimize_lance_memory.py --profile <name> --apply

# 6. 验证
python ~/.hermes/skills/mlops/optimize-lance-memory/references/verify-optimization.py --profile <name>
```

### 日常维护

```bash
# 增量优化新 session
python ~/.hermes/scripts/optimize_lance_memory.py --profile <name> --apply --incremental

# 检查向量区分度
python -c "
import lancedb, requests, numpy as np
db = lancedb.connect('<lance_dir>')
tbl = db.open_table('memories')
df = tbl.to_pandas()
sample = df.sample(min(3, len(df)))
texts = sample['content'].tolist()
resp = requests.post('http://localhost:11434/api/embed', json={'model':'bge-m3:567m','input':texts}, timeout=60)
embs = np.array(resp.json()['embeddings'])
for i in range(len(embs)):
    for j in range(i+1, len(embs)):
        cos = np.dot(embs[i], embs[j])/(np.linalg.norm(embs[i])*np.linalg.norm(embs[j]))
        print(f'cos({i},{j}): {cos:.3f}  {\"✅\" if cos<0.95 else \"⚠️ 区分度不足\"}')"
```

---

## 13. 常见问题排查

### Q: 搜索返回结果全部相似（cos≈1.0）
**原因：** 迁移后未优化，skill 模板前缀污染向量空间。  
**解决：** 运行 `optimize_lance_memory.py --profile <name> --apply`。

### Q: "No new memories after X date"
**先确认是使用量下降还是写入故障：**  
- 检查 `gateway.log` 入站消息数同时期是否下降 → 使用量下降  
- 检查 `agent.log` 是否有 `WARNING.*session_end batch store failed` → 写入故障  
- 检查 Ollama 是否可达：`curl -s http://localhost:11434/api/tags`

### Q: "lance is not fork-safe" 警告
`optimize_lance_memory.py` v2.4.1+ 已内置 bootstrap 自愈代码。警告无害，可忽略。

### Q: "memories table not found, skipping verify"
优化脚本末尾自动验证的已知 bug，数据已正确写入。用独立的 verify 脚本即可。

### Q: 子 agent LanceDB 数据损坏

如果碰到类似 `RuntimeError: lance error: Not found: .../memories.lance/data/...`，说明数据文件缺失：

```bash
# 重建该 profile 的 LanceDB
rm -rf ~/.hermes/profiles/<profile_name>/lance_memory/
python ~/.hermes/scripts/optimize_lance_memory.py --profile <profile_name> --apply
```

---

## 14. 技能间关系

```
lancedb-memory-migration (迁移技能)
    │
    ├── 引用 → optimize-lance-memory (优化技能)
    │           迁移后必须运行
    │
    ├── 引用 → hermes-agent-skill-authoring (技能规范)
    │
    └── references/
          ├── vector-memory-architecture.md    ★ 本文档（统一架构）
          ├── lancedb-lance-storage.md         LanceDB 存储
          ├── actual-storage-format.md         ollama_embed.db 格式
          ├── optimization-guide.md            优化流程
          ├── lancedb-embed-plugin-internals.md Plugin 内部
          └── strip-pipeline-architecture.md   Strip 阶段详解

optimize-lance-memory (优化技能)
    │
    └── references/
          └── fork-safety-bootstrap.md         多进程启动修复
```
