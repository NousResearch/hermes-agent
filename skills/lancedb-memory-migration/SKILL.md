---
name: lancedb-memory-migration
description: "Migrate a Hermes profile's session history from FTS5 (built-in session search) to LanceDB vector memory. Uses unified script with built-in Strip Pipeline — no separate optimization needed. Requires: Ollama with bge-m3:567m, lancedb pip package."
version: 2.0.0
author: Hermes Agent + Kuntao
license: MIT
metadata:
  hermes:
    tags: [memory, lancedb, ollama, migration, sub-agent, vector, fts5]
---

# LanceDB Memory Migration

将 Hermes profile（通常是子 agent profile）的记忆系统从 FTS5（内置会话搜索）迁移到 LanceDB 向量搜索（LanceDB + Ollama 向量）。

## When to Use

- 已有 profile 的历史 session（存在 `state.db` 中）想从 FTS5 会话搜索切换到 LanceDB 向量搜索
- 新建子 agent profile 时，直接配置使用 LanceDB 向量记忆系统（无需迁移）
- 子 agent profile 需要和 default agent 共用同一套记忆架构

## Architecture Overview

```
Session History (state.db)
        ↓  [迁移脚本]
LanceDB Vector DB  ←  Ollama bge-m3:567m
        ↓  [lancedb-embed plugin]
Hermes Agent (vec_memory_* tools)
```

**三层存储：**

| 层 | 组件 | 作用 |
|----|------|------|
| 原始数据 | `state.db` (SQLite) | 原始 session 消息 |
| 向量嵌入 | Ollama + bge-m3:567m | 文本→1024维向量 |
| 向量数据库 | LanceDB (.lance 文件) | ANN 索引检索 |

**两种 profile 场景：**

| 场景 | HERMES_HOME | LanceDB 路径 |
|------|------------|-------------|
| default agent | `~/.hermes/` | `~/.hermes/lance_memory/` |
| 子 agent profile | `~/.hermes/profiles/<name>/` | `~/.hermes/profiles/<name>/lance_memory/` |

> 子 agent 是进程内线程，与父 agent 共享 HERMES_HOME（各自配置文件里的）。因此子 agent 的 LanceDB 路径自动指向各自 profile 下的 `lance_memory/` 目录。

## Prerequisites

### 1. Ollama 服务 + bge-m3:567m 模型

**Step 1a: 启动 Ollama**
```bash
# 检查是否已运行
curl -s http://localhost:11434/api/tags && echo "Ollama already running"

# 如果未运行，启动它（用户环境：WSL2 Ubuntu）
ollama serve &
sleep 5
```

**Step 1b: 检查/拉取模型**
```bash
# 查看已加载模型
ollama list

# 如果没有 bge-m3:567m，拉取（需要能访问 huggingface）
ollama pull bge-m3:567m
```

> **Pitfall — Ollama 在 WSL2 中需要手动启动。** 如果 Ollama 未运行，迁移脚本会直接 abort 并报错 `ERROR: Cannot reach Ollama at http://localhost:11434`。每次重启 WSL 后都需要重新 `ollama serve &`。确保模型已加载后再运行迁移脚本。

### 2. lancedb Python 包

```bash
uv pip install --python ~/.hermes/venv/bin/python lancedb
```

## Step 1: 运行统一迁移脚本

迁移脚本已统一为 `migrate_sessions_to_lancedb.py`，通过 `--profile` 参数指定目标 profile，无需为每个 profile 单独创建脚本。

```bash
# Dry-run 查看待迁移 session
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_sessions_to_lancedb.py \
  --profile <profile_name> --dry-run

# 执行迁移
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_sessions_to_lancedb.py \
  --profile <profile_name>
```

**★ 内置 Strip Pipeline：** 迁移脚本现在在写入 LanceDB 前自动运行 5 阶段 Strip Pipeline（prefix → assistant_frontmatter → trailing → embedded_meta → quality_gate），消除 skill 模板前缀污染。迁移后无需单独运行优化脚本。

**★ Timestamp 修复：** `created_at` 使用消息自身的 timestamp，不再统一用写入时刻的 `time.time()`。

## Step 2: 修改 config.yaml

在目标 profile 的 `config.yaml` 中找到或添加 `memory` 区块：

```yaml
# Memory — 改为 lancedb-embed
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: lancedb-embed
```

**如何定位：**
```bash
grep -n "^memory:" ~/.hermes/profiles/<profile>/config.yaml
```

**如何修改（使用 hermes config 或直接编辑）：**
```bash
hermes config set memory.provider lancedb-embed --profile <profile>
```

> 如果使用 hermes CLI 直接改，不需要重启 gateway。但如果直接编辑 config.yaml，需要重启该 profile 的 gateway。

## Step 3: 配置 lancedb-embed 插件（可选）

`lancedb-embed` 插件会从 `plugins.lancedb-embed` 配置块读取参数。如果不配置，则使用硬编码默认值。

**默认值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_url` | `http://localhost:11434` | Ollama 服务地址 |
| `embedding_model` | `bge-m3:567m` | 嵌入模型名 |
| `lance_dir` | `$HERMES_HOME/lance_memory` | LanceDB 目录（自动解析） |
| `batch_size` | `32` | 每批嵌入数量 |
| `search_top_k` | `5` | 搜索返回数量 |
| `min_content_len` | `50` | 最小内容长度 |

**如果需要自定义，在 profile 的 config.yaml 中添加：**

```yaml
plugins:
  lancedb-embed:
    base_url: http://localhost:11434      # 根据你的环境调整
    embedding_model: bge-m3:567m           # 确保与 Ollama 中实际模型名一致
    lance_dir: $HERMES_HOME/lance_memory   # 自动解析为 profile 的 HERMES_HOME
    batch_size: 32
    search_top_k: 5
```

## Step 4: 执行迁移

### 4.1 Dry-run 验证

```bash
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_sessions_to_lancedb.py \
  --profile <profile_name> --dry-run
```

预期输出：
- 显示找到的 session 数量
- 显示"Already migrated"数量（首次为0）
- 列出将被迁移的 session 及其信息
- 显示 Strip 阶段数和启用数

### 4.2 执行迁移

```bash
~/.hermes/venv/bin/python3 ~/.hermes/scripts/migrate_sessions_to_lancedb.py \
  --profile <profile_name>
```

> ★ 迁移脚本内置 Strip Pipeline，写入的是净化后的内容。迁移完成后会输出 Strip ratio。无需再单独运行 optimize_lance_memory.py。

### 4.3 幂等性

脚本会自动跳过已迁移的 session（通过 session_id 去重）。可以多次运行，不会重复添加。

## Step 5: 全面验证

### 5.1 配置检查

```bash
# 检查 memory.provider
grep -n "provider:" ~/.hermes/profiles/<profile>/config.yaml | grep memory

# 期望输出包含: provider: lancedb-embed
```

### 5.2 LanceDB / Ollama Embed 数据检查

**⚠️ Critical: Use the correct system.** The `vec_memory_*` tools use LanceDB `.lance` format (see `references/lancedb-lance-storage.md`), NOT `ollama_embed.db`.

**LanceDB `.lance` system (correct — use this):**
```bash
# 物理文件
ls ~/.hermes/lance_memory/memories.lance/

# Python inspection
~/.hermes/venv/bin/python -c "
import lancedb, datetime
db = lancedb.connect('/home/ktao/.hermes/lance_memory')
tbl = db.open_table('memories')
df = tbl.to_pandas()
df['ts'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df_sorted = df.sort_values('ts', ascending=False)
print(f'Total: {len(df)} | Distinct sessions: {df[\"session_id\"].nunique()}')
print('Date range:', df['ts'].min().date(), '~', df['ts'].max().date())
print()
print('=== Date distribution ===')
df['date'] = df['ts'].dt.date
for d, c in df['date'].value_counts().sort_index(ascending=False).items():
    print(f'  {d}: {c}')
print()
print('=== Latest 5 ===')
for _, r in df_sorted.head(5).iterrows():
    print(f'  {r[\"ts\"].strftime(\"%Y-%m-%d %H:%M\")}  {r[\"role\"]}')
"

# Schema (用于验证列名)
~/.hermes/venv/bin/python -c "
import lancedb
db = lancedb.connect('/home/ktao/.hermes/lance_memory')
print(db.open_table('memories').schema)
"
```

> **Column name note:** The LanceDB table uses `created_at` (Unix timestamp as double), NOT `timestamp`. The `metadata` column is always `"{}"` (empty JSON). See `references/lancedb-lance-storage.md` for full schema and probe script.

**ollama_embed.db system (legacy — only if you know this is your target):**
```bash
file ~/.hermes/ollama_embed.db
sqlite3 ~/.hermes/ollama_embed.db "SELECT COUNT(*) FROM memories; SELECT MIN(created_at), MAX(created_at) FROM memories;"
```
```

### 5.3 Ollama 模型检查

```bash
# Step 1: 确认 Ollama 进程在跑
curl -s http://localhost:11434/api/tags && echo "Ollama OK"

# Step 2: 确认模型已加载
ollama list | grep bge-m3:567m
```

> 如果 `curl` 报错 "Failed to connect"，先 `ollama serve &` 启动服务。

### 5.4 向量搜索联动测试

```bash
~/.hermes/venv/bin/python -c "
import lancedb, json, requests, numpy as np

OLLAMA_HOST = 'http://localhost:11434'
OLLAMA_MODEL = 'bge-m3:567m'
LANCE_DIR = '/home/ktao/.hermes/lance_memory'  # 或 profiles/<name>/lance_memory

# 获取查询向量
texts = ['<与profile相关的测试查询词>']
resp = requests.post(f'{OLLAMA_HOST}/api/embed', json={'model': OLLAMA_MODEL, 'input': texts}, timeout=60)
emb = np.array(resp.json()['embeddings'][0], dtype=np.float32)

# 搜索（使用 lancedb.connect，NOT lancedb.open）
db = lancedb.connect(LANCE_DIR)
table = db.open_table('memories')
results = table.search(emb).limit(3).to_list()
print(f'Found {len(results)} results:')
for r in results:
    meta = json.loads(r.get('metadata', '{}'))
    print(f'  {r['session_id']} dist={r.get('_distance','N/A')} role={r.get('role')}')
"
```

> **API note:** Use `lancedb.connect()` (not `lancedb.open()`) — the latter raises `AttributeError`.

### 5.5 Gateway 重启（如果需要）

如果 gateway 正在运行且使用了旧的配置，重启使新配置生效：

```bash
# 查找该 profile 的 gateway service 名称
systemctl --user list-units | grep hermes

# 重启（将 <profile> 替换为实际名称）
systemctl --user restart hermes-gateway-<profile>.service
```

## Post-Migration Optimization

**★ Strip Pipeline is now built into the migration script.** Raw content is stripped before writing to LanceDB (5 stages: prefix → assistant_frontmatter → trailing → embedded_meta → quality_gate). The separate `optimize_lance_memory.py` is still available for:

- **Twig-level splitting** — split long sessions into smaller, independently retrievable units
- **Twig-level dedup** — cosine dedup at finer granularity
- **Incremental optimization** — process new sessions only
- **Pattern discovery** — detect new template patterns

Run it periodically after migration if you want Twig splitting:

```bash
~/.hermes/venv/bin/python3 ~/.hermes/scripts/optimize_lance_memory.py \
  --profile <profile_name> --apply --incremental
```

Summary of the optimization problem:
- Skill invocation blocks (~4700 chars each) are identical across ALL sessions
- Raw content embedding → cos ≈ 1.0 between unrelated sessions → search broken
- Fix: Strip Pipeline removes template prefixes at migration time
- Further: Twig split + dedup for fine-grained retrieval

## Pitfalls

### 0. Naive migration stores raw content (FIXED — Strip now built-in)

The unified `migrate_sessions_to_lancedb.py` now applies Strip Pipeline at write time, so migrated content is clean. If you run the old per-profile scripts, the raw content problem still exists — use the new script instead.

### 1. Ollama 模型名不匹配

错误：`WARNING: Model 'bge-m3:567m' not loaded`

原因：Ollama 中注册的模型名与你脚本里写的不一致。

解决：
```bash
# 查看 Ollama 中实际注册的模型名
bash scripts/check_ollama.sh | grep -q "OK" && ollama list | grep "bge-m3:567m"
```
将脚本中 `OLLAMA_MODEL` 改为实际模型名。

### 2. 子 agent 默认不使用 memory

子 agent（`delegate_task`）默认 `skip_memory=True`，跳过了 memory provider 加载。如需子 agent 主动使用向量搜索，需要修改 `delegate_tool.py`：

```python
# tools/delegate_tool.py 第 1105 行附近
# 改 skip_memory=True 为 skip_memory=False
# 同时 DEFAULT_TOOLSETS 加上 "memory"
```

> 注意：这会影响所有子 agent。改动前确认你的需求。

### 3. 向量嵌入超时

大 session（>20000字符）嵌入可能超时。脚本已将 timeout 设为 300 秒。如果仍超时，可能是 Ollama 处理速度慢或网络问题。

### 4. content 拼接格式（与原始 Ollama embed 格式保持一致）

脚本中的拼接逻辑：按 user/assistant 顺序交替拼接，每一对是 `[user]\n用户内容\n[assistant]\n助手内容`。这样生成的内容与 Ollama embed 写入的原始格式一致，搜索时语义一致。

### 6. "No new memories after X date" — Real Usage Drop vs. Write Failure

When the user reports "no new memories written after [date]" but they were using the system, the first step is to check the actual memory data before assuming a bug:

**Step 1: Check the date distribution**
```bash
~/.hermes/venv/bin/python -c "
import lancedb, datetime
db = lancedb.connect('/home/ktao/.hermes/lance_memory')
tbl = db.open_table('memories')
df = tbl.to_pandas()
df['ts'] = df['created_at'].apply(lambda x: datetime.datetime.fromtimestamp(x))
df['date'] = df['ts'].dt.date
for d, c in df['date'].value_counts().sort_index(ascending=False).items():
    print(f'{d}: {c}')
"
```

**If counts drop after a date — this is REAL usage drop, NOT a bug.** Check parallel evidence:
- `gateway.log` inbound message count drops on same dates
- `agent.log` shows fewer "Memory provider registered" events
- Sub-agent profiles show near-zero messages on same dates

**Actual write failures have these signatures (real bugs):**
- `agent.log` shows `WARNING.*session_end batch store failed`
- `agent.log` shows `Connection refused` to Ollama
- Old date counts suddenly decrease (data corruption)

**Do NOT assume the memory system is broken** just because fewer memories appear after a certain date. The date distribution IS the ground truth.

### 8. 时间戳完整性链路（v2 修复 2026-05-17）

**完整链路有 4 个缺口，均已修复：**

| 缺口 | 文件 | 修复内容 |
|------|------|---------|
| 缺口1 | `run_agent.py` | `_sync_external_memory_for_turn()` 新增 `turn_timestamp` 参数，传 `time.time()` |
| 缺口2 | `lancedb-embed/__init__.py` `sync_turn()` | metadata 从 `"{}"` 扩展为含 preview 的 JSON |
| 缺口3 | `lancedb-embed/__init__.py` `on_session_end()` | metadata 扩展含 `user_ts`/`asst_ts`，`created_at` 优先用 user timestamp |
| 缺口4 | `scripts/optimize_lance_memory.py` | 从 `state.db` 读消息时保留 timestamp，写入 `metadata.message_timestamps[]` |

**实时写入的 fallback**：gateway 侧没有把飞书消息到达时间传下来，`run_agent` 层 fallback 到 `time.time()`（turn 完成时刻），这是当前能做到的最好方案。如需精确到消息级，需要 gateway 侧也传 Unix 时间戳格式。

**Metadata 设计原则**：时间戳存 metadata 而非 content，避免污染向量空间。

### 9. Profile 独立性问题

各 profile 的 LanceDB 完全独立。如果多个 profile 想共用同一份记忆数据，需要手动 symlink 或统一 `lance_dir` 配置。

### Storage Architecture (Two Systems)

> **Critical fix (2026-05-16):** The `vec_memory_*` tools use **LanceDB `.lance` format** at `lance_memory/memories.lance`, NOT `ollama_embed.db`. `ollama_embed.db` is a **separate** SQLite store used only by the Ollama embed plugin. Do NOT confuse the two — see `references/actual-storage-format.md` for the authoritative schema.

### System 1 — Ollama Embed SQLite (plugin-only, NOT for vec_memory tools)

```
~/.hermes/ollama_embed.db                    # default profile
~/.hermes/profiles/<name>/ollama_embed.db     # sub-agent profiles
```
- **Format:** SQLite (NOT `.lance` directory)
- **Purpose:** Only used by Ollama embed plugin internally. NOT queried by `vec_memory_*` tools.
- **Tables:** `memories` + `sessions` (BLOB vectors, no ANN index)
- See `references/actual-storage-format.md` Section 1

### System 2 — LanceDB `.lance` (vec_memory tools — THIS IS THE ONE)

```
~/.hermes/lance_memory/memories.lance/        # default profile (confirmed 2026-05-16: 195 items)
~/.hermes/profiles/<name>/lance_memory/        # sub-agent profiles
```
- **Format:** LanceDB `.lance` directory with HNSW index
- **Purpose:** All `vec_memory_*` tools read/write here
- **Schema (confirmed 2026-05-16):**
  - `id`: string (UUID)
  - `content`: string (stored text)
  - `role`: string (`turn` / `session_migrated` / `session_end`)
  - `session_id`: string (e.g. `20260513_223106_8757f640`)
  - `vector`: fixed_size_list&lt;float32&gt;[1024]
  - `created_at`: double (Unix timestamp)
  - `metadata`: string (JSON, currently `{}`)
- See `references/actual-storage-format.md` Section 2 and probe script

## File Locations

| 文件 | 路径 |
|------|------|
| 迁移脚本 | `~/.hermes/scripts/migrate_sessions_to_lancedb.py`（统一脚本，`--profile <name>` 指定目标）|
| 优化脚本 | `~/.hermes/scripts/optimize_lance_memory.py` |
| 验证脚本 | `references/verify-optimization.py`（通过 `skill_view(name='optimize-lance-memory', file_path='references/verify-optimization.py')` 访问） |
| 优化状态 | `~/.hermes/profiles/<profile>/.optimization_state.json` |
| **★ 向量记忆系统架构（权威）** | `references/vector-memory-architecture.md` |
| **Strip Pipeline 架构详解** | `references/strip-pipeline-architecture.md` |
| **优化指南** | `references/optimization-guide.md` |
| **实际存储格式（权威）** | `references/actual-storage-format.md` |
| **Plugin 内部机制** | `references/lancedb-embed-plugin-internals.md` |
| **LanceDB .lance 系统** | `references/lancedb-lance-storage.md` |
| Profile config | `~/.hermes/profiles/<profile>/config.yaml` |
| 原始 session | `~/.hermes/profiles/<profile>/state.db` |
| lancedb-embed 插件 | `~/.hermes/plugins/memory/lancedb-embed/__init__.py` |
| Ollama 服务 | `localhost:11434` |
