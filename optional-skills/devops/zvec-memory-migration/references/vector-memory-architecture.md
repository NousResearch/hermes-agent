# Hermes Vector Memory System Architecture

> **Version:** 1.0  
> **Date:** 2026-05-17  
> **Status:** Authoritative reference document (consolidates lancedb-memory-migration + optimize-lance-memory + the lancedb-embed plugin)  
> **Target readers:** system maintainers, skill developers

---

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Hermes Agent (run_agent.py)                              │
│  ┌────────────────────┐     ┌────────────────────┐     ┌──────────────────────┐  │
│  │   vec_memory_add   │     │  vec_memory_search │     │   vec_memory_list    │  │
│  │   vec_memory_delete│     │  vec_memory_stats  │     │  (tool layer: 5 tools)│  │
│  └────────┬───────────┘     └────────┬───────────┘     └──────────┬───────────┘  │
│           │                          │                              │             │
│           └──────────────────────────┼──────────────────────────────┘             │
│                                      ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │              lancedb-embed Plugin (plugins/memory/lancedb-embed/)          │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐    │   │
│  │  │  _tool_add()    │  │  _tool_search() │  │  on_session_end()       │    │   │
│  │  │  real-time write│  │  HNSW ANN search│  │  session-end batch write│    │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘    │   │
│  │           │                    │                        │                  │   │
│  │           ▼                    ▼                        ▼                  │   │
│  │  ┌──────────────────────────────────────────────────────────────────┐     │   │
│  │  │               Ollama Client (_ollama_embed)                        │     │   │
│  │  │               HTTP POST /api/embed  {model:"bge-m3:latest", input}  │     │   │
│  │  └───────────────────────────────┬──────────────────────────────────┘     │   │
│  └──────────────────────────────────┼──────────────────────────────────────┘   │
└─────────────────────────────────────┼──────────────────────────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────┐
                    │    Ollama Server (localhost:11434)  │
                    │    bge-m3:latest → 1024-dim vector   │
                    └─────────────────┬──────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       Storage Layer (dual system)                                 │
│                                                                                    │
│  ┌────────────────────────────────┐    ┌─────────────────────────────────────┐   │
│  │  System 2: LanceDB .lance      │    │  System 1: ollama_embed.db (LEGACY) │   │
│  │  ★ Primary system              │    │  SQLite, Ollama embed plugin only   │   │
│  │  ★ vec_memory_* read/write     │    │  Not involved in vector search      │   │
│  │  ★ HNSW ANN index              │    │  BLOB vectors, no ANN index         │   │
│  │  ~/.hermes/lance_memory/       │    │  ~/.hermes/ollama_embed.db           │   │
│  │  Profile: profiles/<n>/        │    │  Profile: profiles/<n>/              │   │
│  │           lance_memory/        │    │           ollama_embed.db            │   │
│  └────────────────────────────────┘    └─────────────────────────────────────┘   │
│                                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  Raw data: state.db (SQLite)                                                 │  │
│  │  Tables: sessions (id, message_count, started_at) + messages (role, content, │  │
│  │       timestamp)                                                             │  │
│  │  ★ source data for migration scripts, query source for optimization scripts │  │
│  │  ~/.hermes/state.db (default)  /  ~/.hermes/profiles/<n>/state.db            │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Path | Role |
|------|------|------|
| **Plugin** | `~/.hermes/plugins/memory/lancedb-embed/__init__.py` (765 lines) | Implementation of the 5 tools: add/search/list/delete/stats |
| **Ollama embedding** | `localhost:11434` / `bge-m3:latest` | text → 1024-dim vector |
| **LanceDB** | `~/.hermes/lance_memory/memories.lance/` | HNSW ANN index, the main vector-search workhorse |
| **state.db** | `~/.hermes/state.db` (default) | Raw session messages, source data for migration/optimization |
| **ollama_embed.db** | `~/.hermes/ollama_embed.db` | ⚠️ legacy system, plugin-internal use only |
| **Migration script** | `~/.hermes/scripts/migrate_<p>_sessions_to_lancedb.py` | FTS5 → LanceDB bulk migration |
| **Optimization script** | `~/.hermes/scripts/optimize_lance_memory.py` | Strip + Twig + dedup |
| **Optimization state** | `~/.hermes/profiles/<p>/.optimization_state.json` | Incremental optimization progress |

---

## 2. The Two Storage Systems in Detail

⚠️ **This is the most easily confused part.** There are two distinct storage systems in this setup, serving different purposes.

### System 1: ollama_embed.db (legacy — plugin-internal only)

```
Format:  SQLite
Path:    ~/.hermes/ollama_embed.db
Role:    Ollama embed plugin-internal bookkeeping only. Not involved in vector search.
Tables:  memories (BLOB vector) + sessions
Index:   no ANN index (the BLOB column is not searchable)
```

### System 2: LanceDB .lance (primary — the system the vec_memory_* tools read/write)

```
Format:  LanceDB .lance directory + HNSW index
Path:    ~/.hermes/lance_memory/memories.lance/
Role:    read/written by all vec_memory_add / vec_memory_search / vec_memory_list tools
Schema:
  id              string             UUID
  content         string             stored text
  role            string             turn | session_end | session_migrated
  session_id      string             source session (e.g. 20260513_223106_8757f640)
  vector          list<float32>[1024] bge-m3 embedding
  created_at      double             Unix timestamp
  metadata        string             JSON (message_timestamps, user_preview, etc.)
```

**Discrimination rules:**
- If the path ends with `.lance/` → System 2, the primary system
- If the filename is `.db` → System 1, the legacy system
- The `vec_memory_*` tools use System 2 100% of the time

---

## 3. The Two Write Paths

### Path A: Real-time writes (agent runtime)

```
run_agent.py  turn completed
    → _sync_external_memory_for_turn(turn_timestamp=time.time())
    → lancedb-embed plugin.sync_turn()
    → Ollama /api/embed
    → LanceDB table.add()

run_agent.py  session ended
    → memory_manager.on_session_end(messages)
    → lancedb-embed plugin.on_session_end()
    → for each message, read messages[i]["timestamp"] (fixed 2026-05-16)
    → Ollama /api/embed (batched)
    → LanceDB table.add()
```

**Key fix (2026-05-16):** `on_session_end()` previously used the write-time `time.time()` for all rows; it now reads each message's own `timestamp`.

### Path B: Bulk migration (from state.db)

```
state.db (SQLite)
    → migrate_<profile>_sessions_to_lancedb.py
    → read sessions + messages
    → process_session() — filter, concatenate [user]/[assistant]
    → Ollama /api/embed (batches of 4)
    → LanceDB table.add()
    → ⚠️ after migration you MUST run optimize_lance_memory.py!
```

### Path C: Optimization overwrite (Strip + Twig + dedup)

```
state.db → Strip Pipeline → Twig Split → Deduplication → LanceDB (overwrite)
```

See Section 5 for details.

---

## 4. Plugin Internal Architecture

```
lancedb-embed/__init__.py (765 lines)
│
├── _get_lance_db()          → lancedb.connect(), lazy-loaded
├── _build_schema()          → PyArrow schema (7 fields)
├── _ollama_embed()          → HTTP POST /api/embed (60s timeout)
├── _sanitize_metadata()     → ensures metadata is a valid JSON string
│
├── lance_memory class (inherits MemoryProvider)
│   ├── __init__()           → read config from config.yaml, connect to LanceDB
│   ├── get_tools()          → register the 5 tool schemas
│   │
│   ├── sync_turn()          → per-turn write (role="turn")
│   ├── on_session_end()     → session-end batch write (role="session_end")
│   │                          fix: use messages[i].get("timestamp")
│   │
│   ├── _tool_add()          → vec_memory_add implementation
│   ├── _tool_search()       → vec_memory_search implementation (HNSW ANN)
│   ├── _tool_list()         → vec_memory_list implementation
│   ├── _tool_delete()       → vec_memory_delete implementation
│   └── _tool_stats()        → vec_memory_stats implementation
```

### Configuration Reading

Read from the `plugins.lancedb-embed` block in `config.yaml`; defaults:

| Parameter | Default | Description |
|------|--------|------|
| `base_url` | `http://localhost:11434` | Ollama service |
| `embedding_model` | `bge-m3:latest` | Embedding model |
| `lance_dir` | `$HERMES_HOME/lance_memory` | LanceDB directory |
| `batch_size` | `32` | Embeddings per batch |
| `search_top_k` | `5` | Number of search results |
| `min_content_len` | `50` | Minimum content length |

---

## 5. Migration Pipeline

```
state.db ──▶ process_session() ──▶ Ollama embed ──▶ LanceDB
                │
                ├── filter: short acknowledgements / pure greetings
                ├── concatenate: [user]\n...\n[assistant]\n...
                └── ⚠️ raw content contains skill template prefixes (the cos≈1.0 problem)

         ⚠️ optimization is mandatory after migration!
         │
         ▼
state.db ──▶ Strip Pipeline ──▶ Twig Split ──▶ Dedup ──▶ LanceDB (overwrite)
```

### Problem: Naive Migration Causes Vector Collapse

```
Session A: [IMPORTANT: skill... 4700 chars] + [assistant] + REAL REPORT 500 chars
Session B: [IMPORTANT: skill... 4700 chars] + [assistant] + REAL REPORT 480 chars
                      ↑── the 4700 chars are completely identical ──↑

Vector A ≈ Vector B  (cos ≈ 1.0 → search breaks down)
```

**Root cause:** skill invocation blocks (`[IMPORTANT: ...]` + YAML frontmatter) account for 60–80% of session content and are completely identical across sessions.

---

## 6. Strip Pipeline (5 Stages)

The Strip Pipeline is the core of `optimize_lance_memory.py`; it removes template noise with a pluggable chain of stages.

```
raw content
  │
  ├── Stage 1: prefix (strip_before)
  │   removes the [IMPORTANT: skill...]---<yaml>--- prefix block
  │   e.g. r'\[IMPORTANT:[^\]]*\]\n\n---\n[\s\S]{50,5000}\n---\n+'
  │
  ├── Stage 2: assistant_frontmatter (strip_after_match)
  │   removes the ## Overview / ## When to Use section headers that follow [assistant]
  │   key point: keep the [assistant]\n prefix, delete only the template headers after it
  │   pitfall: variable-width lookbehind cannot be used; use the strip_after_match action
  │
  ├── Stage 3: trailing (strip_after)
  │   removes fixed end-of-document markers: disclaimers, trailing ---, excess blank lines
  │   ⚠️ must use the lookahead (?=[\n\s]*$) to match only the end of the document
  │   wrong: r'^\s*---\s*$.*'  → would swallow every --- separator in the report!
  │   correct: r'^\s*---\s*$(?=[\n\s]*$)'
  │
  ├── Stage 4: embedded_meta (replace)
  │   cleans embedded UUIDs, timestamps and other meta information
  │
  └── Stage 5: quality_gate (filter)
      filters out too-short content, pure punctuation, leftover template fragments
```

### Stage Configuration Format

```python
{
    "id": "prefix",
    "name": "Template prefix strip",
    "enabled": True,
    "type": "regex",           # "regex" | "quality"
    "action": "strip_before",  # strip_before | strip_after_match | strip_after | replace
    "patterns": [...],
    "flags": re.DOTALL | re.IGNORECASE,
    "stop_on_first": True,     # True = stop on first match
}
```

### Actions Explained

| Action | Effect | Typical use |
|--------|------|----------|
| `strip_before` | `text = text[m.end():]` | remove the match and everything before it |
| `strip_after_match` | `text = text[:m.start()] + text[m.end():]` | remove the matched part, keep the prefix |
| `strip_after` | `text = text[:last.start()]` | remove everything after the last match |
| `replace` | `text = compiled.sub(repl, text)` | in-place substitution |

---

## 7. Twig Split Strategy

The optimization script splits sessions into **Twigs** (minimum retrievable units) rather than storing the whole session.

| Strategy | Trigger condition | Scenario |
|------|----------|------|
| `pair_uai` | ≥2 `[user]/[assistant]` pairs | multiple independent reports inside a cron job |
| `section_header` | ≥2 `##` headers + clean_len > 3000 | split long reports by section |
| `last_pair_only` | single pair, content fully extracted | short session, keep only the last pair |
| `no_split` | content < 3000 chars | short session kept intact |

Each Twig carries metadata: `twig_id`, `twig_index`, `twig_count`, `twig_strategy`.

---

## 8. Deduplication Strategy

- **Stage**: after the Strip Pipeline, after Twig Split
- **Granularity**: Twig level (not session level)
- **Threshold**: cos > 0.98 (default)
- **Rule**: keep the newer Twig, delete the older one

---

## 9. Timestamp Integrity Chain (v2 fixes)

The complete chain has 4 links, all fixed as of 2026-05-17:

| Link | File | Fix |
|------|------|----------|
| ① turn layer | `run_agent.py` | `_sync_external_memory_for_turn()` gains a `turn_timestamp` parameter |
| ② sync layer | `lancedb-embed/__init__.py` `sync_turn()` | metadata expanded from `"{}"` to JSON including previews |
| ③ session layer | `lancedb-embed/__init__.py` `on_session_end()` | metadata includes `user_ts`/`asst_ts`; `created_at` prefers the user timestamp |
| ④ migration/optimization layer | `scripts/optimize_lance_memory.py` | read timestamps from `state.db` → `metadata.message_timestamps[]` |

### Metadata Field Reference

| Field | Type | Source | Purpose |
|------|------|------|------|
| `created_at` | float (Unix) | first message timestamp | sorting / range queries |
| `message_timestamps[]` | string[] (ISO) | all message timestamps | "find the conversation by reported time" |
| `user_preview` | string | first user summary | quick preview |
| `asst_preview` | string | first assistant summary | quick preview |
| `user_ts` / `asst_ts` | float | corresponding message timestamps | dual timestamps within a Twig |
| `topics` | string[] | classifier | topic tags |
| `recency_weight` | float | exponential decay computation | time-based ordering |
| `strip_ratio` | float | Strip Pipeline | optimization quality metric |

### Core Principle

> **Store timestamps in metadata, never in content.** This is an iron rule — content holds only sanitized semantic content; any time information pollutes the bge-m3 vector space and dilutes semantic separation.

---

## 10. Complete File Path Table

```
# ─ source ─
~/.hermes/plugins/memory/lancedb-embed/__init__.py    # Plugin implementation (765 lines)

# ─ config ─
$HERMES_HOME/config.yaml                                # default agent
$HERMES_HOME/profiles/<name>/config.yaml                # sub-agent
  → memory.provider: lancedb-embed
  → plugins.lancedb-embed: {base_url, embedding_model, lance_dir, ...}

# ─ data ─
~/.hermes/state.db                                     # default session raw data
~/.hermes/profiles/<name>/state.db                    # sub-profile session raw data
~/.hermes/lance_memory/memories.lance/                 # default vector database
~/.hermes/profiles/<name>/lance_memory/                # sub-profile vector database
~/.hermes/ollama_embed.db                              # legacy system (can be ignored)

# ─ scripts ─
~/.hermes/scripts/migrate_<profile>_sessions_to_lancedb.py   # migration script
~/.hermes/scripts/optimize_lance_memory.py                   # optimization script (v2.4.1+)

# ─ state ─
~/.hermes/.optimization_state.json                     # default optimization state
~/.hermes/profiles/<name>/.optimization_state.json    # sub-profile optimization state

# ─ skill docs ─
~/.hermes/skills/lancedb-memory-migration/SKILL.md    # migration skill
~/.hermes/skills/lancedb-memory-migration/references/ # reference docs
  ├── vector-memory-architecture.md                    # ★ this document
  ├── lancedb-lance-storage.md                        # LanceDB storage format
  ├── actual-storage-format.md                        # ollama_embed.db format
  ├── optimization-guide.md                           # optimization guide
  ├── lancedb-embed-plugin-internals.md               # plugin internals
  └── strip-pipeline-architecture.md                  # Strip Pipeline in detail
~/.hermes/skills/mlops/optimize-lance-memory/SKILL.md # optimization skill

# ─ deployment output ─
# (2026-09-07 cleanup: the user-accessible old copy under hermes out was deleted; this file is the single authoritative version)
```

---

## 11. Profile Independence

Each profile's vector memory system is fully independent; data and state.db do not interfere with each other:

| Agent type | LanceDB path | state.db path |
|-----------|-------------|---------------|
| default agent | `~/.hermes/lance_memory/` | `~/.hermes/state.db` |
| sub-agent (`--profile <name>`) | `~/.hermes/profiles/<name>/lance_memory/` | `~/.hermes/profiles/<name>/state.db` |

> ⚠️ No configuration inheritance between profiles. Every profile (including default) must independently configure `memory.provider: lancedb-embed`.  
> ⚠️ The `lancedb-embed` plugin code (`~/.hermes/plugins/memory/lancedb-embed/__init__.py`) is shared by all profiles; changes take effect globally.  
> ⚠️ If you want multiple profiles to share the same memory, you must manually symlink or unify the `lance_dir` configuration.

---

## 12. Routine Operations

### First-Time Setup (new profile)

```bash
# 1. Install dependencies
uv pip install --python ~/.hermes/venv/bin/python lancedb

# 2. Start Ollama
ollama serve &
ollama pull bge-m3:latest

# 3. Configure config.yaml
hermes config set memory.provider lancedb-embed --profile <name>

# 4. Migrate historical sessions
python ~/.hermes/scripts/migrate_<name>_sessions_to_lancedb.py --dry-run
python ~/.hermes/scripts/migrate_<name>_sessions_to_lancedb.py

# 5. Optimize vector quality (mandatory!)
python ~/.hermes/scripts/optimize_lance_memory.py --profile <name> --apply

# 6. Verify
python ~/.hermes/skills/mlops/optimize-lance-memory/references/verify-optimization.py --profile <name>
```

### Day-to-Day Maintenance

```bash
# Incrementally optimize new sessions
python ~/.hermes/scripts/optimize_lance_memory.py --profile <name> --apply --incremental

# Check vector separation
python -c "
import lancedb, requests, numpy as np
db = lancedb.connect('<lance_dir>')
tbl = db.open_table('memories')
df = tbl.to_pandas()
sample = df.sample(min(3, len(df)))
texts = sample['content'].tolist()
resp = requests.post('http://localhost:11434/api/embed', json={'model':'bge-m3:latest','input':texts}, timeout=60)
embs = np.array(resp.json()['embeddings'])
for i in range(len(embs)):
    for j in range(i+1, len(embs)):
        cos = np.dot(embs[i], embs[j])/(np.linalg.norm(embs[i])*np.linalg.norm(embs[j]))
        print(f'cos({i},{j}): {cos:.3f}  {\"✅\" if cos<0.95 else \"⚠️ insufficient separation\"}')"
```

---

## 13. Troubleshooting FAQ

### Q: Search results are all similar to each other (cos≈1.0)
**Cause:** no optimization after migration; skill template prefixes pollute the vector space.  
**Fix:** run `optimize_lance_memory.py --profile <name> --apply`.

### Q: "No new memories after X date"
**First determine whether it is declining usage or a write failure:**  
- check whether the inbound message count in `gateway.log` dropped in the same period → declining usage  
- check `agent.log` for `WARNING.*session_end batch store failed` → write failure  
- check whether Ollama is reachable: `curl -s http://localhost:11434/api/tags`

### Q: "lance is not fork-safe" warning
`optimize_lance_memory.py` v2.4.1+ has built-in bootstrap self-healing code. The warning is harmless and can be ignored.

### Q: "memories table not found, skipping verify"
A known bug in the auto-verification at the end of the optimization script; the data was written correctly. Just use the standalone verify script.

### Q: Corrupted sub-agent LanceDB data

If you hit something like `RuntimeError: lance error: Not found: .../memories.lance/data/...`, data files are missing:

```bash
# Rebuild this profile's LanceDB
rm -rf ~/.hermes/profiles/<profile_name>/lance_memory/
python ~/.hermes/scripts/optimize_lance_memory.py --profile <profile_name> --apply
```

---

## 14. Relationships Between Skills

```
lancedb-memory-migration (migration skill)
    │
    ├── refs → optimize-lance-memory (optimization skill)
    │           must be run after migration
    │
    ├── refs → hermes-agent-skill-authoring (skill authoring spec)
    │
    └── references/
          ├── vector-memory-architecture.md    ★ this document (unified architecture)
          ├── lancedb-lance-storage.md         LanceDB storage
          ├── actual-storage-format.md         ollama_embed.db format
          ├── optimization-guide.md            optimization workflow
          ├── lancedb-embed-plugin-internals.md plugin internals
          └── strip-pipeline-architecture.md   Strip stages in detail

optimize-lance-memory (optimization skill)
    │
    └── references/
          └── fork-safety-bootstrap.md         multi-process startup fix
```
