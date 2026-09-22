---
name: zvec-memory-migration
description: Deploy, migrate, health-check and repair the Zvec memory backend for Hermes Agent — full playbook for switching profiles to Zvec, including LanceDB data migration, index repair, and 12-point functional verification.
version: 3.4.0
author: kuntao2011
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [vector-database, zvec, lancedb, migration, memory]
    category: devops
    requires_toolsets: [terminal]
    related_skills: []
---

# Zvec Vector DB — Deployment, Migration and Operations Playbook

The complete operations manual for the Hermes memory system (memory-zvec).
Covers deployment, migration, health checks, fault diagnosis, index repair, and data backup/restore.

> **Merge history**: v3.0.0 merged in the useful content of `memory-lancedb-admin` (ops checks)
> and `lancedb-memory-migration` (LanceDB migration).
> The original skills were archived and deleted.

---

## Memory System Architecture

### Dual-Storage Architecture (⚠️ Common Misconception)

Hermes uses a **dual-storage** system; both layers **store automatically and simultaneously**:

| Layer | Component | Purpose | Auto/Manual |
|----|------|------|-----------|
| Full history | `state.db` (SQLite/FTS5) | All messages (user/AI/tool calls) | Auto (written every turn) |
| Semantic retrieval | Zvec collection | Vector embeddings + FTS index | Auto (plugin hooks) |

**memory-zvec plugin auto-sync mechanism** (orchestrated on the Hermes side, see [`references/memory-write-hooks.md`](references/memory-write-hooks.md)):
- `sync_turn(user_content, assistant_content)`: real-time store after every conversation turn (role="turn")
  - Called by `run_agent._mirror_to_memory()` after each turn completes
  - Executed in a background thread via `MemoryManager._submit_background()`, without blocking the conversation
  - **Interrupted turns are skipped and not written** (source: `if interrupted: return`)
- `on_session_end(messages)`: batch store at session end (role="session_end")
  - Triggered by `commit_memory_session()` or `shutdown_memory_provider()`
  - Trigger points: `/new`, `/reset`, CLI exit, gateway session expiry
  - Walks the entire messages list, extracts all user+assistant pairs, one record per pair
  - `commit_memory_session()`: calls only `on_session_end`, does not tear down the provider (session rotation)
  - `shutdown_memory_provider()`: calls `on_session_end` + `shutdown_all()` (true shutdown)

**⚠️ Dual-write mechanism**: the same conversation may be stored as two records (role="turn" + role="session_end"); this is not a conflict.
`sync_turn` is the real-time guarantee, `on_session_end` is the fallback — if a given sync_turn fails due to an Ollama outage,
on_session_end still back-fills it.

**⚠️ Do not confuse**: manually calling `vec_memory_add` is an **explicit addition**, not a substitute for automatic storage.
Automatic storage goes through plugin hooks with the full conversation as content; manual additions are suited for storing refined knowledge snippets.

**⚠️ Interactive CLI sessions do not attach external providers**: on the CLI side `skip_memory=self.ignore_rules` (`cli_agent_setup_mixin.py`);
even with `memory.provider: memory-zvec` configured, CLI sessions load only the built-in MEMORY.md and **do not load memory-zvec** —
the logs contain no `Memory provider 'memory-zvec' registered/activated` and the agent has no `vec_memory_*` tools.
Therefore: no matter how many turns a CLI session (including `-z` one-shot runs) chats, it will **never** produce Zvec records; verifying
"auto-write every turn" must be done with a gateway (Feishu/CRON) session. Do not keep retrying via CLI, or you will misdiagnose the hooks as broken.


### Storage File Locations

```
~/.hermes/                              # default profile
├── state.db                            # full session history
├── 记忆数据库/zvec_memory/memories/    # Zvec collection
└── 记忆数据库/lance_memory/             # [archived] old LanceDB data (kept as backup)

~/.hermes/profiles/<name>/              # sub profile
├── state.db                            # full session history
├── 记忆数据库/zvec_memory/memories/    # Zvec collection
└── 记忆数据库/lance_memory/             # [archived] old LanceDB data
```

> **⚠️ Business vector stores**: some profiles also have domain-data vector stores such as `lancedb_news/` and
> `lancedb_analysis/`, fully independent from session memory. See
> [`references/multi-profile-business-lancedb.md`](references/multi-profile-business-lancedb.md) for details.

---

## Architectural Prerequisite: Hermes Plugin/Skill Profile Isolation

**Key concept**: Hermes uses the `HERMES_HOME` environment variable to isolate each profile's plugins and skills.

```
HERMES_HOME（无 profile / default） = ~/.hermes/
HERMES_HOME（有 profile 时）        = ~/.hermes/profiles/<profile_name>/
```

> **⚠️ Note**: the default profile has no `~/.hermes/profiles/default/` directory.
> Its `HERMES_HOME` is simply `~/.hermes/`. Migration and verification scripts
> must special-case `default` (see the pitfall log §8).

> **⚠️ $HERMES_HOME resolution trap**: each profile's systemd service sets
> `Environment="HERMES_HOME=/home/<user>/.hermes/profiles/<name>"`,
> so `$HERMES_HOME/...` in config.yaml expands at runtime to **that profile's own directory**.
> Multiple profiles' config.yaml files appear to contain identical `$HERMES_HOME/...` paths,
> but their runtime resolutions **differ**. Verification method:
> ```bash
> for p in default chip_expert financial_expert health_manager zunhunfan; do
>   pid=$(pgrep -f "hermes.*$p" 2>/dev/null | head -1)
>   [ -n "$pid" ] && cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep HERMES_HOME
> done
> ```

Plugin scan path: `$HERMES_HOME/plugins/`
Skill scan path: `$HERMES_HOME/skills/`

**This means**: a plugin placed under `~/.hermes/plugins/` is discovered only by the profile-less root gateway;
every gateway started with `--profile` will not scan it.

---

## Step 1: Install the memory-zvec Plugin Itself (User-Level Fork: memory-zvec)

> **Canonical layout as of 2026-09-14**. The plugin is a local fork (upstream NousResearch/hermes-agent
> never shipped memory-zvec). Version history: two rounds of in-tree patches (0910 optimize_fix,
> 0911 shutdown race) → v1.1.0 migration + drain → v1.2.x idle release → v1.3.x handle sharing
> + hardening fixes. The **canonical source** lives solely in the Hermes root directory (the default profile's discovery root):

```
~/.hermes/plugins/memory-zvec/     ← canonical source (the only editable copy, v1.3.1)
├── __init__.py                       ← ZvecMemoryProvider + lock governance
└── plugin.yaml                       ← name: memory-zvec (config key retained)
```

The config block key is still `plugins.memory-zvec` (hard-coded in code for backward compatibility), and the data still lives at
`$HERMES_HOME/记忆数据库/zvec_memory/memories` — **zero migration from the old version**.

### Prerequisites

1. **Zvec Python package**: install it into the hermes-agent venv
   ```bash
   ~/.hermes/hermes-agent/venv/bin/pip install zvec>=0.5.0
   # verify
   ~/.hermes/hermes-agent/venv/bin/python -c "import zvec; print(zvec.__version__)"
   ```

2. **Embedding model (local or hosted)**: the plugin calls an
   **Ollama-compatible API** (`{base_url}/api/embed`). The default is local
   Ollama with `bge-m3` (1024-dim):
   ```bash
   ollama list | grep bge-m3
   # if not present
   ollama pull bge-m3:latest
   ```
   A hosted/online endpoint works too — implement (or proxy) Ollama's
   `/api/embed` and point the plugin's `base_url` at it; the vector dimension
   must match the configured `vector_dim`.

---

## Step 2: Cross-Profile Deployment (⚠️ Physical Copies — Not Symlinks, and Never In-Tree)

> **Optional — multi-profile fleets only.** A single-profile setup needs only
> Step 1 + Step 3: install the plugin once in that profile and configure it.
> Read this step only if you want the same memory backend in several profiles
> (each profile keeps its own physical copy, config and data — nothing is
> shared between profiles).

> **Iron rule: the memory provider's discovery root is profile-scoped.** `~/.hermes/plugins/<name>/` is visible
> only to **default**; a sub-profile without a deployment reports `Plugin: NOT installed`, and because the config
> still lists the provider, external memory **fails silently with zero warnings**.
> Full mechanism: [`references/discovery-and-deployment-verification.md`](references/discovery-and-deployment-verification.md).

**Trade-offs among the three routes (hands-on review of 2026-09-14)**:

| Route | Verdict |
|------|------|
| in-tree (copy into hermes-agent/plugins/memory/ + git exclude) | ❌ Deprecated: `hermes update`'s stash/ZIP tree replacement can wipe whole directories, and it pollutes the upstream tree (the 0910/0911 patches were lost exactly this way) |
| Per-profile **symlink** → canonical source | ❌ Past incident: update's rsync backup/restore drops symlinks, and the loss is silent |
| Per-profile **physical copy** (rsync from the canonical source) | ✅ Current approach: no update path can touch it; cost = must redistribute after editing the canonical source |

```bash
# Distribute (re-run after editing the canonical source; also clean __pycache__)
for p in ~/.hermes/profiles/*/; do
  rsync -a ~/.hermes/plugins/memory-zvec/ "${p}plugins/memory-zvec/"
  rm -rf "${p}plugins/memory-zvec/__pycache__"
done
```

**Verify "discovery" per profile** (checking that files exist is not enough):

```bash
for prof in chip_expert financial_expert health_manager trading_bot zunhunfan; do
  HERMES_HOME="$HOME/.hermes/profiles/$prof" ~/.hermes/hermes-agent/venv/bin/python -c "
import sys; sys.path.insert(0, '/home/<user>/.hermes/hermes-agent')
from plugins.memory import find_provider_dir
import os; print('$prof ->', find_provider_dir('memory-zvec'))"
done
# For default, run once more with HERMES_HOME=~/.hermes; each should point to the physical directory under its own home
```

> Changes take effect after `systemctl --user restart hermes-gateway[-<profile>].service`
> (the dashboard also loads the provider; restart it together); roll one by one — each profile is down for only a few seconds.

---

## Step 3: Configure the Profile

Add to the target profile's `config.yaml`:

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: memory-zvec      # ← fork directory name (the same-named bundled copy is deleted; the old name resolves to None)
```

**Note**: only profiles that declare `provider:` will attempt to load the plugin.
Other profiles will not activate it even if a copy is deployed (and will not error out).

**⚠️ The config block key does not follow the directory name**: the plugin reads/writes the `plugins.memory-zvec:` section
(hard-coded for backward compatibility); config entries such as `zvec_dir` carry over unchanged, so switching the directory name requires zero migration.

---

## Step 4: Data Migration

### Option A: Migrate from LanceDB (Vector Reuse)

If the source data is in LanceDB, the vectors can be reused directly (same bge-m3:latest model, 1024 dims);
no re-embedding is needed.

**Key point**: LanceDB `.lance` directories are opened directly with `lance.dataset()` (not `lancedb.connect().open_table()`).

**Full migration script template** (see [`scripts/migrate-lancedb-to-zvec.py`](scripts/migrate-lancedb-to-zvec.py)):

```python
# Core pattern:
ds = lance.dataset(LANCEDB_PATH)           # open the .lance file directly
df = ds.to_table().to_pandas()              # read everything
coll = zvec.create_and_open(ZVEC_PATH, schema)  # path must not exist
for _, row in df.iterrows():
    doc = zvec.Doc(
        id=str(row['id']),
        vectors={"vector": ast.literal_eval(row['vector']) if isinstance(row['vector'], str) else row['vector']},
        fields={...},                       # scalar fields extracted from row
    )
coll.insert([doc1, ...])                    # batch insert
coll.flush()
coll.optimize()                             # build the HNSW index
```

**Caveats**:
- `zvec.Doc` construction: see the pitfall log §0
- `create_and_open` path must not exist: see the pitfall log §0c
- After inserting you must `flush()` + `optimize()` to ensure the HNSW index is complete
- There is no `close()` method; Python GC releases automatically

### Option B: Create Fresh from Other Sources

The plugin automatically creates the collection on first start (`create_and_open`).
If the directory already exists and has data, it simply `open`s it.

Default data directory location: `$HERMES_HOME/记忆数据库/zvec_memory/memories`

### Option C: Migrate from FTS5 Session History (LanceDB Path, Archived)

> **⚠️ Archived**: the following flow went `state.db` → LanceDB; it has since been changed to `state.db` → Zvec.
> Kept for reference. For the current migration script see Option A.

Historical script: [`scripts/migrate_sessions_to_lancedb.py`](scripts/migrate_sessions_to_lancedb.py)

This script uses a **Strip Pipeline** (5-stage content sanitization) to keep tool-call templates from repeatedly polluting
the vector space. See [`references/strip-pipeline-architecture.md`](references/strip-pipeline-architecture.md).

---

## Step 5: Index Repair (⚠️ Critical Step)

After migration the data is written into Zvec, but **the indexes must be repaired manually**. Below are the three known issues and their fixes:

### Issue 1: HNSW Vector Index Completeness at 0%

**Symptom**: `collection.stats` shows `index_completeness: {"vector": 0.0}`

**Cause**: `optimize()` was never run after the bulk write, so the vectors remain in the flat buffer
and all vector searches degrade to brute-force scans.

> **⚠️ There is a second root cause (easy to miss)**: `optimize()` was originally called only in `on_session_end`;
> **the real-time per-turn write path `sync_turn` never optimizes**, so in daily use completeness stays low for long
> periods (observed 0.71) even though `enable_hnsw_optimize: true` in the config looks "in effect".
> Fix: add amortized counting after the flush in `_insert()` and optimize once every 64 writes.
> See [`references/discovery-and-deployment-verification.md`](references/discovery-and-deployment-verification.md) §5.

**Fix**:
```python
import zvec
coll = zvec.open("/path/to/memories")
coll.optimize()
print("index_completeness:", coll.stats.index_completeness)
# should print {"vector": 1.0}
```

### Issue 2: FTS Index Completely Broken

**Symptom**: all FTS searches (Chinese or English) return 0 results.

**Causes**:
1. At migration time the FTS index files existed but the data was not written correctly
2. The FTS tokenizer used `standard` (ASCII only), so Chinese content could not be indexed at all

**Fix**: rebuild the FTS index with the `jieba` tokenizer:
```python
import zvec
coll = zvec.open("/path/to/memories")
try:
    coll.drop_index("content")
except Exception:
    pass
coll.create_index("content", zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"]))
# verify
results = coll.query(zvec.Query(field_name="content", fts=zvec.Fts(match_string="芯片验证")), topk=5)
print(f"FTS hits: {len(results)}")
```

**Also modify the plugin code** (`memory-zvec/__init__.py`) so new collections use `jieba` by default:
```python
# Original code (standard does not support Chinese!)
zvec.FieldSchema("content", zvec.DataType.STRING,
    index_param=zvec.FtsIndexParam(tokenizer_name="standard")),
# Change to
zvec.FieldSchema("content", zvec.DataType.STRING,
    index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"])),
```

### Issue 3: Scalar Fields Missing InvertIndex

**Symptom**: `index_param` for `role`, `session_id`, and `created_at` is None,
so scalar filters fall back to full-table scans.

**Fix**:
```python
import zvec
coll = zvec.open("/path/to/memories")
for field_name in ["role", "session_id", "created_at"]:
    try:
        coll.drop_index(field_name)
    except Exception:
        pass
    coll.create_index(field_name, zvec.InvertIndexParam(enable_range_optimization=True))
```

### One-Shot Repair Script

The full repair script is at [`references/post-migration-repair.py`](references/post-migration-repair.py).

---

## Step 6: Full Functional Verification Checklist

After migration completes, verify item by item through Hermes tools (these tools are registered by the plugin):

> **Quick verification script**: you can run [`scripts/verify-plugin-tools.py`](scripts/verify-plugin-tools.py) directly;
> it performs all 8 checks automatically (loads the plugin through the Hermes framework, no gateway shutdown needed):
> ```bash
> HERMES_PROFILE=zunhunfan ~/.hermes/hermes-agent/venv/bin/python3 \
>   ~/.hermes/skills/zvec-memory-migration/scripts/verify-plugin-tools.py
> ```
>
> **After any write-path change, additionally run these two regressions** (otherwise you will miss write-loss bugs
> that only surface at shutdown/exit):
> ```bash
> # 1) shutdown() race (pitfall #17)
> ~/.hermes/hermes-agent/venv/bin/python3 scripts/test_shutdown_race.py \
>     ~/.hermes/backups/<old-version-copy>/__init__.py.bak   # with A/B comparison
> # 2) process exit eating in-flight writes (pitfall #18)
> HERMES_MEMORY_ZVEC_EXIT_DRAIN_S=0 ~/.hermes/hermes-agent/venv/bin/python3 \
>     scripts/exit_loss_experiment.py ~/.hermes/hermes-agent/plugins/memory/memory-zvec/__init__.py drain_off
> ```

| # | Tool | What is verified | Expected result |
|---|------|---------|---------|
| 1 | `vec_memory_stats` | Total count, session count, embedding model | `backend: zvec`, not a NoneType error |
| 2 | `vec_memory_list` | List the latest memories | Returns results containing metadata |
| 3 | `vec_memory_add` | Write a test entry | `status: added`, returns an id |
| 4 | `vec_memory_search(mode=vector)` | Vector semantic search | Returns score + results |
| 5 | `vec_memory_search(mode=keyword)` | FTS keyword search (Chinese) | Chinese keywords hit |
| 6 | `vec_memory_search(mode=hybrid)` | Vector + FTS RRF fusion | Returns fused ranking results |
| 7 | `vec_memory_search(session_id=...)` | Filter by session | Returns only the specified session |
| 8 | `vec_memory_delete` | Delete the test entry | `deleted: 1` |

### ⚠️ Verify `_coll` Is Initialized (Guard Against Silent Failure)

An empty result from `vec_memory_search` does **not necessarily** mean the database has no data — when `self._coll is None` it can silently return `{"results":[],"count":0}` via the `except` fallback. Verify with `vec_memory_stats` first (a clear non-NoneType error or a normal count means `_coll` is valid).

If `vec_memory_stats` reports `'NoneType' object has no attribute 'stats'`, then `initialize()` failed to create `_coll`. Troubleshooting order:

1. Check the logs for `name '_open_read_only' is not defined` → fix pitfall #14
2. Check the logs for `still locked after retry` → on the old version (bundled ≤1.0.0) fix per pitfall #12 (`self.shutdown()` at the start of initialize); **v1.3.x does the opposite: deliberately leaves opened handles untouched + drain/backoff retry** — do not apply the old fix to the new version
3. Check whether the logs end with `ZvecMemoryProvider initialized` → if missing, `_coll` is still None
4. Look for `Memory provider 'memory-zvec' activated` in the logs — **this line does not mean initialization succeeded** (hermes-agent prints it even when initialization throws)

**Recommended verification script**: run [`scripts/verify-plugin-tools.py`](scripts/verify-plugin-tools.py) directly (loads the plugin through the Hermes framework, no gateway shutdown needed).

### Plugin-Level Verification (Recommended, No Gateway Shutdown Needed)

Load the plugin and call its tools directly through the Hermes framework, no gateway involvement:

```bash
HERMES_PROFILE=<profile> ~/.hermes/hermes-agent/venv/bin/python3 << 'EOF'
import os; os.environ['HERMES_HOME'] = os.path.expanduser(f'~/.hermes/profiles/<profile>')
import sys; sys.path.insert(0, os.path.expanduser('~/.hermes/hermes-agent/venv/lib/python3.11/site-packages'))
from plugins.memory import load_memory_provider
p = load_memory_provider('memory-zvec')
p.initialize(session_id='verify')
import json
def call(name, params):
    r = p.handle_tool_call(name, params)
    return json.loads(r) if isinstance(r, str) else r
print(call('vec_memory_stats', {}))
print(call('vec_memory_search', {'query':'测试','mode':'keyword','top_k':3}))
print(call('vec_memory_add', {'content':'验证条目，至少50字符。','role':'turn'}))
EOF
```

**Note**: `handle_tool_call` returns a JSON string, not a dict. The parameter name for `vec_memory_delete` is `memory_ids` (an array), not `id`.

---

## Step 7: Log Confirmation

After a successful migration, check the gateway logs to confirm the plugin loaded from the correct path:

```bash
grep "memory-zvec\|MemoryProvider\|memory.*activated" \
    ~/.hermes/profiles/<profile>/logs/agent.log
```

Expected log lines:
```
agent.memory_manager: Memory provider 'memory-zvec' registered (5 tools)
_hermes_user_memory.memory-zvec: ZvecMemoryProvider opened existing collection: .../memories
_hermes_user_memory.memory-zvec: ZvecMemoryProvider initialized — model=bge-m3:latest dim=1024 ...
run_agent: Memory provider 'memory-zvec' activated
```

Key indicators:
- The namespace is `_hermes_user_memory.memory-zvec` (meaning the user plugins path was used)
- The process holding the LOCK file should be that profile's gateway PID

---

## Health Checks (For Ops)

### Ollama Connectivity

In a WSL environment, Ollama usually runs on the Windows side:

```bash
# Quick TCP port check (recommended)
timeout 3 bash -c 'echo > /dev/tcp/localhost/11434 && echo "OPEN"' 2>&1

# Confirm the Windows-side process
tasklist.exe 2>/dev/null | grep -i ollama

# API check (may time out due to slow WSL forwarding)
curl -s --connect-timeout 2 http://localhost:11434/ | head -5
```

> **Pitfall**: under WSL, Ollama reaches the Windows-side process through localhost port forwarding.
> If the Windows Ollama service stops, embedding fails — existing data remains readable, but new writes error out.
> `curl` may time out due to slow WSL forwarding; prefer the TCP port check.

### Quick Ollama Check Script

```bash
bash ~/.hermes/skills/zvec-memory-migration/scripts/check_ollama.sh
```

### "Memory Counts Drop After a Certain Date" — Genuine Usage Decline vs Write Failure

**Three-step triage for a zero-write day** (check the usage side before calling it a fault):
```bash
# 1) Any user DMs that day (gateway.log)
grep -c "<日期>.*Inbound dm message" ~/.hermes/logs/gateway.log
# 2) Real cron runs that day (cron_now field on the Shutdown drain line, or job execution logs)
grep "<日期>" ~/.hermes/logs/gateway.log | grep -c "cron_now=[1-9]"
# 3) Any agent activity that day (agent.log)
```
If all three are zero → genuinely zero usage (normal); do not treat it as a write fault. Note that in count_in-style FTS approximate-count scripts, `created_at` is a **second-granularity** timestamp (do not multiply the filter value by 1000 — multiplied, everything returns 0, easily misread as a write stoppage).

**Lock-contention failures (`still locked after retry`) are a cross-profile common phenomenon**: every profile's log shows 1–7 occurrences per rotation period. Trigger scenario: after `/new`, the previous session's on_session_end fallback batch is still writing in the background (an updated idmap.0 mtime is the evidence), the new session's initialize opens immediately and hits the lock, GC+retry cannot wait out 0.6s for release → read-only is also rejected → `_coll=None`. **All vec_memory_* tools in that session stay paralyzed until the session ends, with no automatic retry** — the user must `/new` or have the gateway rebuild the agent to recover. Do the diagnostic fd scan only after the lock is released (a held lock is normally findable via find /proc; if it cannot be found, the holder has released it or the lock lives in an in-process object).
> **⚠️ Scope**: the above describes bundled v1.0.0 behavior (before 2026-09-14). memory-zvec v1.1+ fixed it at the root — initialize drains in-flight writes first and then retries with backoff, and v1.3.x shares handles across sessions, eliminating this race; if this warning still appears in new-version logs, it is a new problem.

When memory data clearly drops after a certain date, **check the date distribution before concluding anything**:

```bash
# Check the time range of the latest memories via vec_memory_list
vec_memory_list  # observe created_at of the latest entries
```

**If the drop matches**: a concurrent decline in gateway.log message counts → **genuine usage decline, not a bug**.

**Signatures of a genuine write fault**:
- `agent.log` contains `WARNING.*session_end batch store failed`
- `agent.log` contains `Connection refused` to Ollama
- Memory counts for old dates suddenly decrease (data corruption)

---

## Memory Loss Risk Matrix (Source-Code Audit)

The conclusions below come from a systematic source-code audit of `run_agent.py`, `gateway/run.py`, `memory_manager.py`, `turn_finalizer.py`,
and `conversation_compression.py`.

### ✅ No Risk (Confirmed Safe in Source Code)

| Scenario | Mechanism | Source location |
|------|------|----------|
| Normal per-turn conversation | `sync_turn` background write + `on_session_end` fallback | `run_agent.py:3127` / `turn_finalizer.py:441` |
| `/new` or `/reset` | `commit_memory_session(messages)` → `on_session_end` | `run_agent.py:3060` |
| Graceful gateway shutdown (SIGTERM/SIGINT) | `_stop_impl` → `_finalize_shutdown_agents` + idle cache `_cleanup_agent_resources` | `gateway/run.py:6738-6755` |
| Idle cache eviction (LRU cap / TTL) | **Leaves the memory provider untouched**; the session can be restored at any time | `gateway/run.py:13968` comment: "memory provider keeps running" |
| Context compression | `commit_memory_session(messages)` before compressing | `conversation_compression.py:525` |
| Occasional Ollama timeout | `sync_turn` wraps the whole call in `try/except` and drops that turn; `on_session_end` is the fallback | `__init__.py:575-576` |

### ⚠️ Risky but Acceptable

| Risk | Probability | Impact | Reason |
|------|------|------|------|
| **Interrupted conversations are not written** | Medium | Medium | With `interrupted=True`, `sync_turn` returns immediately. The design intent is correct (the user never saw the interrupted reply), but if the user actively interrupts a useful reply, that turn is lost. A later `/new` triggers `on_session_end`, which back-fills it |
| **Ollama down for an extended period** | Low | High | Both `sync_turn` and `on_session_end` need embedding. Ollama being down for an entire session = all writes fail, only exposed at `/new` or shutdown. But `state.db` (full conversation history) is unaffected and can be replayed afterwards |
| **Per-item skip on embedding failure in `on_session_end`** | Low | Low | A single `continue` skips the item; `sync_turn` already wrote the role="turn" version as the fallback. See pitfall §13 |

### 🔴 Scenarios with Real Loss Risk

| Risk | Probability | Impact | Trigger condition |
|------|------|------|----------|
| ~~**daemon write thread killed by process exit**~~ | ~~High~~ | ~~Medium~~ | **Fixed (pitfall #18)**: added a bounded atexit join. Before the fix, cron jobs hit this almost every time (external worker lifetime ~0.7s) |
| **SIGKILL / hard process kill** | Very low | High | OOM killer, `kill -9`, forced Docker destruction. Daemon threads die instantly; `on_session_end` is not called (atexit will not run either) |
| **sync_turn daemon thread dies before flush** | Very low | Low | Process killed after `_coll.insert()` but before `_coll.flush()`. The Zvec WAL mechanism may have persisted it, but this is unconfirmed |
| **Zvec collection corruption** | Very low | High | Disk full, filesystem errors, Zvec bugs. No automatic recovery path |

### Memory Safety Guarantees of Gateway Graceful Shutdown

Key sequence of `_stop_impl()` in `gateway/run.py` (confirmed by source audit):

```
① _drain_active_agents(timeout)       — wait for in-progress turns to finish (including finalize_turn → sync_turn)
② _finalize_shutdown_agents(active)   — for running agents, call _cleanup_agent_resources → on_session_end
③ iterate _agent_cache (idle agents)     — lines 6740-6755, specifically handles idle cached agents
   → _cleanup_agent_resources(agent)  → shutdown_memory_provider(messages) → on_session_end
④ disconnect all adapters
```

**Key point**: step ③ ensures the idle cached agents' memories are not lost either.
This is the core confirmation of this audit — not every eviction path is safe (LRU/idle sweep goes through
`release_clients` and does not call on_session_end),
but **graceful gateway shutdown** handles memory correctly for all agents (running + idle cached).

---

## Data Backup and Restore

### Always Check mtime Before Backup/Restore (⚠️ Critical)

**Never blindly `cp -a` over data files**; compare mtimes first:

```bash
for p in 记忆数据库/zvec_memory state.db; do
  bak=/path/to/backup/$p
  cur=/path/to/current/$p
  if [ -e "$bak" ] && [ -e "$cur" ]; then
    bak_t=$(stat -c%Y "$bak")
    cur_t=$(stat -c%Y "$cur")
    if [ "$cur_t" -gt "$bak_t" ]; then
      echo "  ⚠️  $p  current newer → DO NOT overwrite (data loss risk)"
    else
      echo "  ✅ $p  backup newer or same"
    fi
  fi
done
```

**Decision matrix**:

| Current mtime vs backup | Decision |
|---|---|
| Current is newer | **Never overwrite** (overwriting loses data) |
| Backup newer + current is larger | Restore from backup (possible accidental deletion/truncation)|
| Backup newer + current is smaller or equal | Restore from backup |

**state.db special case**: even if mtime says "backup is newer", the current state.db may contain
sessions created after the backup. The correct approach is to **leave state.db alone** and use the migration script
to incrementally sync from the current state.db (the script dedupes by session_id; it is idempotent and re-runnable).

**Actual restore order**:
1. List all data-type files (zvec_memory, state.db, etc.)
2. Run the mtime + size comparison on each
3. **Ask the user** which to overwrite and which to skip
4. First run `mkdir -p pre-restore-backup-$(date +%Y%m%d)/` and then cp

---

## Pitfall Log

### 0. Zvec Doc Object Format (⚠️ Most Common Migration-Script Error)
**Symptom**: `AttributeError: 'dict' object has no attribute 'id'`
**Cause**: Zvec `insert()` does not accept dicts; you must pass `zvec.Doc` objects.
**Correct form**:
```python
doc = zvec.Doc(
    id="uuid-string",
    vectors={"vector": vec_list},        # vector field name must match VectorSchema.name in the schema
    fields={"content": "...", "role": "...", ...},  # scalar fields
)
coll.insert([doc1, doc2, ...])
```
**Note**: `id` is not a `primary_key` parameter of `FieldSchema` — `FieldSchema` has no `primary_key` parameter; the ID is set via `zvec.Doc.id`.

### 0b. CollectionSchema Construction Essentials
- Scalar fields go in `fields=[]`; vector fields go in `vectors=[VectorSchema(...)]`
- `VectorSchema(name, data_type, dim, index_param=...)` — `dim` is a standalone parameter, not part of `FieldSchema`
- There is no such thing as a `primary_key=True` FieldSchema parameter

### 0c. create_and_open Path Requirement
**Symptom**: `ValueError: path validate failed: path[...] exists`
**Cause**: `zvec.create_and_open()` requires the target path to **not exist at all**; even an empty directory triggers the error.
**Fix**: ensure the target path's **parent directory** exists but the target path itself does not:
```python
os.makedirs(parent_dir, exist_ok=True)
assert not os.path.exists(target_path)  # mandatory!
coll = zvec.create_and_open(target_path, schema)
```

### 0d. Correct Ollama /api/embed Invocation
**Note**: the Ollama embedding API differs from the OpenAI format:
- The request parameter is `input` (not `prompt`): `{"model": "bge-m3:latest", "input": ["text"]}`
- The response field is `embeddings` (an array, not `embedding`): `resp.json()["embeddings"][0]`
- The `prompt` parameter is **silently ignored**, returning an empty array `{"embeddings": []}`, which causes `IndexError: list index out of range`
- This is the trap most likely to waste debugging time — no error is raised; Ollama returns 200 + an empty array
- The memory-zvec plugin uses `input` correctly internally, but hand-written verification scripts hit this constantly

### 1. LOCK Conflict
**Symptom**: `RuntimeError: Can't lock read-write collection`
**Cause**: the gateway process already holds the collection lock; a direct Python connection is refused
**Fix**: verify with Hermes tools (through the loaded plugin instance), or use Python after shutting down the gateway

### 2. create_and_open Errors on an Existing Directory
**Symptom**: `path validate failed: path[...] exists`
**Cause**: the plugin's `initialize()` tries `zvec.open()` first and falls back to `create_and_open()` on failure
**Fix**: if open fails (e.g., the LOCK is held), create_and_open fails too. Release the lock first.

### 3. Cross-Profile Writes Blocked
**Symptom**: `Cross-profile write blocked by soft guard`
**Cause**: when modifying `~/.hermes/plugins/` (owned by the root/default profile), profile-launched agents have write protection
**Fix**: use the `cross_profile=True` parameter, or operate directly under the root profile

### 4. Wrong FTS Tokenizer Breaks Chinese
The `standard` tokenizer is an ASCII whitespace tokenizer and is completely ineffective for Chinese.
You must use `jieba` (built into Zvec, no extra installation needed).

### 5. handle_tool_call Returns a JSON String
**Symptom**: `AttributeError: 'str' object has no attribute 'get'`
**Cause**: `ZvecMemoryProvider.handle_tool_call()` uniformly returns a JSON string, not a dict.
**Fix**: parse with `json.loads(result)` after calling. Note that `vec_memory_add` returns `{"status": "skipped", "reason": "content too short"}` for entries whose content is too short (<50 chars).

### 6. vec_memory_delete's Parameter Name Is memory_ids (an Array)
**Symptom**: `{"error": "memory_ids is required"}` or `Delete(): incompatible function arguments`
**Cause**: `vec_memory_delete` takes `memory_ids: [str]` (an array of strings), not `id: str`.
**Correct call**: `{"memory_ids": ["uuid1", "uuid2"]}`

### 7. Switching Providers Requires Updating the plugins: Section Too (⚠️ Easy to Miss)
**Symptom**: after switching `memory.provider`, the plugin still reads the old config paths, or reports `provider not found`.
**Cause**: Hermes's `memory.provider` field decides which plugin to load, but the plugin's own configuration (Ollama URL,
data directory, etc.) is read from the `plugins.<provider_name>` section. If the plugins section's key name and paths are not updated
in sync, the plugin falls back to hard-coded defaults.
**Fields that must be updated**:
```yaml
# Old config
plugins:
  memory-lancedb:           # ← key name must change
    lance_dir: $HERMES_HOME/记忆数据库/lance_memory  # ← field name must change

# New config
plugins:
  memory-zvec:             # ← key name
    zvec_dir: $HERMES_HOME/记忆数据库/zvec_memory    # ← field name
```
**Verification command**:
```bash
grep -A10 'plugins:' ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml | grep -i 'lancedb'
# Expect: no output (if there is still output, leftovers remain uncleaned)
```

### 8. The default Profile's HERMES_HOME Path Is Special (⚠️ Migration Scripts)
**Symptom**: running the migration script with `HERMES_PROFILE=default` raises `FileNotFoundError` or path errors.
**Cause**: the default profile (profile-less) has `HERMES_HOME` = `~/.hermes/`,
while sub-profiles use `~/.hermes/profiles/<name>/`. If the migration script hard-codes `profiles/<name>`,
default resolves to the nonexistent `~/.hermes/profiles/default/`.
**Correct logic**:
```python
PROFILE = os.environ.get("HERMES_PROFILE", "zunhunfan")
if PROFILE == "default":
    HERMES_HOME = Path.home() / ".hermes"
else:
    HERMES_HOME = Path.home() / ".hermes/profiles" / PROFILE
```
**Likewise**: when the plugin-level verification script sets `os.environ['HERMES_HOME']`, the default profile should get
`~/.hermes/` (not `~/.hermes/profiles/default/`).

### 9. Root config.yaml Is Write-Protected; Patch Tools Refuse
**Symptom**: `Refusing to write to Hermes config file: ~/.hermes/config.yaml`
**Cause**: Hermes's write_file/patch tools have safety protection for the root config.yaml.
**Fix**: use the `terminal` tool to run `sed -i`, or the `hermes config set` CLI.

### 10. on_session_end Embedding Failures Skipped Silently (⚠️ Fixed)

**Problem**: during `on_session_end` batch writes, a failed `_ollama_embed_single` hit `continue` directly —
no retry, no logging. A brief Ollama hiccup silently lost that memory.

**Fix** (2026-06-27): introduced `_embed_with_retry()` — after the first failure, wait 1 second and retry once.
Only skip if both attempts fail, and log `skipped=N`.

### 11. on_session_end Batch Write Flushes Only at the End (⚠️ Fixed)

**Problem**: the original implementation accumulated all documents with `insert(docs)` and called `flush()` once at the end.
If the process was killed before the flush (SIGKILL/OOM), the entire batch was lost.

**Fix** (2026-06-27): changed to per-item `insert + flush` so every write persists immediately.
HNSW `optimize()` still runs once at the end (it does not affect persistence).

### 12. Gateway Multi-Session Shared Collection Causes open Failure (⚠️ Major Bug)

**Symptom**: `vec_memory_stats` reports `'NoneType' object has no attribute 'stats'`,
and `agent.log` shows `ZvecMemoryProvider failed to create collection:

**Root-cause chain**:
1. Session A's agent instance calls `zvec.open()` and takes the rw lock (LOCK file)
2. Session A's agent is shut down, calling the old `shutdown()` → only `self._coll = None`; **the Zvec LOCK is not released**
3. Session B's `initialize()` → `zvec.open()` is blocked by the old Zvec Collection object that GC has not reclaimed
4. Fallback → `create_and_open()` fails because the path exists → `self._coll = None` → all tools crash

**Key findings** ([experimental verification + lock-governance evolution](references/zvec-lock-mechanism.md)):
- Zvec Collection has no `close()`/`release()` methods; locks are released via Python GC (`del coll + gc.collect()`)
- **Two** `zvec.open()` calls in the same process always conflict on the lock (even for different agent instances within the same gateway)
- After `del coll` + `gc.collect()` + `sleep(0.3)`, the lock releases immediately and the third `open()` succeeds
- Cross-process lock conflicts (e.g., a manual Python script holding it) cannot be resolved via GC; a read-only fallback is needed

> **⚠️ This section and the v3.x fix history below are historical records** — the current approach is the user-level fork
> `memory-zvec` v1.3.1 (drain + backoff, idle 30s release + lazy reopen, handle sharing across sessions,
> sid rebinding on session switch). For the full design and the real-incident timeline of 2026-09-14/15 see
> [`references/zvec-lock-mechanism.md`](references/zvec-lock-mechanism.md)
> (latter half). Read that first when troubleshooting; do not follow this section's old shutdown approach.

**Fix evolution**:

### v3.1 (First Fix) — shutdown + Four-Layer Fallback

**`shutdown()`**: release the lock proactively
```python
def shutdown(self) -> None:
    if self._coll is not None:
        try:
            del self._coll
            import gc; gc.collect()
        except Exception:
            pass
```

**`initialize()`**: four-layer fallback

### v3.3 (Supplementary Fix, 2026-07-02) — Three Critical Omissions

**Fix 1: add `self.shutdown()` at the top of `initialize()`** (line 394)
v3.1 only invoked the `_open_read_only` read-only fallback on exception paths, but **when `initialize()` ran multiple times in the same process, the old lock was never released**. Even after the first session's `initialize()` succeeded, no explicit `shutdown()` path in between released the old connection's LOCK. Solution: call `self.shutdown()` at the very top of the `initialize()` method body (before `from hermes_constants`) so every initialization first releases the previous round's lock.

```python
def initialize(self, session_id: str, **kwargs) -> None:
    """Connect to Zvec collection, ensure schema, warm up Ollama."""
    # Release any lock left by a prior session in this process
    self.shutdown()                              # ← added in v3.3
    from hermes_constants import get_hermes_home
```

**Fix 2: add an `if not self._coll:` guard to every tool handler** (lines 1053, 1171, 1234, 1250)
If `initialize()` fails to create a valid `self._coll` (by any path), all tools crash outright with `'NoneType' object has no attribute '...'`. The more graceful approach is to check `self._coll` at **each tool handler's entry point** and return a clear error JSON:

```python
def _tool_add(self, args: dict) -> str:
    if not self._coll:
        return json.dumps({"status": "skipped", "reason": "database not yet initialized"})
    if self._read_only:
        ...
```

The 4 handlers needing the guard: `_tool_add`, `_tool_list`, `_tool_stats`, `_tool_delete`. `_tool_search` already has a downstream `except` fallback, but it **silently returns an empty results array** — users mistake this for normal search results when the database is not connected at all. Add the guard there too.

**Fix 3: bare-name `_open_read_only` → `self._open_read_only`** (see pitfall #14)
If pitfall #14 is already fixed, this is automatically covered. If troubleshooting separately: the `_open_read_only(...)` calls on lines 438 and 446 must be changed to `self._open_read_only(...)`.

### Complete Fallback Chain (v3.3)
```
① zvec.open(rw)           → if successful, use directly
② gc.collect() + retry    → the old session's shutdown released the lock; retry acquiring rw ✅
③ zvec.open(read_only)    → **in zvec 0.5.1, read-only also needs an exclusive LOCK**
                           try opening read-only (via `self._open_read_only()`)
                           → if `_open_read_only` is a bare name it raises NameError and
                              initialize fails entirely (see pitfall #14)
                           → even with the NameError fixed, zvec 0.5.1's read-only open still
                              reports `Can't lock read-only collection` when the LOCK is held
④ zvec.create_and_open    → path does not exist; create new
```

**⚠️ In zvec 0.5.1, read-only also needs an exclusive lock**: experiments confirmed that `zvec.open(path, option=CollectionOption(read_only=True))` raises `RuntimeError: Can't lock read-only collection: .../LOCK` when the LOCK is held. This means step ③ **does not work** on zvec 0.5.1 and `self._coll` ends up None. You must rely on step ② (GC + retry) to acquire the lock successfully.

> **0.6.0 hands-on addendum (wording refined)**: the read-only lock is **shared** (multiple read-only holders can coexist), but read-only and write locks
> are **mutually exclusive** — while an active writing process (rebuild/migration/gateway session) exists, read-only reports the same
> `Can't lock read-only collection`. So first confirm there is no writing process when troubleshooting, or you will misjudge it as database corruption.

**NameError muddies the diagnosis**: pitfall #14's `_open_read_only` NameError shows up in logs as `initialize failed: name '_open_read_only' is not defined`, masking the underlying LOCK problem. Recommended troubleshooting order: ① fix the NameError first → ② then handle the LOCK.

**Blast radius**: triggered when switching sessions within the same gateway process.
Most common on the default profile (multi-turn conversations); sub-profile gateways usually have only one active session.

### 11. After Upgrading the lancedb Library, Old Tables Report "Table not found" (⚠️ Legacy LanceDB)
**Symptom**: `.lance` files written by old lancedb (<0.10) lack the `_versions/` directory;
the new version (0.30.x) refuses to open them.
**Fix**:
```bash
~/.hermes/venv/bin/python ~/.hermes/skills/zvec-memory-migration/scripts/lancedb_rebuild_table.py \
    --profile <prof>
```
See [`references/lancedb-table-rebuild.md`](references/lancedb-table-rebuild.md) for details.

### 12. sync_turn Uses a Daemon Thread; SIGKILL May Lose Writes (⚠️ Unfixable)
**Symptom**: after the gateway is SIGKILLed, the most recent turns are absent from Zvec.
**Cause**: `sync_turn` runs in `threading.Thread(daemon=True)`.
Daemon threads die instantly when the main process is SIGKILLed, with **no chance to flush**.
Graceful shutdowns triggered by SIGTERM/SIGINT have a drain mechanism (waiting for active agents to finish) and do not trigger this.
**Mitigations**:
- `state.db` is unaffected (SQLite WAL mode is more durable) and can be replayed afterwards
- `on_session_end` already serves as the fallback in the normal shutdown path
- Unless OOM killer or `kill -9` happens frequently, this is not a practical risk
### 13. on_session_end Write Failures — Two Distinct Root Causes

`on_session_end` catches `Exception` wholesale in the `_batch_store` thread, producing the uniform `WARNING session_end batch store failed: XXX` log line. Under that identical log prefix there are two completely different root causes:

#### Type A: Per-Item Embedding Skip

**Symptom**: log `WARNING _hermes_user_memory.memory-zvec: session_end batch store failed: ...` with embedding-related errors (Ollama timeouts, HTTP errors, etc.).

**Root cause**: in `on_session_end`'s batch write loop, a failed `_ollama_embed_single()` hits `continue` —
no retry, no logging, no alerting (`__init__.py` lines 643-646, `except Exception: continue`).
**Actual impact**: low. Because `sync_turn` already writes the role="turn" version in real time each turn,
`on_session_end`'s role="session_end" version is the fallback — even if a few are skipped, memory completeness is unaffected.
**But if sync_turn also failed** (Ollama down during that turn), that turn's conversation is lost entirely.
**Improvement suggestion**: add a retry (at least once) in `on_session_end`'s `except`, or log a warning.

#### Type B: `_coll` Not Initialized (AttributeError)

**Symptom**:
```
WARNING _hermes_user_memory.memory-zvec: session_end batch store failed:
  'ZvecMemoryProvider' object has no attribute '_coll'
```
while every `vec_memory_stats` call reports `'NoneType' object has no attribute 'stats'`.

**Root cause**: `__init__()` never initialized `self._coll = None` (a bug in the old plugin code). When `on_session_end` reached `self._coll.insert(doc)` inside `_batch_store`, the `_coll` attribute did not exist at all, triggering AttributeError.

**v3.3 fix**: `__init__` line 308 now includes `self._coll = None`, and `on_session_end` line 585 gained the `if not self._coll: return` entry guard.

**Diagnostic commands**:
```bash
# Check whether the error still occurs (an old gateway may still run old code)
grep "has no attribute '_coll'" ~/.hermes/logs/agent.log ~/.hermes/profiles/*/logs/agent.log | tail -5
# Confirm the plugin version
grep "self._coll = None" ~/.hermes/plugins/memory-zvec/__init__.py
```

**Fix**: if this error still appears in the logs, the gateway is running the old plugin code. Run:
```bash
sudo systemctl restart hermes-gateway-<profile>
```

See Layer 4 runtime diagnostics in [`references/multi-profile-health-check.md`](references/multi-profile-health-check.md).

### 14. `_open_read_only` NameError Leaves `_coll` as None (‼️ Most Common Cause)

**Symptom**: `vec_memory_add`/`list`/`stats` all fail; the logs show:
```
WARNING agent.memory_manager: Memory provider 'memory-zvec' initialize failed:
  name '_open_read_only' is not defined
```
and every subsequent tool call reports:
```
"Stats failed: 'NoneType' object has no attribute 'stats'"
"Failed to add memory: 'NoneType' object has no attribute 'insert'"
"List failed: 'NoneType' object has no attribute 'query'"
```
But `vec_memory_search` may return an empty results array (not an error) — because `_tool_search`'s `except` fallback returns `{"results":[],"count":0}`; in reality the database is not connected.

**Root cause**: `memory-zvec/__init__.py` lines 438 and 446 call `_open_read_only(...)` by **bare name**, but that method is a `@staticmethod` (line 383). In Python, a `@staticmethod` defined inside a class body cannot be accessed by bare name from other methods — it must go through `self._open_read_only()` or `ZvecMemoryProvider._open_read_only()`.

```python
# Lines 383-384: definition
@staticmethod
def _open_read_only(path, option):

# Line 438: call site (wrong! bare name → NameError)
self._coll = _open_read_only(collection_path, _ro_option)
              ↑ NameError: name '_open_read_only' is not defined
```

**Fix**: change `_open_read_only(...)` to `self._open_read_only(...)` in both places:
```python
# change both locations
self._coll = self._open_read_only(collection_path, _ro_option)  # line 438
self._coll = self._open_read_only(collection_path, _ro_option)  # line 446
```

**Key differences from the LOCK conflict (pitfall #12)**:

| Trait | LOCK conflict (#12) | NameError null reference (#14) |
|------|-----------------|----------------------|
| Search | ❌ All fail | ✅ Silently returns an empty array (database not connected) |
| Writes | ❌ Fail | ❌ Fail |
| Log clues | Contains `Can't lock` and `still locked` | Contains `name '_open_read_only' is not defined` |
| initialize log | `still locked after retry, falling back to read-only` | `initialize failed: name '_open_read_only' is not defined` |
| Root cause | Zvec lock contention | Python scoping error — a bug in the plugin code |

**Why the first log line is a False Negative**: hermes-agent's `run_agent.py` still prints `Memory provider '...' activated` (`run_agent: Memory provider 'memory-zvec' activated`) even after `memory_manager.initialize()` throws. Seeing that line therefore does not mean initialization succeeded — you must also check whether `initialize failed` appears earlier.

**Relation to pitfall #12**: even with the NameError fixed, the read-only fallback still fails on zvec 0.5.1 (see the pitfall #12 update), because zvec 0.5.1's `open(read_only=True)` also needs an exclusive LOCK.

**Full diagnostic commands**:
```bash
# 1. Confirm the zvec version
python3 -c "import zvec; print(zvec.__version__)"
# 2. Check the logs for the NameError
grep "initialize failed" ~/.hermes/logs/agent.log ~/.hermes/profiles/*/logs/agent.log
# 3. Confirm whether an old connection in the same process holds the LOCK (locks are independent per profile)
ls -la ~/.hermes/profiles/*/记忆数据库/zvec_memory/memories/LOCK 2>/dev/null
# 4. Inspect the plugin code directly
grep -n "_open_read_only" ~/.hermes/plugins/memory-zvec/__init__.py
```

**Verify the fix**: after the change, restart the agent session in the same process; the logs should show:
```
_hermes_user_memory.memory-zvec: ZvecMemoryProvider opened existing collection: .../memories
_hermes_user_memory.memory-zvec: ZvecMemoryProvider initialized — model=bge-m3:latest dim=1024
```
Then `vec_memory_stats` should return normal counts and `vec_memory_add` should return `{"status": "added", "id": "..."}`.

### 15. Idle Cache Eviction (LRU Cap / TTL Sweep) Does Not Trigger on_session_end (⚠️ Normal Behavior, Not a Bug)
**Symptom**: after LRU eviction or an idle sweep, the evicted session's `on_session_end` is not called.
**Cause**: eviction goes through `_release_evicted_agent_soft()` → `release_clients()`, whose comment says plainly
"memory provider (has its own lifecycle; keeps running)". **`shutdown_memory_provider()` is not called**.
**Why this is safe**: the evicted agent's ZvecMemoryProvider instance is still in memory (Python GC has not reclaimed it),
holding the rw lock; `on_session_end` can still be triggered later by other paths (e.g., graceful gateway shutdown step ③).
If the gateway **shuts down normally** after eviction, idle agents are cleaned up correctly by `_stop_impl` lines 6740-6755.
**The only risk**: if the gateway is SIGKILLed after eviction but before normal shutdown, those idle agents' on_session_end never runs.

**Observed consequence chain (confirmed on trading_bot, 2026-09-14) — post-eviction lock leak → the next new session gets _coll=None**:
The evicted agent's provider instance is strongly referenced by gateway internal structures, so GC never reclaims it → the rw lock stays held. The next new session's initialize:
`lock detected → GC+retry fails → read-only also rejected (read/write mutual exclusion) → _coll=None`, yet the logs **still print**
`Memory provider 'memory-zvec' activated` (false positive). From then on every vec_memory_* reports `database not yet initialized`,
and this session's memory reads/writes are silently broken (sync_turn fails too; the conversation goes only into state.db).
Three-step diagnosis (read-only throughout): ① use `readlink` on `/proc/*/fd` for an **exact full-path comparison** of the LOCK (do not grep substrings — identical names across profiles guarantee false matches); ② holder = this profile's own gateway → same-process lock leak; a cross-profile mixup shows another PID; ③ grep the logs for `still locked after retry` + `failed to open (read-only)` + `activated` — all three appearing together confirms the diagnosis.
Fix: restart that profile's gateway (process exit releases the lock; **this disconnects its Feishu sessions** and needs informed user confirmation).
Side checks (both ruled out here): whether the configured embedding_model tag can actually embed, and whether HERMES_HOME points at this profile.

### 16. Ollama Embedding Model-Name Drift Silently Stops Automatic Memory Writes (‼️ Discovered 2026-09-07)
**Symptom**: when the configured `plugins.memory-zvec.embedding_model` does not match the actual Ollama model tag, all `sync_turn`/`on_session_end` embedding calls fail **without any prominent error** (the skill log has `session_end batch store failed`, but grep for "not found" in gateway.log counts 0 — very easy to miss). It shows up as turn/session_end entries in Zvec stopping after a certain date, while the collection itself is healthy (stats normal, manual writes normal).
**Real case** (financial_expert): on 8/22 the Ollama-side model tag changed from `bge-m3:567m` to `bge-m3:latest` (`ollama list` showed only `bge-m3:latest`) while config.yaml still said `bge-m3:567m` → `/api/embed` returned `{"error": "model not found"}` → after 8/22, automatic turn/session_end entries dropped to 0 (previously 195 in July and 32 in early August), undetected for 16 days.
**Diagnosis** (three steps, 2 minutes):
```bash
# 1. Compare config with reality
grep embedding_model ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml
curl -s http://localhost:11434/api/tags | python3 -m json.tool | grep '"name"'
# 2. Test embed directly (try both tags)
curl -s http://localhost:11434/api/embed -d '{"model":"bge-m3:567m","input":["t"]}' | head -c 120
# 3. Count Zvec entries per month to find the cut-off point (the month where turn/session_end hits zero is where the stoppage began)
```
**Fix**: change config.yaml's `embedding_model` to the actual Ollama tag (e.g., `bge-m3:latest`), or restore the old tag with `ollama pull bge-m3:567m`, then restart that profile's gateway. If the dimension is unchanged (1024), existing vectors need no migration.
**Prevention**: tags change when Ollama pulls new models or removes old ones, and the memory backend has zero tolerance for model-name drift; health-check scripts should add a "can the configured tag actually embed" check item.

**⚠️ When fixing the tag, fix every profile at once**: this problem recurred once — the first time only the affected
profile was fixed, and the other four kept running with the broken tag for half a month unnoticed. Command:
```bash
# Per profile, verify "the configured tag can actually embed"
for f in ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml; do
  tag=$(grep -m1 'embedding_model' "$f" | awk '{print $2}')
  printf '%-52s %-16s ' "$f" "$tag"
  curl -s http://localhost:11434/api/embed -d "{\"model\":\"$tag\",\"input\":[\"t\"]}" \
    | grep -q '"embeddings"' && echo '✅' || echo '❌ 该 tag 无法 embed'
done
```

**One level deeper**: this problem's dead-search signal is swallowed by `logger.debug` inside `is_available()` (no WARNING is printed when it returns False). Deployment verification must run both layers — discovery + `hermes memory status` — and cannot rely on counting how many entries are in the collection (see the references in the same directory).

### 17. shutdown() Deletes the `_coll` Attribute → the Whole on_session_end Background Batch Is Lost (‼️ Real Bug, Fixed)

**Symptom**: at session close, a single occurrence of
```
WARNING plugins.memory.memory-zvec: session_end batch store failed:
  'ZvecMemoryProvider' object has no attribute '_coll'
```
after which reads and writes to the collection are normal again → very easy to dismiss as noise. The real consequence: that session's **session_end fallback batch wrote nothing at all**
(the `role="turn"` records had already landed, so if content was not lost it is nearly invisible to the eye; this is "the fallback defense line failing silently").

**Root cause**: `shutdown()` released the Zvec LOCK by writing `del self._coll` (**deleting the instance attribute**),
while `on_session_end`'s batch write runs in `threading.Thread(daemon=True)`. At agent close,
`on_session_end` (spawning the thread) and `shutdown()` (deleting the attribute) race → the background thread reading `self._coll` raises AttributeError,
which the thread's `except Exception` swallows into a single WARNING.

**Fix (both parts are mandatory)**:
1. `shutdown()` must not delete the attribute:
   ```python
   coll = self._coll
   self._coll = None          # keep the attribute so all `if not self._coll` guards stay valid
   if coll is not None:
       del coll; import gc; gc.collect()   # the lock is released as usual
   ```
2. The background write path must **carry its own strong reference**: `sync_turn` / `on_session_end` capture `coll = self._coll` before spawning the thread
   and use `coll` inside it (`_insert(..., coll=coll)`). Otherwise, after step 1 sets the attribute to None, the same race becomes
   `'NoneType' object has no attribute 'insert'` — a different failure mode, batch lost all the same.

**Regression test**: [`scripts/test_shutdown_race.py`](scripts/test_shutdown_race.py) (A/B comparison of old/new implementations:
old version 4 pending writes → 0 landed + 1 warning; new version 4/4 + 0 warnings). Mandatory after any change to this plugin.

**Why functional acceptance testing cannot catch it**: the 8/11-item style "write → three kinds of retrieval → delete" cases are serial and never hit the
concurrency window at agent close; and the failure is swallowed into a WARNING. When troubleshooting memory writes, **always grep for this WARNING first**:
```bash
grep -a "session_end batch store failed" ~/.hermes/logs/agent.log* ~/.hermes/profiles/*/logs/agent.log*
```
`has no attribute '_coll'` appearing = this pitfall (write loss); embed/HTTP errors appearing = pitfall #13 Type A.

### 18. Process Exit Eats In-Flight Writes → Cron Jobs Silently Lose Memories (‼️ Real Bug, Fixed — 2026-09-11)

**Symptom**: cron jobs (AI daily briefs / retrospectives / news collection) finish, but the memory store has no records for those sessions, **with zero errors in the logs**.
Typical shape: a profile's cron sessions have `role='session_end'` records that **never existed**, while `role='turn'` is hit-or-miss.

**Root cause (double async + process exit)**:
- Core side: `sync_all()` hands writes to a `DaemonThreadPoolExecutor` (daemon threads).
- Plugin side: `sync_turn()` / `on_session_end()` **each spawn yet another daemon thread**.
- Each cron agent task is an **independent external worker process** (`hermes-worker-cron-<jobid>-exec-*.scope`):
  `_finalize_cron_session` → `agent.close()` → `on_session_end(messages)` → the interpreter exits immediately (measured total lifetime ~0.7s).
- Interpreter teardown **kills daemon threads outright**, so the write currently doing its Ollama embedding and the whole session_end batch vanish into thin air.
- Hermes's `shutdown_all()` can only drain its own executor and **cannot drain threads the plugin spawned itself** — so this path must be plugged by the plugin.

**Reproduction experiment** (mandatory after plugin changes): [`scripts/exit_loss_experiment.py`](scripts/exit_loss_experiment.py)
```bash
# the child process exits immediately after sync_turn/on_session_end; the parent counts landed rows independently
HERMES_MEMORY_ZVEC_EXIT_DRAIN_S=0 ~/.hermes/hermes-agent/venv/bin/python3 \
    scripts/exit_loss_experiment.py ~/.hermes/hermes-agent/plugins/memory/memory-zvec/__init__.py drain_off
# measured: drain off → sync_turn 0 rows / session_end 0 rows; drain on → 1 row / 4 rows
```

**Fix**: the plugin registers an atexit fallback — register in-flight write threads and perform a **bounded join**
before interpreter exit (`atexit` runs before daemon threads are reaped; same precedent as the openviking plugin):
```python
_EXIT_DRAIN_TIMEOUT_S = float(os.environ.get("HERMES_MEMORY_ZVEC_EXIT_DRAIN_S", "30"))
# sync_turn / on_session_end: call _track_write_thread(t) before spawning; _untrack_write_thread in the thread's finally
atexit.register(_drain_pending_writes)
```
`HERMES_MEMORY_ZVEC_EXIT_DRAIN_S=0` disables it (for A/B testing or when an instant exit is required).

**Production verification (2026-09-11)**: a temporary cron job (a true external worker process) ran the full chain and
`role='turn'` and `role='session_end'` **each landed exactly 1 record**; before the fix, that profile's cron sessions had 0 session_end records.

**Reading the results (do not mistake normal for broken)**: a cron session = **a single turn** (1 user + N assistant/tool messages),
so each cron job's expected yield is **1 turn record + 1 session_end record**, not "one per message".
When judging write health by "latest record time", you must distinguish gateway sessions (multi-turn) from cron jobs (single-turn).

### 19. A Live Idle-Cached Agent Holds the Lock → GC+Retry Fails, New Session Stuck at _coll=None (‼️ Verified on health_manager, 2026-09-14)

**Symptom**: within the same gateway process, session A's agent successfully `zvec.open()`s, takes the rw lock, and enters the **idle cache (still strongly referenced, not GC'd)**;
session B's agent `initialize()` runs `self.shutdown()` (clearing only its own `_coll`) + `gc.collect()` + retry,
but **the lock still is not released** → the read-only fallback also fails due to write-lock mutual exclusion → B's `_coll=None`, all `vec_memory_*` in this session
report "database not yet initialized", and automatic memory writes stop for the whole session. The log shows only one `still locked after retry`.

**Difference from pitfall #12**: in #12, GC+retry can succeed because the old instance no longer has strong references (it can be reclaimed);
in this pitfall the old instance is **actively held by the gateway's idle agent cache** (see #15 "memory provider keeps running"),
so GC can never reclaim it and retry is guaranteed to fail. There is no in-process self-recovery path; only **restarting that profile's gateway** lets the next session take the lock.

**Reading it**: `vec_memory_stats` returns `database not yet initialized` + the log has `still locked after retry`
+ the data directory still shows an mtime from other sessions writing today = this session's instance collided with a live instance's lock — not database corruption, not an Ollama problem.

**Troubleshooting notes (do not misjudge in health checks)**:
- To assess data health, bypass the tool layer and read `*/scalar.0.ipc` directly (pyarrow; `uv run --with pyarrow` when the venv lacks pyarrow);
  do not declare the database empty just because this session's stats say not initialized.
- To determine a "write stoppage", you must cross-check against state.db's `sessions/messages` (timestamps are **unix epoch floats**; use
  `datetime(started_at,'unixepoch','localtime')`, not TEXT dates): session activity present but zero Zvec writes is the actual fault signature.
- If agent.log does not even contain `Memory provider 'memory-zvec' registered`, the gateway process never loaded the plugin at all
  (common after a user-tree→bundled migration when a long-running gateway was not restarted); this must be distinguished from embedding tag drift
  (#16: registered present but no writes).
- The standalone verification script (verify-plugin-tools.py, session_id=`verify_*`) being able to write only proves "a fresh process + plugin + database are fine";
  it does not prove the running gateway has loaded it; also, verify entries are junk data and must be deleted after verification.

---

## File Location Overview

| Item | Location | Notes |
|------|------|------|
| memory-zvec canonical source | `~/.hermes/plugins/memory-zvec/` | User-level fork v1.3.x (the only editable copy); the directory name carries -x while the yaml internal name remains memory-zvec |
| Profile deployment copies | `~/.hermes/profiles/*/plugins/memory-zvec/` | Physical copies (rsync-distributed, not symlinks) |
| Bundled old-version archive | `~/.hermes/backups/memory-zvec-shutdown-race-20260911/bundled_memory-zvec_removed_20260914/` | Archive from when it was removed from the system tree on 2026-09-14 |
| zvec skill | `~/.hermes/skills/zvec/` | Zvec API reference manual |
| This playbook | `~/.hermes/skills/zvec-memory-migration/` | Globally shared |
| Profile data directory | `~/.hermes/profiles/*/记忆数据库/zvec_memory/` | Independent per profile |

---

## Business Data Migration

Financial news, telegraph, and analysis data migration from LanceDB → Zvec.
See [`references/financial-data-migration.md`](references/financial-data-migration.md) for
call graph, schema designs, function mapping, and phased execution plan.

> **Merge record (2026-07-11)**: `hardware/vector-db-migration` was merged into this skill. Merged content:
> - `references/migration-audit-checklist.md` (the generic 3-dimension version) → `references/migration-audit-checklist-3dimension.md`
> - Other files (`zvec-api-reference.md`, `migrate-lancedb-to-zvec.py`) already existed in this skill (version updates; the newer versions were kept)

## References

### Zvec / memory-zvec

- [`references/discovery-and-deployment-verification.md`](references/discovery-and-deployment-verification.md) — **iron rules for cross-profile deployment and deployment verification** (discovery mechanism, embedding tag drift, 0.6.0 differences, amortized optimize, mirror-tree shadowing, absolute backup checks, rebuild from state.db)
- [`references/memory-zvec-readme.md`](references/memory-zvec-readme.md) — plugin feature description
- [`references/zvec-api-reference.md`](references/zvec-api-reference.md) — Zvec Python API quick reference
- [`references/migration-audit-checklist.md`](references/migration-audit-checklist.md) — migration verification checklist
- [`references/post-migration-repair.py`](references/post-migration-repair.py) — one-shot index repair script

### Lock & Concurrency

- [`references/zvec-lock-mechanism.md`](references/zvec-lock-mechanism.md) — Zvec lock-semantics experiments + the memory-zvec fork's lock-governance evolution (v1.1–v1.3.1: drain/backoff, idle release, handle sharing, real-incident timeline, ops quick reference)
- [`references/multi-profile-health-check.md`](references/multi-profile-health-check.md) — multi-profile 4-layer memory health-check methodology (config/plugin/data/runtime)

### Scripts

- [`scripts/migrate-lancedb-to-zvec.py`](scripts/migrate-lancedb-to-zvec.py) — full LanceDB→Zvec migration script
- [`scripts/verify-plugin-tools.py`](scripts/verify-plugin-tools.py) — 8-point full functional verification script
- [`scripts/check_ollama.sh`](scripts/check_ollama.sh) — Ollama health check
- [`scripts/check_all_profiles.py`](scripts/check_all_profiles.py) — multi-profile memory health check (HNSW/FTS/LOCK)

### Legacy LanceDB (Archived Reference)

- [`scripts/migrate_sessions_to_lancedb.py`](scripts/migrate_sessions_to_lancedb.py) — FTS5→LanceDB migration (historical)
- [`scripts/lancedb_rebuild_table.py`](scripts/lancedb_rebuild_table.py) — LanceDB table rebuild script
- [`references/vector-memory-architecture.md`](references/vector-memory-architecture.md) — overall memory system architecture
- [`references/multi-profile-business-lancedb.md`](references/multi-profile-business-lancedb.md) — multi-profile business LanceDB
- [`references/actual-storage-format.md`](references/actual-storage-format.md) — LanceDB storage format (schema)
- [`references/lancedb-table-rebuild.md`](references/lancedb-table-rebuild.md) — detailed table-rebuild guide
- [`references/strip-pipeline-architecture.md`](references/strip-pipeline-architecture.md) — Strip Pipeline architecture
- [`references/profile-lancedb-layout.md`](references/profile-lancedb-layout.md) — old-profile LanceDB layout
