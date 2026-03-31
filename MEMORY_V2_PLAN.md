# Hermes Memory System V2 — Implementation Plan

**Branch:** `feat/memory-system-v2`
**Goal:** Replace flat-file memory (2 files, 3.5KB cap, no search, no extraction) with a
SQLite-backed memory engine with hybrid search, automatic extraction, type taxonomy,
tiered lifecycle, and scheduled consolidation.

**Sources cannibalized:**
- HiveMind (~/HiveMind/HIVE/src-tauri/src/memory.rs) — SQLite schema, hybrid search,
  tiers, YAKE keywords, lifecycle management, power-law decay
- Claude Code (~/claude-code-leaked/src/) — auto-extraction, autoDream scheduling,
  memory type taxonomy, relevance selection, consolidation prompts

**Constraints:**
- MUST be backward-compatible (existing MEMORY.md/USER.md migrated on first run)
- MUST NOT break prompt caching (frozen snapshot pattern preserved)
- MUST NOT add required dependencies (embeddings are optional enhancement)
- MUST preserve existing test suite (32 memory tests + gateway flush tests)
- MUST work headless (no UI required — CLI, Telegram, Discord all work)
- All existing config.yaml options continue to work

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    System Prompt                      │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ MEMORY block  │  │  USER block  │  (frozen at      │
│  │ (selected)    │  │  (selected)  │   session start)  │
│  └──────────────┘  └──────────────┘                  │
└──────────┬──────────────────┬────────────────────────┘
           │                  │
     relevance_select()  relevance_select()
           │                  │
┌──────────┴──────────────────┴────────────────────────┐
│              MemoryEngine (NEW)                        │
│  ┌────────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ MemoryStore │ │ Searcher │ │ LifecycleManager   │  │
│  │  (SQLite)   │ │(FTS5+opt │ │ (promote/archive/  │  │
│  │             │ │ vector)  │ │  consolidate)      │  │
│  └────────────┘ └──────────┘ └────────────────────┘  │
└──────────┬──────────────────────────────────────────┘
           │
     ┌─────┴──────────────────────────────┐
     │         Auto-Extractor              │
     │  (post-response hook, aux LLM)      │
     └─────────────────────────────────────┘
```

---

## Phase 1: SQLite Memory Schema + Migration

**Files:** `tools/memory_engine.py` (NEW), `tools/memory_tool.py` (MODIFY)

### Schema (memory.db in ~/.hermes/memories/)

```sql
-- Core memory storage
CREATE TABLE memories (
    id          TEXT PRIMARY KEY,       -- uuid4
    content     TEXT NOT NULL,
    target      TEXT NOT NULL DEFAULT 'memory',  -- 'memory' or 'user'
    type        TEXT DEFAULT 'general', -- general/preference/correction/project/reference
    source      TEXT DEFAULT 'agent',   -- agent/extraction/consolidation/migration
    tags        TEXT DEFAULT '',         -- comma-separated
    created_at  TEXT NOT NULL,          -- ISO8601
    updated_at  TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    strength    REAL DEFAULT 1.0,       -- logarithmic reinforcement
    tier        TEXT DEFAULT 'active',  -- active/archived/consolidated/superseded
    superseded_by TEXT,                 -- id of newer memory
    session_id  TEXT                    -- session where created
);

-- Full-text search
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, tags, type,
    content='memories', content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers for FTS sync
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, type)
    VALUES (new.rowid, new.content, new.tags, new.type);
END;
CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
    VALUES ('delete', old.rowid, old.content, old.tags, old.type);
END;
CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, type)
    VALUES ('delete', old.rowid, old.content, old.tags, old.type);
    INSERT INTO memories_fts(rowid, content, tags, type)
    VALUES (new.rowid, new.content, new.tags, new.type);
END;

-- Schema version
CREATE TABLE memory_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
INSERT INTO memory_meta VALUES ('schema_version', '1');
INSERT INTO memory_meta VALUES ('migrated_from_flat', '0');
```

### Migration from flat files

On first run, if MEMORY.md/USER.md exist and memory.db doesn't:
1. Parse existing entries (split by §)
2. Insert each as a memory with source='migration', tier='active'
3. Set migrated_from_flat='1' in meta
4. Keep flat files as backup (rename to .bak)

### MemoryEngine class

```python
class MemoryEngine:
    """SQLite-backed memory store with FTS5 search and tiered lifecycle."""
    
    def __init__(self, db_path=None, config=None): ...
    
    # Core CRUD
    def add(self, content, target='memory', type='general', tags='', source='agent', session_id=None) -> dict
    def replace(self, memory_id, new_content) -> dict
    def remove(self, memory_id) -> dict
    def get(self, memory_id) -> dict
    
    # Search
    def search(self, query, target=None, limit=10, min_relevance=0.1) -> list[dict]
    def search_fts(self, query, target=None, limit=20) -> list[dict]  # BM25 only
    
    # Lifecycle
    def reinforce(self, memory_id): ...  # increment access_count, update strength
    def archive_stale(self, days=90, min_strength=1.1): ...
    def supersede(self, old_id, new_id): ...
    def consolidate_similar(self, threshold=0.85): ...  # needs embeddings
    
    # For system prompt injection
    def get_active_memories(self, target, limit=None) -> list[dict]
    def format_for_prompt(self, target, selected_ids=None) -> str
    
    # Snapshot for cache stability
    def snapshot(self) -> dict:  # frozen at session start
    def format_snapshot(self, target) -> str  # render frozen snapshot
    
    # Stats
    def stats(self) -> dict  # counts by target, tier, type
    
    # Migration
    def migrate_from_flat_files(self): ...
```

### Compatibility layer

The existing MemoryStore class wraps MemoryEngine, presenting the same interface.
run_agent.py continues to call MemoryStore methods. MemoryStore delegates to MemoryEngine.

---

## Phase 2: Search + Retrieval

**From HiveMind:** Hybrid BM25 + recency scoring. No vector embeddings initially (optional later).

### Scoring function (adapted from HiveMind's memory.rs)

```python
def score_memory(memory, query_tokens, now):
    """Score a memory for relevance to a query."""
    # BM25 score from FTS5 (normalized 0-1)
    bm25 = memory['bm25_score']
    
    # Recency: power-law decay (from HiveMind)
    hours = (now - memory['updated_at']).total_seconds() / 3600
    recency = (1 + hours) ** -0.3
    
    # Strength: logarithmic reinforcement (from HiveMind)
    strength = 1.0 + 0.1 * math.log(1 + memory['access_count'])
    
    # Tier weight (from HiveMind)
    tier_weights = {'active': 1.0, 'archived': 0.5, 'consolidated': 0.3, 'superseded': 0.2}
    tier_w = tier_weights.get(memory['tier'], 0.5)
    
    # Type boost (preferences/corrections are more valuable than general)
    type_boost = {'preference': 1.2, 'correction': 1.3, 'project': 1.0, 'reference': 0.8, 'general': 1.0}
    type_w = type_boost.get(memory['type'], 1.0)
    
    return bm25 * recency * strength * tier_w * type_w
```

### Budget control (from HiveMind)

Inject at most 10% of model context window as memory. For 200K context = 20K tokens = ~80K chars.
Current system injects everything (~3.5KB). New system can inject much more but caps at budget.

---

## Phase 3: Automatic Memory Extraction

**From Claude Code:** Post-response hook that extracts durable memories.
**Adapted for Hermes:** Uses auxiliary_client (cheap model), not a forked full agent.

### Hook point: after each assistant response in run_conversation()

```python
# In run_agent.py, after response delivered:
if self._memory_engine and self._auto_extract_enabled:
    if self._turns_since_extraction >= self._extract_interval:
        self._extract_memories_background(recent_messages)
        self._turns_since_extraction = 0
```

### Extraction prompt (adapted from Claude Code + HiveMind quality scoring)

```
Review the recent conversation and extract any durable facts worth remembering.

EXISTING MEMORIES (do not duplicate):
{current_memory_manifest}

RECENT MESSAGES:
{last_N_messages}

EXTRACT memories that are:
- User preferences, corrections, or personal details
- Environment facts (OS, tools, project structure)
- Conventions or workflow patterns
- Corrections to your previous behavior

DO NOT extract:
- Task progress or session outcomes
- Temporary debugging state
- Facts easily re-derived from files
- Anything already in existing memories

For each memory, output JSON:
{"action": "add", "target": "memory"|"user", "type": "preference"|"correction"|"project"|"reference"|"general", "content": "..."}
```

### Integration with existing flush_memories()

The existing flush_memories() and background review system continue to work.
Auto-extraction is a SUPPLEMENT, not a replacement. It runs more frequently
(every 3 turns vs every 10) but is lighter weight (aux model, structured output).

---

## Phase 4: Prompt Builder Integration

### Frozen snapshot with relevance selection

At session start:
1. Load all active memories from MemoryEngine
2. If total chars < budget: inject all (current behavior, cache-friendly)
3. If total chars > budget: use FTS5 search against initial user message to rank,
   then inject top N within budget

Snapshot is frozen for the session (preserving prefix cache).

### Memory blocks in system prompt

```
══════════════════════════════════════════════
MEMORY (your personal notes) [85% — 1,870/2,200 chars]
══════════════════════════════════════════════
[pref] User prefers terse responses, swears when frustrated — fix fast
§
[proj] ImpactProtocol: GeoTracker, milestones. Backlog: pic upload, 3-step form
§
[corr] DISCORD OUTPUT: only final text visible. Narrate conclusions explicitly.
```

Type tags ([pref], [proj], [corr], [ref], [gen]) are added automatically.
Existing format preserved for backward compatibility.

---

## Phase 5: Memory Tool Upgrade

### New schema (backward-compatible)

Add optional parameters to existing tool:
- `type`: preference/correction/project/reference/general (default: general)
- `search_query`: NEW action "search" — returns matching memories with scores

### New action: "search"

```json
{"action": "search", "target": "memory", "search_query": "discord output formatting"}
```

Returns ranked results with scores, enabling the model to find specific memories
before deciding to update/replace them.

### Dedup upgrade

Replace exact-match dedup with FTS5 similarity check:
- Before adding, search for similar content
- If BM25 score > threshold, suggest replace instead of add
- Report near-duplicates in response

---

## Phase 6: Consolidation via Cron

**From Claude Code:** autoDream with 5-gate system.
**Adapted for Hermes:** Runs as a cron job via Hermes' existing cron infrastructure.

### Gate system (cheapest first, from Claude Code)

1. Feature enabled? (config memory.consolidation_enabled)
2. Time: hours since last consolidation >= 24
3. Sessions: 5+ sessions since last consolidation
4. Lock: prevent concurrent consolidation

### Consolidation prompt (adapted from Claude Code + HiveMind)

```
You are reviewing and consolidating memories for an AI agent.

CURRENT MEMORIES:
{all_active_memories_with_metadata}

RECENT SESSION SUMMARIES:
{last_5_session_summaries}

TASKS:
1. Merge memories that cover the same topic
2. Update stale facts (convert relative dates to absolute)
3. Delete memories contradicted by newer information
4. Mark low-value memories for archival
5. Ensure total active memory stays under budget

Output JSON array of operations:
[
  {"action": "merge", "ids": ["id1", "id2"], "merged_content": "..."},
  {"action": "update", "id": "...", "new_content": "..."},
  {"action": "archive", "id": "...", "reason": "..."},
  {"action": "supersede", "old_id": "...", "new_id": "...", "reason": "..."}
]
```

### Cron job registration

```python
# Auto-registered when memory.consolidation_enabled = true
# Runs every 24h, checks gates before executing
```

---

## File Change Summary

### New files:
- `tools/memory_engine.py` — SQLite engine, search, lifecycle (~400 lines)
- `agent/memory_extractor.py` — Auto-extraction hook (~150 lines)
- `agent/memory_consolidator.py` — Consolidation logic (~200 lines)
- `tests/tools/test_memory_engine.py` — Engine tests (~300 lines)
- `tests/agent/test_memory_extractor.py` — Extraction tests (~100 lines)

### Modified files:
- `tools/memory_tool.py` — MemoryStore wraps MemoryEngine, new search action (~100 lines changed)
- `run_agent.py` — Hook extraction, MemoryEngine init (~50 lines changed)
- `agent/prompt_builder.py` — Type-tagged memory formatting (~20 lines)
- `hermes_cli/config.py` — New config options in DEFAULT_CONFIG (~10 lines)

### Unchanged files:
- `hermes_state.py` — Session DB stays separate (different concern)
- `tools/todo_tool.py` — Unrelated
- Gateway flush paths — Continue to work via MemoryStore compatibility layer

### Estimated total: ~1,300 lines new + ~180 lines modified

---

## Config Changes (config.yaml)

```yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200      # backward-compat (now soft limit for prompt injection)
  user_char_limit: 1375         # backward-compat
  nudge_interval: 10            # existing
  flush_min_turns: 6            # existing
  # NEW v2 options:
  engine: sqlite                # 'sqlite' (new) or 'flat' (legacy)
  auto_extract: true            # automatic memory extraction
  extract_interval: 3           # extract every N turns
  consolidation_enabled: true   # periodic consolidation
  consolidation_interval_hours: 24
  consolidation_min_sessions: 5
  prompt_budget_pct: 10         # max % of context window for memory
  search_min_relevance: 0.1     # minimum score for search results
```

---

## Implementation Order

1. **Phase 1** — MemoryEngine + schema + migration (foundation)
2. **Phase 5** — Memory tool upgrade (enables testing immediately)
3. **Phase 2** — Search + scoring (makes tool useful)
4. **Phase 4** — Prompt builder integration (connects to agent)
5. **Phase 3** — Auto-extraction (the big behavioral change)
6. **Phase 6** — Consolidation cron (maintenance layer)

Tests written alongside each phase. Full suite run after each phase.

---

## Risk Mitigation

- **Migration failure:** Flat file backup (.bak) preserved. Config `engine: flat` falls back.
- **Cache invalidation:** Frozen snapshot pattern preserved. No mid-session prompt changes.
- **Performance:** SQLite WAL mode, FTS5 is fast. No vector search by default.
- **Dependency creep:** Zero new dependencies. SQLite is stdlib. FTS5 is built-in.
- **Existing tests:** MemoryStore compatibility layer means 32 existing tests pass unchanged.
