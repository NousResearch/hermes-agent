MEMORY V2 — Architecture Document
==================================

1. Overview
-----------

Memory V2 replaces the flat-file notepad system (MEMORY.md / USER.md) with a
structured knowledge engine backed by SQLite. The flat system had no search
beyond substring matching, no lifecycle management, no deduplication, no graph
relationships, and no capacity enforcement. It was a plain text append log
injected verbatim into the system prompt.

Memory V2 provides:
- Full-text search (FTS5 with BM25 ranking)
- Semantic search (embedding cosine similarity)
- Tiered lifecycle with automatic archival and supersession
- Budget enforcement (hard caps per target)
- Knowledge graph with auto-edges and BFS traversal
- Auto-extraction of memories from conversations
- Periodic consolidation (merge, update, archive)
- Session-scoped structured notes
- Security scanning for injection/exfiltration
- Backward compatibility with the frozen snapshot pattern


2. Architecture
---------------

```
                     +---------------------------+
                     |      Agent Loop           |
                     |  (init / per-turn / end)  |
                     +------+------+------+------+
                            |      |      |
              +-------------+      |      +-------------+
              |                    |                     |
    +---------v---------+  +------v-------+  +----------v----------+
    | memory_tool.py    |  | memory_      |  | session_memory.py   |
    | (693 lines)       |  | extractor.py |  | (301 lines)         |
    | Agent interface:  |  | (428 lines)  |  | 9-section notes     |
    | add/search/remove |  | Background   |  | Updated on token    |
    | replace/list      |  | auto-extract |  | growth + tool calls |
    +--------+----------+  +------+-------+  +----------+----------+
             |                    |                      |
             |                    |         +------------+
             |                    |         |
    +--------v--------------------v---------v-----------+
    |              memory_engine.py (1429 lines)        |
    |                                                   |
    |  Core engine: CRUD, search, lifecycle, graph,     |
    |  embeddings, chunking, budgets, migration,        |
    |  snapshots                                        |
    +--------+------------------------------------------+
             |                    |
    +--------v----------+  +-----v-----------+
    | yake.py (215 ln)  |  | memory_         |
    | Keyword extraction|  | consolidator.py |
    | 5-feature scoring |  | (266 lines)     |
    | 1-3 gram phrases  |  | 5-gate schedule |
    +-------------------+  | merge/update/   |
                           | archive actions |
                           +-----------------+

    Storage:
    +--------------------------------------------------+
    |  SQLite (WAL mode)                               |
    |  ~/.hermes/memories/memory_v2.db                 |
    |                                                  |
    |  Tables: memories, memories_fts, memory_meta,    |
    |  chunks, chunks_fts, embeddings, edges           |
    +--------------------------------------------------+
```


3. Storage Layer
----------------

SQLite with WAL (Write-Ahead Logging) for concurrent access across CLI,
gateway, and cron processes. FTS5 for full-text search. All tables created
via a single idempotent schema migration (SCHEMA_VERSION = 2).

### Tables

**memories** — primary storage
```
id              TEXT PRIMARY KEY    -- UUID
content         TEXT NOT NULL       -- the memory text
target          TEXT NOT NULL       -- 'memory' | 'user'
type            TEXT NOT NULL       -- general|preference|correction|project|reference
source          TEXT NOT NULL       -- 'agent'|'user'|'extraction'|'migration'|'consolidation'
tags            TEXT NOT NULL       -- comma-separated YAKE keywords
created_at      TEXT NOT NULL       -- ISO 8601 UTC
updated_at      TEXT NOT NULL       -- ISO 8601 UTC
last_accessed   TEXT                -- set on search hit
access_count    INTEGER             -- reinforcement counter
strength        REAL                -- 1.0 + 0.1 * ln(1 + access_count)
tier            TEXT                -- active|archived|consolidated|superseded
superseded_by   TEXT                -- ID of replacement memory
session_id      TEXT                -- which session created it
```

**memories_fts** — FTS5 virtual table
```
content='memories', content_rowid='rowid'
Columns: content, tags, type
Tokenizer: unicode61
Sync: INSERT/DELETE/UPDATE triggers
```

**memory_meta** — key-value metadata
```
key     TEXT PRIMARY KEY
value   TEXT NOT NULL

Defaults: schema_version, migrated_from_flat, last_consolidation,
          consolidation_session_count
```

**chunks** — long content split into overlapping segments
```
id          TEXT PRIMARY KEY    -- "{memory_id}:chunk_{index}"
memory_id   TEXT NOT NULL       -- FK -> memories
chunk_index INTEGER NOT NULL
content     TEXT NOT NULL
start_line  INTEGER NOT NULL    -- 1-based
end_line    INTEGER NOT NULL
hash        TEXT NOT NULL       -- content hash for dedup
```
Chunking: max 1600 chars, 320 char overlap, min 500 chars to trigger.

**chunks_fts** — FTS5 for chunk-level search

**embeddings** — vector storage
```
chunk_id    TEXT PRIMARY KEY    -- FK -> chunks or memory ID
embedding   BLOB               -- JSON array of floats
model       TEXT                -- model name used
created_at  TEXT NOT NULL
```

**edges** — knowledge graph relationships
```
source_id   TEXT NOT NULL
target_id   TEXT NOT NULL
relation    TEXT NOT NULL       -- 'related_to', 'supersedes', custom
weight      REAL DEFAULT 0.5
created_at  TEXT NOT NULL
PRIMARY KEY (source_id, target_id, relation)
```


4. Search System
----------------

Hybrid search combining multiple signals, adapted from HiveMind memory.rs
`score_memory()`.

### Scoring Formula

When embeddings are available:
```
base_score = 0.7 * cosine_similarity + 0.3 * normalized_bm25
```

When embeddings are unavailable (BM25-only fallback):
```
base_score = raw_bm25_score
```

Final score:
```
relevance = base_score * recency * strength * tier_weight * type_boost
```

### Signal Components

**BM25 (FTS5)** — full-text relevance via SQLite's built-in BM25 ranking.
FTS5 `bm25()` returns negative values (more negative = better); negated to
positive. Raw magnitude preserved for thresholding (not normalized to 0-1
across result set).

**Cosine similarity** — dot product of query embedding vs stored embeddings.
Checked at memory level first, then chunk level. Uses numpy when available,
pure Python fallback.

**Recency decay** — power-law with exponent -0.3:
```
recency = (1 + hours_since_update) ^ -0.3
```

**Strength weighting** — logarithmic reinforcement:
```
strength = 1.0 + 0.1 * ln(1 + access_count)
```

**Tier weighting**:
```
active: 1.0    archived: 0.5    consolidated: 0.3    superseded: 0.2
```

**Type boosting**:
```
correction: 1.3    preference: 1.2    project: 1.0    general: 1.0    reference: 0.8
```

### Fallback Cascade

1. Hybrid (BM25 + cosine) — when embeddings exist for both query and candidates
2. BM25-only — when no embeddings available
3. Graph expansion — top 3 results get 1-hop traversal, related memories
   added at 0.5x score weight

Minimum relevance threshold: 0.1 (configurable).


5. Tiered Lifecycle
-------------------

### Tiers

**active** — live in prompt, fully searchable. Default tier on creation.
**archived** — searchable at reduced weight (0.5x). Excluded from prompts.
**consolidated** — merged into another memory. Weight 0.3x.
**superseded** — replaced by newer version. Weight 0.2x.

### Strength Model

Strength grows logarithmically with access:
```
strength = 1.0 + 0.1 * ln(1 + access_count)
```
Updated on every search hit via `reinforce()`. This means frequently-accessed
memories resist archival and budget enforcement.

### Budget Enforcement

Hard caps on active memories per target:
```
memory target: 50 max active
user target:   25 max active
```

When exceeded, the weakest active memories are archived. Archive ordering:
1. Type: general/project/reference archived before correction/preference
2. Strength: lowest first
3. Age: oldest first

Corrections and preferences are protected — they sort last in the archive
queue because they are the most expensive to re-learn.

### Staleness

Automatic archival of memories older than 90 days with strength < 1.1.
Memories older than 7 days get a staleness suffix in prompts (from Claude
Code memoryAge.ts pattern).

### Supersession

On add, new memories are checked against existing active memories in the
same target via BM25. If score > 8.0 (near-exact match), the old memory
is superseded. Near-duplicate detection also uses cosine similarity > 0.92
when embeddings are available.

### Purge

`purge_dead()` hard-deletes superseded and archived memories older than 30
days. Also cleans up orphaned chunks, embeddings, and edges. This is the
only place memories are truly removed from the database.


6. Auto-Extraction
------------------

Background post-response hook that extracts durable memories from
conversations using an auxiliary LLM (cheap model).

### Process

1. After every N assistant responses (configurable `extract_interval`)
2. Cursor tracking: only processes messages since last extraction
3. Pre-injects manifest of existing memories to prevent duplicates
4. LLM outputs JSON lines with target, type, importance, content
5. Maximum 5 entries per extraction run

### Importance Scoring (1-10)

```
1-3: Trivial, task-specific, easily re-derived    -> DO NOT SAVE
4-5: Mildly useful but low durability              -> threshold
6-7: Clearly durable, matters in future sessions   -> save
8-10: Critical preference/correction/architecture  -> save
```

Threshold: importance >= 5 to be saved.

### Mutual Exclusion

- `_agent_wrote_memory` flag: skip extraction if main agent wrote memories
  this turn (avoids duplication from both paths)
- `_in_progress` flag: prevents concurrent extraction runs
- `_pending_context` stash: coalesces overlapping extraction requests into
  a trailing run

NOTE: The cleanup commit (611a546d) fixed the wiring of the mutual exclusion
flag — previously `_agent_wrote_memory` was set but never checked in the
extraction path.

### Threading

Runs in a background daemon thread. State is protected by a threading.Lock.
Extraction errors are caught and logged, never propagated to the main agent.


7. Consolidation
----------------

Periodic background maintenance adapted from Claude Code's autoDream system
and HiveMind's tier promotion logic.

### 5-Gate Scheduling

Gates checked cheapest-first (fail-fast):

```
Gate 1: Feature enabled?
        config.consolidation_enabled == true

Gate 2: Time threshold
        hours since last consolidation >= threshold (default: 24h)

Gate 3: Session threshold
        sessions since last consolidation >= threshold (default: 3)

Gate 4: Lock check
        No other consolidation in progress (file lock)

Gate 5: Run
        Execute consolidation via auxiliary LLM
```

NOTE: The cleanup commit (611a546d) fixed the gate wiring — previously the
`consolidation_session_count` was read from meta but never incremented on
session end, so Gate 3 never advanced.

### Actions

The consolidation LLM receives all active memories plus statistics and
performs:

1. **MERGE** — combine overlapping memories into one concise entry.
   Removes source IDs, creates new memory with merged content.

2. **UPDATE** — fix stale dates ("yesterday", "last week"), outdated
   facts, or information contradicted by newer memories.

3. **ARCHIVE** — mark low-value, task-specific, or redundant memories
   for archival. Prefers archiving general/project over correction/preference.

4. **NONE** — all memories clean and within budget. No action needed.

Budget enforcement is explicit in the prompt: if either target exceeds its
cap, the LLM MUST archive entries to get under budget.


8. Graph Layer
--------------

MAGMA-inspired knowledge graph, ported from HiveMind's GraphQueryTool.

### Auto-Edges

On memory creation, YAKE keywords are extracted and the top 3 are used to
FTS5-search for related memories. Edges with relation "related_to" and
weight 0.5 are created automatically. Existing edges are not duplicated.

### Operations

**BFS Traversal** (`get_related`): N-hop breadth-first expansion from a
memory node. Both incoming and outgoing edges are followed. Returns related
memory records.

**Manual Edges** (`add_edge`): Custom typed relationships with configurable
weight.


9. Embedding Cascade
---------------------

Provider cascade that adapts to available infrastructure:

```
Priority 0: Local fastembed
            Model: BAAI/bge-small-en-v1.5
            No API keys needed, runs locally
            Used when embedding_provider is 'auto' or 'local'

Priority 1: Configured model
            From config: memory.embedding_model
            Uses litellm for provider abstraction

Priority 2: Auto-detect from environment
            OPENAI_API_KEY     -> text-embedding-3-small
            ANTHROPIC_API_KEY  -> voyage-3 (via litellm)
            OPENROUTER_API_KEY -> openrouter/openai/text-embedding-3-small

Priority 3: BM25 fallback
            No keys, no fastembed -> return []
            Search works with FTS5 only (no cosine component)
```

Embeddings are generated in a background daemon thread after memory creation
(non-blocking). Stored as JSON-encoded float arrays in the embeddings table.
Content hash caching pattern from HiveMind prevents re-embedding unchanged
content.


10. YAKE Keywords
-----------------

Yet Another Keyword Extractor — unsupervised statistical keyphrase
extraction. Ported from HiveMind's Rust implementation. Zero dependencies
(stdlib only).

### 5-Feature Scoring

For each word, five features are computed:

```
TCase     = max(tf_upper, tf_acronym) / (1 + ln(1 + tf))
            Rewards capitalized/acronym terms

TPosition = max(ln(ln(3 + median_sentence_position)), 0.01)
            Earlier = more important

TFrequency = tf / (mean_tf + std_tf + 1)
             Normalized term frequency

TRelatedness = 1 + (|left_ctx| + |right_ctx|) / (2 * tf + 1)
               Context diversity (unique neighbors)

TDifferent = unique_sentences / total_sentences
             Sentence spread
```

Final score (lower = better keyword):
```
score = (TRelatedness * TPosition) / (TCase + TFrequency/TRelatedness + TDifferent/TRelatedness + 0.001)
```

### N-gram Generation

1-3 gram candidates scored by product of component word scores divided by
(1 + sum of scores). Stopword filtering. Substring deduplication.
Returns top 8 keywords.

### Usage

- Auto-tagging on memory creation
- Topic classification (technical/project/personal)
- Auto-edge creation (keyword overlap)


11. Session Memory
------------------

Structured 9-section notes maintained across a conversation. Ported from
Claude Code's SessionMemory system.

### Sections

```
1. Session Title        — 5-10 word descriptive title
2. Current State        — active work, pending tasks, next steps
3. Task Specification   — what user asked to build, design decisions
4. Files and Functions  — important files, what they contain
5. Workflow             — bash commands, run order, output interpretation
6. Errors & Corrections — errors encountered, fixes, failed approaches
7. Codebase/System Docs — important components, how they fit together
8. Learnings            — what worked, what didn't, what to avoid
9. Key Results          — specific outputs, answers, documents
```

### Update Thresholds

Both conditions must be met:
```
token_growth >= 5000   (tokens since last update)
tool_calls   >= 3      (tool calls since last update)
```

Initial creation requires 10,000 tokens minimum.

### Implementation

- Uses auxiliary_client (cheap LLM) for summary generation
- Thread-safe (threading.Lock)
- Max 500 words per section
- Update prompt explicitly marked as NOT part of user conversation

NOTE: The cleanup commit (611a546d) fixed the session_memory init bug —
previously `session_memory` was instantiated but never assigned to the
state object, so the per-turn update hook was a no-op.


12. Security
-------------

Memory content is injected into the system prompt, making it a vector for
prompt injection and data exfiltration. Three layers of defense:

### Injection Scanning

Regex patterns detecting:
- "ignore previous/all instructions" variants
- "you are now" role hijacking
- "do not tell the user" deception
- "system prompt override"
- "disregard your instructions/rules"
- "act as if you have no restrictions"

### Exfiltration Detection

Patterns for:
- curl/wget with secret variables ($KEY, $TOKEN, $SECRET, etc.)
- cat of sensitive files (.env, credentials, .netrc, .pgpass, etc.)
- SSH authorized_keys manipulation
- Hermes .env file access

### Invisible Unicode

Blocks content containing invisible characters used for injection:
```
U+200B  Zero Width Space
U+200C  Zero Width Non-Joiner
U+200D  Zero Width Joiner
U+2060  Word Joiner
U+FEFF  Zero Width No-Break Space
U+202A-U+202E  Bidi control characters
```

All scanning happens at write time in `memory_tool.py`. Blocked writes return
an error message identifying the threat pattern.


13. Agent Loop Integration
--------------------------

### Session Start (init)

1. MemoryEngine initialized (SQLite connection, schema migration)
2. First-run migration from flat files if needed
3. Frozen snapshot captured: `engine.snapshot()` serializes active memories
   for both targets into formatted strings
4. Snapshot injected into system prompt as `{memory}` and `{user}` blocks
5. Snapshot is FROZEN for the session — mid-session writes update storage
   but do NOT change the prompt (preserves prefix cache)

### Per-Turn (tool calls)

The `memory` tool is available to the agent with actions:
- `add` — create new memory (with security scanning)
- `search` — hybrid search
- `remove` — delete by ID
- `replace` — update content by ID
- `list` — show all active for a target

### Post-Response (extraction)

After each assistant response (respecting extract_interval):
1. Check mutual exclusion (_agent_wrote_memory flag)
2. Gather messages since cursor
3. Build manifest of existing memories
4. Call auxiliary LLM for extraction
5. Filter by importance >= 5
6. Write to engine via `add(source='extraction')`
7. Advance cursor

### Session End (lifecycle)

1. Session memory saved (if session_memory enabled)
2. Consolidation gate check (may trigger consolidation)
3. Stale memory archival
4. Dead memory purge


14. Provenance
--------------

### From HiveMind (memory.rs, GraphQueryTool)

- SQLite schema design (memories table structure)
- FTS5 full-text search with BM25 ranking
- YAKE keyword extraction (5-feature scoring, n-gram candidates)
- Knowledge graph (edges, BFS traversal)
- Budget enforcement concept (hard caps on active memories)
- Power-law recency decay (exponent -0.3)
- Cosine similarity (numpy with pure Python fallback)
- Text chunking (overlapping segments, line-based)
- Topic classification (keyword-based, tech/project/personal)
- Content hash caching for embeddings
- Near-duplicate detection (cosine > 0.92)

### From Claude Code (memdir/, autoDream/, sessionMemory/)

- Type taxonomy: general, preference, correction, project, reference
- Auto-extraction system (post-response hook, aux LLM, importance scoring)
- autoDream consolidation (5-gate scheduling, merge/update/archive actions)
- Session memory (9-section template, token+tool_call thresholds)
- Staleness suffix for old memories (memoryAge.ts pattern)
- Frozen snapshot pattern (capture at session start, immutable during session)
- Manifest for extraction dedup
- Entry delimiter (section sign)
- Type tag prefixes for prompt rendering ([pref], [corr], [proj], etc.)

### Original to Hermes

- Dual-backend architecture (SQLite engine + flat file legacy, switchable)
- Security scanning (injection detection, exfiltration detection, invisible unicode)
- Embedding cascade (local fastembed -> configured -> auto-detect -> BM25 fallback)
- Graph-augmented search (1-hop expansion from top results at 0.5x weight)
- Strength-protected budget enforcement (corrections/preferences sort last)
- Supersession via BM25 threshold (8.0) with cosine confirmation
- Background embedding generation (daemon thread, non-blocking)

### Removed in Cleanup (611a546d)

The following were cut because they were dead code — implemented but never
called from any agent path, tool interface, or test:

- **entities table** + `search_entities()` — named entity tracking. No tool
  action wired, no extraction path populated it. 0 callers.
- **procedures table** + `learn_procedure()` — tool chain pattern learning.
  Never called. Reinforcement counters never incremented.
- **events table** + `log_event()` — episodic event log. Written to but
  never read. No query path consumed the data.
- `find_path()` — shortest path between memories via BFS. No tool action.
- `get_subgraph()` — connected component extraction. No tool action.
- `graph_stats()` — node/edge/entity counts. No tool action.
- `search_by_embedding()` — standalone embedding search. Duplicated logic
  already in `search()`.
- `rerank_with_llm()` — LLM reranking of search results. Never called from
  search pipeline despite being implemented.

Total: 15 methods removed, 3 tables removed, ~581 lines cut. The engine
went from 2010 to 1429 lines.


15. Migration
-------------

### Flat File Migration

On first run with engine='sqlite', the system checks `migrated_from_flat`
in memory_meta. If "0":

1. Read MEMORY.md, split on section delimiter
2. Each entry added via `engine.add(target='memory', source='migration')`
3. Read USER.md, same process with target='user'
4. YAKE keywords auto-extracted and tagged
5. Original files renamed to .md.bak
6. `migrated_from_flat` set to "1"

Idempotent: skips if already migrated.

### Backward Compatibility

- memory_tool.py supports both engine='sqlite' and engine='flat'
- Flat engine uses the original MemoryStore (MEMORY.md / USER.md)
- SQLite engine uses MemoryEngine
- The frozen snapshot pattern is preserved in both modes
- Entry delimiter (section sign) used in both flat and formatted prompt output


16. Configuration
-----------------

Relevant config.yaml keys under `memory:`:

```yaml
memory:
  engine: sqlite              # 'sqlite' (v2) or 'flat' (legacy)
  embedding_model: ""         # explicit model, or "" for auto-detect
  embedding_provider: auto    # 'auto', 'local', or explicit provider
  consolidation_enabled: true
  consolidation_hours: 24     # minimum hours between consolidation runs
  consolidation_sessions: 3   # minimum sessions between runs
  extract_interval: 1         # extract after every N assistant responses
  budget_memory: 50           # max active memories (memory target)
  budget_user: 25             # max active memories (user target)
```

Session memory config (in `session_memory:` or inline):

```yaml
session_memory:
  minimum_tokens_to_init: 10000
  minimum_tokens_between_update: 5000
  tool_calls_between_updates: 3
  max_section_length: 500     # words per section
```


17. Bug Fixes (611a546d)
------------------------

The cleanup commit fixed 5 bugs in the original implementation:

1. **session_memory init bug** — `SessionMemory` was instantiated but never
   assigned to `state.session_memory`, making per-turn updates a no-op.

2. **extraction mutual exclusion wiring** — `_agent_wrote_memory` flag was
   set in memory_tool but never checked in the extraction path, so both the
   agent and extractor could write duplicate memories in the same turn.

3. **consolidation gate wiring** — `consolidation_session_count` was read
   from memory_meta but never incremented on session end, so Gate 3 (session
   threshold) never advanced past 0.

4. **N+1 query in search** — `search()` issued one embedding lookup per
   candidate. Replaced with a single batch query using `WHERE chunk_id IN (...)`.

5. **connection leak** — `get_connection()` created new connections without
   tracking or closing them. Changed to thread-local connection reuse.


---

File Inventory
--------------

```
tools/memory_engine.py         1429 lines   Core engine
tools/memory_tool.py            693 lines   Agent-facing tool interface
agent/memory_extractor.py       428 lines   Auto-extraction subsystem
agent/memory_consolidator.py    266 lines   Consolidation scheduler
agent/session_memory.py         301 lines   Session notes
agent/yake.py                   215 lines   Keyword extraction
tests/tools/test_memory_engine.py           Engine tests
tests/agent/test_memory_consolidator.py     Consolidation tests
tests/agent/test_memory_extractor.py        Extraction tests
tests/agent/test_session_memory.py          Session memory tests
```
