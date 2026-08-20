# Holographic Memory System — Agent Development Guide

Instructions for AI coding assistants and developers working with the holographic memory system.

## Overview

The holographic memory system provides structured fact storage with entity resolution, trust scoring, and HRR-based compositional retrieval. It runs as a plugin (`hermes-memory-store`) using the `MemoryProvider` interface.

**Key properties:**
- SQLite-backed (always available, no external dependencies)
- FTS5 full-text search with BM25 ranking
- Entity extraction and resolution (regex-based)
- Trust scoring with asymmetric feedback
- Optional HRR (Holographic Reduced Representations) for compositional queries
- No external API calls — fully local, privacy-first

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Agent (AIAgent)                     │
│  ┌───────────────┐  ┌──────────────────────────────┐ │
│  │ MemoryProvider │  │ Tools: fact_store, fact_feedback│
│  │   interface    │  │ (registered via plugin)       │ │
│  └───────┬───────┘  └──────────┬───────────────────┘ │
│          │                      │                     │
│  ┌───────▼──────────────────────▼───────────────────┐│
│  │         HolographicMemoryProvider                 ││
│  │  ┌─────────────┐  ┌───────────────────────────┐  ││
│  │  │ MemoryStore  │  │     FactRetriever         │  ││
│  │  │ (SQLite+FTS5)│  │  (hybrid search+rerank)   │  ││
│  │  └──────┬──────┘  └──────────┬────────────────┘  ││
│  │         │                    │                    ││
│  │  ┌──────▼────────────────────▼────────────────┐  ││
│  │  │           SQLite Database                   │  ││
│  │  │  facts │ entities │ fact_entities │ banks   │  ││
│  │  └────────────────────────────────────────────┘  ││
│  └───────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

### File Layout

```
plugins/memory/holographic/
├── __init__.py       # HolographicMemoryProvider, plugin entry point
├── store.py          # MemoryStore — SQLite schema, CRUD, entity extraction
├── retrieval.py      # FactRetriever — hybrid search, probe, related, reason
├── holographic.py    # HRR algebra (numpy optional)
└── README.md         # User-facing setup docs
```

### Data Flow

1. **Write path:** `fact_store(action='add')` → `MemoryStore.add_fact()` → entity extraction → entity resolution → HRR vector computation → SQLite insert
2. **Read path:** `fact_store(action='search')` → `FactRetriever.search()` → FTS5 candidates → Jaccard rerank → HRR similarity → trust weighting → results
3. **Entity query:** `fact_store(action='probe')` → HRR unbind from memory bank → vector similarity scoring

---

## Database Schema

```sql
-- Core fact storage
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB           -- HRR vector (optional, numpy-dependent)
);

-- Entity registry
CREATE TABLE entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',   -- comma-separated
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many: facts ↔ entities
CREATE TABLE fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

-- Full-text search index (auto-synced via triggers)
CREATE VIRTUAL TABLE facts_fts USING fts5(
    content, tags,
    content=facts, content_rowid=fact_id
);

-- HRR memory banks (category-level aggregated vectors)
CREATE TABLE memory_banks (
    bank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    bank_name  TEXT NOT NULL UNIQUE,      -- "cat:general", "cat:user_pref", etc.
    vector     BLOB NOT NULL,             -- aggregated HRR vector
    dim        INTEGER NOT NULL,
    fact_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Indexes

```sql
CREATE INDEX idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX idx_facts_category ON facts(category);
CREATE INDEX idx_entities_name  ON entities(name);
```

### FTS5 Triggers (auto-maintained)

Three triggers keep `facts_fts` synchronized with `facts`:
- `facts_ai` — insert
- `facts_ad` — delete
- `facts_au` — update

**Do not manually modify `facts_fts`.** The triggers handle everything.

---

## Tool APIs

### `fact_store` — 9 actions

#### `add` — Store a fact

```json
{
  "action": "add",
  "content": "Timmy uses Python for backend development",
  "category": "project",        // optional: general|user_pref|project|tool
  "tags": "python, backend"     // optional: comma-separated
}
```

Returns: `{"fact_id": 42, "status": "added"}`

**Behavior:**
- Deduplicates by content (UNIQUE constraint) — returns existing `fact_id` on duplicate
- Auto-extracts entities via regex (capitalized phrases, quoted terms, AKA patterns)
- Computes HRR vector if numpy available
- Rebuilds category memory bank

#### `search` — Full-text search

```json
{
  "action": "search",
  "query": "editor config",
  "category": "user_pref",      // optional filter
  "min_trust": 0.3,             // optional: default 0.3
  "limit": 10                   // optional: default 10
}
```

Returns: `{"results": [...], "count": N}`

Each result includes: `fact_id`, `content`, `category`, `tags`, `trust_score`, `retrieval_count`, `helpful_count`, `created_at`, `updated_at`, `score`.

#### `probe` — Entity recall

```json
{
  "action": "probe",
  "entity": "Timmy",
  "category": "project",        // optional
  "limit": 10                   // optional
}
```

Returns all facts where the entity plays a structural role. Uses HRR algebra (unbinds entity from memory bank) when numpy available, falls back to keyword search.

#### `related` — Structural adjacency

```json
{
  "action": "related",
  "entity": "Hermes",
  "limit": 10
}
```

Finds facts connected through shared context — entities mentioned alongside the target, or content that overlaps structurally.

#### `reason` — Multi-entity compositional query

```json
{
  "action": "reason",
  "entities": ["Timmy", "crisis"],
  "limit": 10
}
```

**This is the most powerful query.** Algebraically intersects structural connections to find facts related to ALL entities simultaneously. Uses AND semantics (min of per-entity scores).

Example: `reason(["peppi", "backend"])` finds facts where both peppi AND backend play structural roles — without keyword matching.

#### `contradict` — Memory hygiene

```json
{
  "action": "contradict",
  "category": "project",        // optional
  "limit": 10
}
```

Finds potentially contradictory facts via entity overlap + content divergence. Two facts contradict when they share entities (same subject) but have low content-vector similarity (different claims).

Returns pairs: `{"fact_a": {...}, "fact_b": {...}, "entity_overlap": 0.8, "content_similarity": -0.2, "contradiction_score": 0.96, "shared_entities": ["Timmy"]}`

#### `update` — Modify a fact

```json
{
  "action": "update",
  "fact_id": 42,
  "content": "Updated content",       // optional
  "trust_delta": 0.1,                 // optional: additive adjustment
  "tags": "new,tags",                 // optional
  "category": "tool"                  // optional
}
```

Returns: `{"updated": true/false}`

#### `remove` — Delete a fact

```json
{
  "action": "remove",
  "fact_id": 42
}
```

Returns: `{"removed": true/false}`

#### `list` — Browse facts

```json
{
  "action": "list",
  "category": "user_pref",      // optional
  "min_trust": 0.0,             // optional: default 0.0
  "limit": 50                   // optional: default 50
}
```

---

### `fact_feedback` — Train trust scores

```json
{
  "action": "helpful",          // or "unhelpful"
  "fact_id": 42
}
```

Returns: `{"fact_id": 42, "old_trust": 0.5, "new_trust": 0.55, "helpful_count": 3}`

**Trust adjustment:**
- `helpful` → trust += 0.05
- `unhelpful` → trust -= 0.10
- Clamped to [0.0, 1.0]

---

## Entity Resolution

### Extraction Rules (in order)

1. **Capitalized multi-word phrases** — `John Doe`, `Timmy Foundation`
2. **Double-quoted terms** — `"Python"`
3. **Single-quoted terms** — `'pytest'`
4. **AKA patterns** — `Guido aka BDFL` → two entities: `Guido`, `BDFL`

### Resolution

- Case-insensitive exact match on `entities.name`
- Alias search: comma-separated `aliases` field, matched with `%boundary%` LIKE
- Creates new entity if no match found

### Linking

- `fact_entities` junction table (many-to-many)
- `INSERT OR IGNORE` prevents duplicate links
- On fact update with new content, entity links are rebuilt

---

## Trust Scoring

### Default Trust

New facts start at `default_trust` (configurable, default 0.5).

### Feedback Loop

```
helpful=True   → trust += 0.05
helpful=False  → trust -= 0.10
```

Asymmetric: bad feedback hurts more than good feedback helps. This is intentional — it makes trust drift downward for unverified facts.

### Trust Bounds

Clamped to [0.0, 1.0] — always.

### Trust in Retrieval

- `min_trust` filter excludes low-trust facts from search results (default 0.3)
- `score = relevance * trust_score` — trust directly multiplies relevance
- Trust persists across sessions (stored in SQLite)

---

## Retrieval Pipeline

### Search (`FactRetriever.search()`)

```
Query
  │
  ▼
FTS5 Candidates (limit × 3)
  │
  ▼
Jaccard Rerank (token overlap)
  │
  ▼
HRR Similarity (optional, numpy)
  │
  ▼
Combined Score = fts_weight × fts_rank
              + jaccard_weight × jaccard
              + hrr_weight × hrr_sim
  │
  ▼
Trust Weighting: score ×= trust_score
  │
  ▼
Temporal Decay (optional): score ×= 0.5^(age_days / half_life)
  │
  ▼
Sort by score, return top N
```

### Weight Redistribution

If numpy unavailable:
- `fts_weight` → 0.6
- `jaccard_weight` → 0.4
- `hrr_weight` → 0.0

### Probe (`FactRetriever.probe()`)

1. Encode entity as role-bound vector: `bind(entity_vec, role_entity)`
2. Unbind from memory bank or individual fact vectors
3. Compare residual to content signal
4. Score = similarity × trust

### Reason (`FactRetriever.reason()`)

1. For each entity, compute probe key
2. Unbind each probe key from each fact vector
3. AND semantics: `min(per_entity_scores)`
4. Score = (min_sim + 1) / 2 × trust

### Contradict (`FactRetriever.contradict()`)

1. Get all facts with vectors + their linked entities
2. For each pair: entity overlap (Jaccard) + content similarity (HRR)
3. `contradiction_score = entity_overlap × (1 - normalized_content_sim)`
4. Threshold: 0.3 default

---

## Integration Patterns

### In the System Prompt

The provider's `system_prompt_block()` returns a context block injected into the system prompt:

```
# Holographic Memory
Active. 42 facts stored with entity resolution and trust scoring.
Use fact_store to search, probe entities, reason across entities, or add facts.
Use fact_feedback to rate facts after using them (trains trust scores).
```

### Prefetch on Query

`prefetch(query)` runs before each agent turn. Returns top-5 matching facts:

```
## Holographic Memory
- [0.8] user prefers dark mode
- [0.6] project uses PostgreSQL
```

These appear in the system prompt context, giving the agent memory without tool calls.

### Auto-Extraction

When `auto_extract: true` in config, `on_session_end()` scans user messages for:
- Preference patterns: "I prefer/like/love/use/want/need X"
- Decision patterns: "we decided/agreed/chose to X"

Extracted facts are stored with `default_trust` (0.5) — lower than manually added facts.

### Mirroring Built-in Memory

`on_memory_write(action='add', target, content)` mirrors built-in `memory` tool writes as facts. Category: `user_pref` for user-targeted, `general` otherwise.

---

## Configuration

In `config.yaml` under `plugins.hermes-memory-store`:

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memory_store.db` | SQLite database path |
| `auto_extract` | `false` | Auto-extract facts at session end |
| `default_trust` | `0.5` | Default trust score for new facts |
| `hrr_dim` | `1024` | HRR vector dimensions |
| `hrr_weight` | `0.3` | Weight of HRR similarity in hybrid search |
| `min_trust_threshold` | `0.3` | Minimum trust for search results |
| `temporal_decay_half_life` | `0` | Days for 50% decay (0 = disabled) |

### Setup

```bash
hermes memory setup    # select "holographic"
```

Or manually:
```bash
hermes config set memory.provider holographic
```

---

## Testing

### Running Tests

```bash
source venv/bin/activate
python -m pytest tests/agent/test_memory_provider.py -v
```

### Test Patterns

```python
# Isolate HERMES_HOME for each test
@pytest.fixture
def isolated_store(tmp_path):
    db_path = str(tmp_path / "test_memory.db")
    store = MemoryStore(db_path=db_path, default_trust=0.5)
    return store

# Test entity extraction
def test_extract_entities(store):
    fact_id = store.add_fact("Timmy Foundation uses Python for backend")
    # Verify entities extracted
    ...

# Test trust scoring
def test_trust_feedback(store):
    fid = store.add_fact("test fact")
    result = store.record_feedback(fid, helpful=True)
    assert result["new_trust"] == 0.55
```

### Key Test Scenarios

1. **Deduplication** — adding duplicate content returns existing `fact_id`
2. **Entity resolution** — "John Doe" and "John" resolve to same entity
3. **Trust bounds** — trust never exceeds [0.0, 1.0]
4. **FTS5 sync** — search finds newly added facts
5. **HRR fallback** — system works without numpy
6. **Auto-extraction** — preference patterns are captured
7. **Contradiction detection** — conflicting facts are surfaced

---

## Performance Considerations

### SQLite Optimization

- WAL mode enabled by default
- FTS5 uses content=external (synced via triggers)
- Indexes on `trust_score DESC` and `category`

### Scale Limits

- **Facts:** Tested to ~10K facts. Above that, consider periodic VACUUM and index optimization.
- **Contradiction detection:** O(n²) on fact pairs. Guarded at 500 facts max per check.
- **HRR vectors:** Each vector is `hrr_dim × 8 bytes` (default: 1024 × 8 = 8KB per fact). At 10K facts: ~80MB.

### Memory

- `MemoryStore` holds a single SQLite connection (thread-safe with `check_same_thread=False`)
- HRR vectors are loaded on-demand, not cached in memory
- `FactRetriever` is stateless — no caching between queries

### Concurrency

- `threading.RLock` protects all write operations
- SQLite WAL mode allows concurrent reads
- No distributed locking — single-process only

---

## Common Pitfalls

### 1. Fact content must be unique

```python
store.add_fact("Timmy likes Python")
store.add_fact("Timmy likes Python")  # Returns existing fact_id, no error
```

This is by design. If you need variant facts, add distinguishing context to the content.

### 2. Entity extraction is regex-based

It catches capitalized names and quoted terms, but misses:
- Single-word lowercase entities ("timmy" won't be extracted unless quoted)
- Entities in ALL CAPS
- Complex multi-word names with lowercase particles ("van der Berg")

For critical entities, include them in quotes or capitalized in the fact content.

### 3. HRR is optional, not required

The system works fully without numpy. FTS5 + Jaccard provides good retrieval. HRR adds compositional queries (`reason`, `probe`, `related`) but isn't required for basic search.

### 4. Trust drifts downward by default

If no feedback is given, trust stays at `default_trust` (0.5). But unhelpful feedback (-0.10) hurts more than helpful feedback (+0.05) helps. This means actively verified facts rise, while ignored facts stay neutral.

### 5. Auto-extraction captures preferences, not facts

The auto-extraction patterns (`I prefer X`, `we decided Y`) are heuristic. They capture user preferences and decisions, but miss nuanced facts. Manual `fact_store(action='add')` is always more reliable.

### 6. Memory banks are per-category

`probe()` and `reason()` try category-specific banks first. If a bank doesn't exist (few facts in that category), they fall back to individual fact vectors. This means entity queries work even with sparse data, but are more accurate with more facts.

### 7. FTS5 MATCH syntax

FTS5 uses its own query syntax. Special characters (`-`, `"`, `*`) have meaning. If search returns unexpected results, check the query syntax:
- `"exact phrase"` — phrase search
- `term1 OR term2` — boolean
- `prefix*` — prefix search

### 8. Thread safety

`MemoryStore` uses `threading.RLock` for writes. SQLite connection uses `check_same_thread=False`. Multiple threads can read simultaneously (WAL mode), but writes are serialized.

---

## Quick Reference

### Common Operations

```python
# Add a fact
fact_store(action='add', content='Timmy prefers vim', category='user_pref')

# Search
fact_store(action='search', query='editor preferences')

# Find facts about an entity
fact_store(action='probe', entity='Timmy')

# Cross-entity query
fact_store(action='reason', entities=['Timmy', 'editor'])

# Check for contradictions
fact_store(action='contradict')

# Train trust
fact_feedback(action='helpful', fact_id=42)
```

### Environment

- **Provider:** SQLite (always available)
- **HRR:** numpy optional (pip install numpy)
- **Config:** `plugins.hermes-memory-store` in config.yaml
- **DB:** `$HERMES_HOME/memory_store.db` (profile-scoped)

---

*Sovereignty and service always.*
