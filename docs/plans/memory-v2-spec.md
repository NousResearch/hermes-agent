# Memory v2 Specification and Evaluation Contract

> **For Hermes:** Use this document as the memory contract before implementing `memory_v2`. Implementation should be test-driven against `tests/fixtures/memory_v2_benchmark.yaml` and future tests under `tests/memory_v2/`.

**Goal:** Build a robust, low-compute, source-grounded memory system for Hermes that keeps short-term task state sharp and long-term memory more reliable than context-window accumulation.

**Eval docs:** See [`../memory-v2-evals.md`](../memory-v2-evals.md) for local deterministic eval purpose, metrics, fixtures, and regression commands.

**Architecture:** Memory v2 is a profile-scoped external memory provider with human-readable canonical files, SQLite indexes, optional embeddings, and lightweight temporal graph links. It should retrieve small routed memory packets per turn, not dump a large memory blob into the prompt.

**Primary insight:** Raw logs are evidence. Summaries are indexes. Semantic memories are current beliefs. Skills are procedures. The prompt receives only a small selected packet.

---

## 1. Non-negotiable requirements

1. **Profile-safe:** all persistent data lives under the active `hermes_home`; never hardcode `~/.hermes`.
2. **Low online compute:** ordinary user turns should use cheap routing, FTS/BM25, small working/core memory, and bounded packet composition.
3. **Source-grounded:** durable semantic memories must carry source references wherever possible.
4. **Inspectable:** canonical memory records should be readable/editable files. SQLite/vector/graph indexes are derived and rebuildable.
5. **Selective:** do not blindly accumulate old summaries. Compose active context by routing and weighting relevant prior states.
6. **Gated writes:** every candidate memory is classified before promotion, update, supersession, expiry, or rejection.
7. **Temporal:** facts can change. Records need `created_at`, `updated_at`, status, optional validity windows, and supersession links.
8. **Procedures stay skills:** repeatable workflows belong in Hermes skills; Memory v2 may point to skills but should not duplicate procedural docs.
9. **Prompt-cache aware:** stable identity/core memory may appear in `system_prompt_block()`, but dynamic recall must use `prefetch()` memory context.
10. **Measurable:** retrieval quality, stale-fact behavior, irrelevant-memory suppression, and source recall must be tested.

---

## 2. Inspirations translated into design constraints

### Attention Residuals

Attention Residuals replace fixed residual accumulation with selective attention over prior layer outputs. Memory v2 should likewise avoid uncontrolled accumulation of old summaries. Active context is a selective aggregate over prior memory states.

Design rules:
- project summaries are periodically rewritten, not endlessly appended;
- stale claims are superseded instead of left equally active;
- memory packet composition chooses relevant records under a budget.

### Kimi Linear / Kimi Delta Attention

KDA-style finite-state memory suggests compact state plus gates.

Design rules:
- maintain compact working/project/core states;
- gate every write: discard, archive, candidate, promote, update, supersede, expire, skill candidate;
- use expensive global recall only when needed.

### MoBA / block attention

Block routing suggests retrieving coherent blocks before tiny chunks.

Design rules:
- retrieve project/session/entity blocks first;
- drill down into chunks/raw sources only after block selection;
- avoid pure vector-only retrieval.

### Titans / Infini-attention / LightMem

Online path should be cheap; offline consolidation should do the heavy lifting.

Design rules:
- online: route, retrieve compact packet, answer, append event, create candidates;
- offline/session-end/daily: summarize, promote, link entities, resolve contradictions, rebuild indexes.

### GraphRAG / HippoRAG / Zep / Graphiti

Graphs help relational and temporal recall, but full GraphRAG is unnecessary at first.

Design rules:
- implement lightweight entities and edges;
- use graph expansion only after lexical/semantic candidates or for relational queries;
- keep graph records temporal and source-backed.

---

## 3. Memory layers

### Layer 0: Raw archive

Purpose: evidence, reconstruction, auditing, correction.

Examples:
- user/assistant turn summaries;
- session ids;
- tool-call references;
- source file references;
- candidate extraction traces.

Raw archive is cheap to store but normally not injected.

### Layer 1: Working memory

Purpose: current task/session state.

Properties:
- small;
- frequently overwritten;
- archived at session end;
- not treated as durable truth.

Canonical path:

```text
memory_v2/working/current.yaml
memory_v2/working/open_loops.yaml
```

### Layer 2: Core memory

Purpose: stable high-confidence facts and preferences useful across most sessions.

Examples:
- stable user preferences;
- active assistant identity/profile notes;
- active projects list;
- environment facts that affect tool use.

Core memory has the strictest promotion threshold.

### Layer 3: Semantic memory

Purpose: durable facts, preferences, beliefs, constraints, project states, environment details.

Every semantic memory must have status and source references where possible.

### Layer 4: Episodic memory

Purpose: historical records of what happened, especially useful for “where did we leave off?”

Episodic memories should not automatically become stable facts. They are time-bound event summaries.

### Layer 5: Procedural memory pointers

Purpose: point to skills/workflows.

Actual procedures belong in `~/.hermes/skills/` or repo skills, not in semantic fact blobs.

### Layer 6: Derived indexes

Purpose: fast retrieval.

Indexes are rebuildable:
- SQLite FTS5;
- optional vector index;
- optional lightweight graph index;
- retrieval logs.

---

## 4. Proposed profile-scoped file layout

```text
{hermes_home}/memory_v2/
  README.md
  config.yaml

  working/
    current.yaml
    open_loops.yaml
    recent_entities.yaml

  core/
    user.yaml
    assistant_identity.yaml
    environment.yaml
    active_projects.yaml

  inbox/
    raw_events.jsonl
    candidates.jsonl
    rejected.jsonl

  semantic/
    facts.yaml
    preferences.yaml
    beliefs.yaml
    constraints.yaml
    projects/
    environment/

  episodic/
    daily/
    sessions/

  graph/
    entities.yaml
    edges.yaml

  indexes/
    memory.sqlite
    vector/
    rebuild_state.json

  evals/
    memory_benchmark.yaml
    retrieval_regressions.jsonl
    stale_fact_tests.yaml

  reports/
    daily_consolidation/
    weekly_reflection/
```

---

## 5. Canonical schemas

These are logical schemas. Implementation can use dataclasses/Pydantic/TypedDict, but file records should remain human-readable.

### 5.1 Source reference

```yaml
source_ref:
  id: source_...
  type: session | message | file | tool_result | memory | skill | web | manual
  uri: "session:..."  # or file path, URL, tool-call id, etc.
  title: "Short human-readable label"
  observed_at: "2026-05-26T00:00:00Z"
  quote: "Optional short exact quote or excerpt"
```

### 5.2 Memory item

```yaml
id: mem_...
type: fact | preference | belief | constraint | environment | project_state | episode | procedure_ref
subject: "user"
predicate: "prefers_response_style"
value: "direct, no-BS, tool-grounded help"
body: null
summary: "The user prefers direct, tool-grounded help."
status: active  # active | superseded | uncertain | archived | rejected
confidence: 0.95
importance: 0.9
created_at: "2026-05-26T00:00:00Z"
updated_at: "2026-05-26T00:00:00Z"
valid_from: null
valid_until: null
expires_at: null
source_refs:
  - source_...
supersedes: []
superseded_by: null
tags:
  - user_preference
```

### 5.3 Project card

```yaml
id: project:hermes-memory-v2
name: "Hermes Memory v2"
status: active  # active | paused | archived
importance: 0.85
updated_at: "2026-05-26T00:00:00Z"
goal: "Build robust, low-compute, source-grounded memory for Hermes."
why_it_matters: "Improves long-term continuity beyond context-window limits."
current_state: "Spec and benchmark-first implementation planning."
decisions:
  - "Use external MemoryProvider first, not a built-in memory rewrite."
open_questions:
  - "How aggressive should candidate auto-promotion be?"
next_actions:
  - "Implement file store and SQLite FTS."
source_refs:
  - source_...
related_entities:
  - user
  - Hermes
  - Attention Residuals
injection_policy:
  inject_when:
    - "user asks about memory_v2"
    - "user asks where memory work left off"
    - "code changes touch memory provider architecture"
  default_budget_tokens: 500
```

### 5.4 Working memory

```yaml
session_id: "..."
updated_at: "2026-05-26T00:00:00Z"
focus:
  task: "Design Memory v2"
  intent: "Define robust long-term memory architecture and implementation contract."
  constraints:
    - "low online compute"
    - "source-grounded recall"
    - "human-inspectable files"
  active_entities:
    - user
    - Hermes
    - memory_v2
  decisions_made: []
  unresolved_questions: []
  next_actions: []
scratchpad:
  relevant_paths: []
  relevant_commands: []
  retrieved_memory_ids: []
```

### 5.5 Candidate memory

```yaml
id: cand_...
created_at: "2026-05-26T00:00:00Z"
type: project_state
claim: "The user wants Memory v2 to be robust, low-compute, and source-grounded."
proposed_destination: "semantic/projects/hermes-memory-v2.yaml"
importance: 0.8
confidence: 0.9
promotion_reason: "Explicit user request; likely relevant during implementation."
source_refs:
  - source_...
gate_decision: pending  # pending | promoted | rejected | archived_only | superseded
```

### 5.6 Memory packet

```yaml
route: project_continuity
confidence: high
token_budget: 1200
items:
  - id: project:hermes-memory-v2
    type: project_state
    summary: "..."
    source_refs: [source_...]
warnings:
  - "Some retrieved items are older than 90 days."
```

---

## 6. Read/retrieval policy

### 6.1 Query classes

Initial router classes:

- `no_memory_needed`
- `core_personalization`
- `current_task`
- `project_continuity`
- `past_conversation_exact`
- `preference_recall`
- `procedure_lookup`
- `environment_fact`
- `research_recall`
- `contradiction_check`
- `deep_recall`

### 6.2 Routing rules v0

Start rule-based, then measure before replacing with LLM routing.

Examples:

- If the query contains “where did we leave”, “what were we doing”, “continue”, or known project names: route to `project_continuity`.
- If the query asks “what do I prefer”, “how do I like”, or “remember that I”: route to `preference_recall` or candidate write path.
- If the query asks how to perform a workflow: route to `procedure_lookup` and skills.
- If the query asks for exact prior wording/date/source: route to `past_conversation_exact` and require source refs.
- If no durable context is likely useful: route to `no_memory_needed`.

### 6.3 Hybrid ranking

Use multiple signals:

```text
score =
  lexical_bm25 * lexical_weight
+ semantic_similarity * semantic_weight
+ entity_match * entity_weight
+ recency * recency_weight
+ importance * importance_weight
+ source_reliability * source_weight
- stale_penalty
- contradiction_penalty
```

Weights depend on route. Exact IDs/paths/names should heavily weight lexical search. Conceptual recall can weight semantic similarity more.

### 6.4 Packet composition

Default packet budget: 800–1500 tokens.

Deep recall packet budget: 3000–6000 tokens, only when the query needs it.

Packet should include:
- route;
- confidence;
- relevant records;
- status/staleness warnings;
- source refs;
- explicit uncertainty when needed.

Do not return provider-generated `<memory-context>` wrappers; Hermes already fences provider output.

### 6.5 Source verification

Require exact source retrieval for:
- high-stakes claims;
- claims involving external side effects;
- exact quotes/dates/IDs/paths;
- suspected contradictions;
- user asks “when/where did I say that?”;
- memory edits that supersede prior facts.

---

## 7. Write policy

### 7.1 Online write path

After a successful, non-interrupted turn:

1. Append raw event or source ref.
2. Update working memory if the task focus changed.
3. Generate candidate memories only when useful.
4. Avoid direct durable promotion except for low-risk explicit user preferences or facts.

### 7.2 Promotion gate

A candidate must answer:

1. Will this likely matter in 7+ days?
2. Is it stable, or should it expire?
3. What type is it: preference, fact, project state, episode, procedure, environment, open loop?
4. Does it duplicate an existing memory?
5. Does it contradict an existing memory?
6. Is there a source reference?
7. Should it be core, semantic, episodic, archive-only, or skill?

### 7.3 Gate outcomes

- `discard`
- `archive_only`
- `episodic_only`
- `semantic_fact`
- `project_update`
- `core_update`
- `skill_candidate`
- `open_loop`
- `supersede_existing`

### 7.4 Supersession

When a new claim conflicts with an existing active claim about the same subject/predicate:

- do not keep both as equally active;
- set old record `status: superseded` if the new claim clearly replaces it;
- set both `status: uncertain` if unresolved;
- preserve source refs for both.

### 7.5 Expiry

Temporary facts should use `expires_at` or `valid_until`.

Examples:
- “today’s task” expires soon;
- “current sprint goal” may expire in weeks;
- stable preferences usually do not expire.

---

## 8. Offline consolidation policy

### Session-end consolidation

- Write session episode summary.
- Update working memory archive.
- Extract candidates.
- Link obvious entities.

### Daily consolidation

- Promote/reject pending candidates.
- Update active project cards.
- Detect contradictions.
- Expire stale memories.
- Rebuild FTS indexes for changed files.

### Weekly consolidation

- Merge duplicates.
- Rewrite stale project summaries from evidence.
- Archive inactive projects.
- Run memory benchmark.
- Produce retrieval quality report.

---

## 9. SQLite schema v0

```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  subject TEXT,
  predicate TEXT,
  value TEXT,
  body TEXT,
  summary TEXT,
  status TEXT DEFAULT 'active',
  confidence REAL DEFAULT 0.7,
  importance REAL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  valid_from TEXT,
  valid_until TEXT,
  expires_at TEXT,
  source_refs TEXT,
  tags TEXT,
  file_path TEXT
);

CREATE VIRTUAL TABLE memories_fts USING fts5(
  id UNINDEXED,
  subject,
  predicate,
  value,
  body,
  summary,
  tags
);

CREATE TABLE entities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT,
  aliases TEXT,
  created_at TEXT,
  updated_at TEXT
);

CREATE TABLE edges (
  source_id TEXT,
  relation TEXT,
  target_id TEXT,
  confidence REAL DEFAULT 0.7,
  created_at TEXT,
  updated_at TEXT,
  valid_from TEXT,
  valid_until TEXT,
  source_refs TEXT,
  PRIMARY KEY (source_id, relation, target_id)
);

CREATE TABLE memory_chunks (
  chunk_id TEXT PRIMARY KEY,
  memory_id TEXT,
  chunk_text TEXT,
  chunk_kind TEXT,
  token_count INTEGER,
  embedding_id TEXT,
  created_at TEXT
);

CREATE TABLE retrieval_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query TEXT,
  route TEXT,
  retrieved_ids TEXT,
  accepted_ids TEXT,
  rejected_ids TEXT,
  token_budget INTEGER,
  created_at TEXT
);
```

---

## 10. Hermes integration contract

Implement Memory v2 as an external memory provider first.

Suggested path:

```text
plugins/memory/memory_v2/__init__.py
plugins/memory/memory_v2/store.py
plugins/memory/memory_v2/index.py
plugins/memory/memory_v2/schemas.py
plugins/memory/memory_v2/router.py
plugins/memory/memory_v2/retrieval.py
plugins/memory/memory_v2/compose.py
plugins/memory/memory_v2/consolidate.py
```

### Provider methods

Implement:

- `name = "memory_v2"`
- `is_available()`
- `initialize(session_id, hermes_home, platform=None, **kwargs)`
- `system_prompt_block()` for stable small core memory only
- `prefetch(query, session_id="")` for dynamic routed recall
- `queue_prefetch(query, session_id="")` if useful later
- `sync_turn(user_content, assistant_content, session_id="")`
- `on_session_end(messages)`
- `on_session_switch(new_session_id, parent_session_id="", **kwargs)`
- `on_memory_write(action, target, content, metadata=None)`
- optional `get_tool_schemas()` / `handle_tool_call()`

### Existing Hermes files to respect

- `agent/memory_provider.py`
- `agent/memory_manager.py`
- `agent/conversation_loop.py`
- `run_agent.py`
- `tools/memory_tool.py`
- `agent/system_prompt.py`
- `hermes_state.py`
- `hermes_constants.py`

### Known implementation caveat

`MemoryProvider.on_pre_compress()` is documented as returning text to include in compression, but current compression code appears to call it and discard the return value. Do not rely on that returned text unless the compression path is fixed and tested.

---

## 11. Provider tools v0

Keep the provider tool surface small.

### `memory_v2_search`

Manual search/debugging.

```json
{
  "query": "qwen reasoning loop",
  "types": ["project_state", "episode", "fact"],
  "limit": 10,
  "include_sources": true
}
```

### `memory_v2_write_candidate`

Create candidate memory, not direct durable memory.

```json
{
  "type": "project_state",
  "content": "...",
  "source_ref": "...",
  "importance": 0.8
}
```

### `memory_v2_promote`

Promote a candidate after gate review.

```json
{
  "candidate_id": "cand_...",
  "destination": "semantic/projects/hermes-memory-v2.yaml"
}
```

### `memory_v2_status`

Show health, counts, pending candidates, last consolidation, and index status.

---

## 12. Evaluation contract

The benchmark should test whether memory improves reliability, not vibes.

Required categories:

1. **Preference recall:** remembers stable user preferences.
2. **Project continuity:** answers where active projects left off.
3. **Stale fact handling:** active facts supersede old facts correctly.
4. **Exact source recall:** can identify source refs for claims.
5. **Irrelevant memory suppression:** avoids injecting unrelated private/project memory.
6. **Contradiction detection:** flags conflicts or scope differences.
7. **Procedure routing:** uses skills/procedure refs rather than semantic facts for workflows.
8. **Token budget discipline:** memory packet remains below route budget.
9. **Archive fallback:** can retrieve raw evidence when semantic summary is insufficient.
10. **Profile isolation:** no cross-profile memory leakage.

Initial fixture path:

```text
tests/fixtures/memory_v2_benchmark.yaml
```

Future tests should load the fixture and validate route, retrieval ids, source requirements, and exclusion rules.

---

## 13. Implementation phases

### Phase 1: Spec, fixture, and empty provider skeleton

- Create this spec.
- Create benchmark fixture.
- Add provider skeleton with no-op methods.
- Add tests for initialization paths and profile scoping.

### Phase 2: File store and SQLite FTS

- Create folder structure.
- Append raw events.
- Read/write memory cards.
- Index cards into SQLite FTS.
- Search by keyword.

### Phase 3: Rule-based router and packet composer

- Implement query classification.
- Retrieve candidate records.
- Compose bounded memory packets.
- Log retrieval decisions.

### Phase 4: Online ingestion

- Implement `sync_turn()`.
- Update working memory.
- Create candidate memories.
- Avoid auto-promotion except explicit safe cases.

### Phase 5: Consolidation

- Session-end summaries.
- Daily promotion/rejection.
- Supersession and expiry.
- Project card updates.

### Phase 6: Optional embeddings and graph expansion

- Add incremental embeddings only for changed records.
- Add lightweight entity/edge graph expansion.
- Keep both optional and rebuildable.

### Phase 7: Eval harness and regression reports

- Run fixture-backed retrieval tests.
- Add stale-fact and irrelevant-retrieval regressions.
- Track retrieval logs and failures.

---

## 14. Acceptance criteria for v0

Memory v2 v0 is successful when:

- it initializes under a temporary `HERMES_HOME` without touching the real profile;
- it creates the expected profile-scoped directory tree;
- it writes raw events and candidate memories;
- it indexes and retrieves memory records with SQLite FTS;
- `prefetch()` returns a bounded packet with route/confidence/source refs;
- `sync_turn()` does not promote junk into durable memory;
- stale fact tests demonstrate active/superseded behavior;
- irrelevant-memory tests demonstrate suppression;
- all tests pass without network access.

---

## 15. Explicit non-goals for v0

- No full GraphRAG.
- No mandatory vector database.
- No embedding of every raw message.
- No replacement of built-in `memory` tool yet.
- No cross-profile memory sharing.
- No automatic mutation of core user memory without a gate.
- No large prompt injection of the whole memory store.

---

## 16. First implementation task after this spec

Create the provider skeleton and tests:

```text
plugins/memory/memory_v2/__init__.py
tests/memory_v2/test_memory_v2_provider.py
```

The first tests should verify:

1. provider is available;
2. initialization creates only temp-profile paths;
3. `system_prompt_block()` is small and stable;
4. `prefetch()` returns empty/low-confidence context when no memories exist;
5. no network or external services are required.
