# Persistent Memory V2 Implementation Plan

> For Hermes: use subagent-driven-development to implement this plan task-by-task.

Goal: replace the current flat MEMORY.md/USER.md-only design with a typed persistent memory layer that reliably remembers user preferences, corrections, instructions, and durable context across sessions while preserving the existing memory tool surface.

Architecture: keep the current memory tool API (`memory add/replace/remove`) so behavior stays stable, but move storage to a new SQLite-backed memory store with typed records, tombstones, scoring, and retrieval. At session start, inject a compact top-k memory packet instead of the full flat files. Keep flat markdown export/import as a compatibility layer and fallback.

Tech Stack: Python 3, sqlite3, existing Hermes SessionDB patterns in `hermes_state.py`, existing memory tool in `tools/memory_tool.py`, existing session recall in `tools/session_search_tool.py`.

---

## Design constraints

1. Do not break the existing `memory` tool contract.
2. Do not rely on remote memory services for the core feature.
3. Preserve current security scanning for prompt-injected memory content.
4. Support explicit forgetting and correction without ghost facts reappearing.
5. Keep startup prompt small: inject only high-value retrieved memory, not the entire database.
6. Preserve compatibility with existing `~/.hermes/memories/MEMORY.md` and `USER.md`.

---

## Data model

Create a new local database: `~/.hermes/memory.db`

Table: `memory_entries`
- `id TEXT PRIMARY KEY`
- `target TEXT NOT NULL` (`user` | `memory`)
- `kind TEXT NOT NULL` (`preference`, `instruction`, `identity`, `environment`, `project`, `constraint`, `workflow`, `lesson`)
- `content TEXT NOT NULL`
- `status TEXT NOT NULL` (`active`, `superseded`, `forgotten`)
- `scope TEXT NOT NULL DEFAULT 'global'` (`global`, `project`, `platform`)
- `scope_value TEXT`
- `source TEXT NOT NULL DEFAULT 'manual'` (`manual`, `flush`, `migration`)
- `confidence REAL NOT NULL DEFAULT 1.0`
- `importance REAL NOT NULL DEFAULT 0.5`
- `created_at REAL NOT NULL`
- `updated_at REAL NOT NULL`
- `last_used_at REAL`
- `use_count INTEGER NOT NULL DEFAULT 0`
- `supersedes_id TEXT`
- `fingerprint TEXT NOT NULL`

Indexes:
- `(target, status)`
- `(kind, status)`
- `(scope, scope_value, status)`
- `(updated_at DESC)`
- unique partial-like dedupe via `fingerprint`

Optional table: `memory_events`
- append-only audit trail for add/replace/remove/use

Optional FTS table: `memory_entries_fts`
- searchable content for retrieval and edit matching

---

## Retrieval policy

At session start, build a compact injected memory packet from:
1. active `user` facts with highest importance
2. active project-scoped facts matching current cwd/repo when detectable
3. active global constraints/instructions
4. a small number of recent/high-use lessons

Retrieval score should combine:
- explicit importance
- target bias (`user` > `memory`)
- kind bias (`instruction`/`preference`/`constraint` > others)
- recency
- use_count
- scope match bonus

Hard caps:
- injected `user` packet target: ~900 chars
- injected `memory` packet target: ~1400 chars
- reserve some budget for future growth

Important: retrieval must respect tombstones. If an entry is forgotten or superseded, it must never outrank its replacement or reappear in injected memory.

---

## Compatibility strategy

Phase 1 keeps `MEMORY.md` and `USER.md` as generated exports from the new DB, not the canonical source.

Rules:
- reads at startup prefer DB if present
- if DB missing, bootstrap from markdown files
- after any DB mutation, regenerate markdown exports
- existing docs/tooling that inspect markdown continue to work

---

## Files to create or modify

Create:
- `tools/persistent_memory_store.py`
- `tests/tools/test_persistent_memory_store.py`
- `tests/tools/test_memory_retrieval_policy.py`
- `tests/tools/test_memory_markdown_compat.py`
- `plans/persistent-memory-v2.md`

Modify:
- `tools/memory_tool.py`
- `run_agent.py`
- `website/docs/user-guide/features/memory.md`
- optionally `hermes_cli/doctor.py` to report DB + export health

---

## Task 1: Add the new persistent store skeleton

Objective: create a SQLite-backed memory store with schema init and CRUD primitives.

Files:
- Create: `tools/persistent_memory_store.py`
- Test: `tests/tools/test_persistent_memory_store.py`

Step 1: Write failing tests for init and roundtrip
- verify database file creation
- verify add/get/list/remove primitives
- verify status transitions (`active` -> `forgotten`)

Step 2: Run tests to verify failure
Run: `pytest -q tests/tools/test_persistent_memory_store.py`
Expected: FAIL — module/file missing

Step 3: Implement minimal store
Include:
- schema initialization
- `add_entry()`
- `replace_entry()`
- `forget_entry()`
- `list_entries()`
- `find_by_substring()`
- `export_markdown(target)`

Step 4: Run tests to verify pass
Run: `pytest -q tests/tools/test_persistent_memory_store.py`
Expected: PASS

Step 5: Commit
`git add tools/persistent_memory_store.py tests/tools/test_persistent_memory_store.py && git commit -m "feat: add sqlite-backed persistent memory store"`

---

## Task 2: Add fingerprint dedupe and supersede semantics

Objective: stop duplicates and make corrections actually replace old truth.

Files:
- Modify: `tools/persistent_memory_store.py`
- Modify: `tests/tools/test_persistent_memory_store.py`

Step 1: Write failing tests
- exact duplicate add should no-op
- replace should mark old row `superseded`
- forgotten rows should stay excluded from active listings

Step 2: Run targeted tests
Run: `pytest -q tests/tools/test_persistent_memory_store.py -k "duplicate or supersede or forgotten"`
Expected: FAIL

Step 3: Implement
- normalize content for fingerprints
- add `supersedes_id`
- filter active rows by status

Step 4: Re-run tests
Expected: PASS

Step 5: Commit
`git add tools/persistent_memory_store.py tests/tools/test_persistent_memory_store.py && git commit -m "feat: add dedupe and supersede semantics to memory"`

---

## Task 3: Add retrieval scoring and compact prompt rendering

Objective: inject the right memories, not the whole landfill.

Files:
- Modify: `tools/persistent_memory_store.py`
- Create: `tests/tools/test_memory_retrieval_policy.py`

Step 1: Write failing tests
- high-importance user preferences outrank generic lessons
- scope-matched project facts outrank unrelated global facts
- forgotten/superseded rows never appear
- rendered block respects char budgets

Step 2: Run tests
Run: `pytest -q tests/tools/test_memory_retrieval_policy.py`
Expected: FAIL

Step 3: Implement
- `retrieve_for_prompt(target, scope_context, char_limit)`
- scoring function
- compact block renderer with same header style Hermes already uses

Step 4: Run tests
Expected: PASS

Step 5: Commit
`git add tools/persistent_memory_store.py tests/tools/test_memory_retrieval_policy.py && git commit -m "feat: add scored memory retrieval for prompt injection"`

---

## Task 4: Wire the `memory` tool to the new store

Objective: keep the public tool stable while swapping the backend.

Files:
- Modify: `tools/memory_tool.py`
- Modify: `tests/tools/test_memory_tool.py`
- Create: `tests/tools/test_memory_markdown_compat.py`

Step 1: Write failing tests
- existing tool API still returns same success/error shapes
- add/replace/remove now update DB and regenerated markdown
- startup migration imports legacy markdown into DB once

Step 2: Run tests
Run: `pytest -q tests/tools/test_memory_tool.py tests/tools/test_memory_markdown_compat.py`
Expected: FAIL

Step 3: Implement
- instantiate new store inside `memory_tool.py`
- preserve security scanning
- keep substring matching behavior for user-facing edits
- regenerate `MEMORY.md` / `USER.md` after mutations

Step 4: Run tests
Expected: PASS

Step 5: Commit
`git add tools/memory_tool.py tests/tools/test_memory_tool.py tests/tools/test_memory_markdown_compat.py && git commit -m "refactor: back memory tool with persistent sqlite store"`

---

## Task 5: Replace startup injection to use retrieved top-k memory

Objective: stop loading full flat files and instead load compact relevant memory blocks.

Files:
- Modify: `run_agent.py`
- Modify: `tests/tools/test_memory_tool.py` or create `tests/test_run_agent_memory_injection.py`

Step 1: Write failing tests
- session startup reads from DB-backed store
- injected system prompt contains compact retrieved facts
- project-scoped facts appear only when scope matches

Step 2: Run tests
Run: `pytest -q tests/test_run_agent_memory_injection.py`
Expected: FAIL

Step 3: Implement
- replace or extend `load_from_disk()` path
- detect scope context from cwd/repo/session source where possible
- call `retrieve_for_prompt()` instead of dumping all entries

Step 4: Run tests
Expected: PASS

Step 5: Commit
`git add run_agent.py tests/test_run_agent_memory_injection.py && git commit -m "feat: inject retrieved top-k memory on session start"`

---

## Task 6: Add migration and rollback safety

Objective: avoid corrupting existing users’ memory.

Files:
- Modify: `tools/persistent_memory_store.py`
- Modify: `hermes_cli/doctor.py`
- Modify: `website/docs/user-guide/features/memory.md`

Step 1: Write failing tests
- first boot migrates existing markdown to DB
- migration is idempotent
- doctor reports DB present / export in sync / migration status

Step 2: Run tests
Run: `pytest -q tests/tools/test_memory_markdown_compat.py`
Expected: FAIL

Step 3: Implement
- one-time migration marker
- export regeneration command/helper
- doctor checks for db existence and markdown export freshness

Step 4: Run tests
Expected: PASS

Step 5: Commit
`git add tools/persistent_memory_store.py hermes_cli/doctor.py website/docs/user-guide/features/memory.md tests/tools/test_memory_markdown_compat.py && git commit -m "feat: add memory migration and health checks"`

---

## Task 7: Add memory-use feedback loop

Objective: let the system learn which memories actually matter without inventing magic.

Files:
- Modify: `tools/persistent_memory_store.py`
- Modify: `run_agent.py`
- Test: `tests/tools/test_memory_retrieval_policy.py`

Step 1: Write failing tests
- injected memories can be marked as used
- `use_count` and `last_used_at` alter ranking over time
- low-value stale items fall below stronger repeated preferences

Step 2: Run tests
Run: `pytest -q tests/tools/test_memory_retrieval_policy.py -k "use_count or ranking"`
Expected: FAIL

Step 3: Implement
- include memory ids in retrieval result metadata
- mark used ids after session or after tool-confirmed relevance
- keep this conservative; no blind reinforcement on every turn

Step 4: Run tests
Expected: PASS

Step 5: Commit
`git add tools/persistent_memory_store.py run_agent.py tests/tools/test_memory_retrieval_policy.py && git commit -m "feat: add conservative memory reinforcement loop"`

---

## Task 8: Verification pass in a disposable Hermes home

Objective: prove cross-session behavior end-to-end.

Files:
- no code changes required unless bugs are found

Step 1: Prepare temp home
- copy minimal config/auth
- seed a few memory entries

Step 2: Verify fresh process recall
Commands:
- `HERMES_HOME=/tmp/... hermes chat -Q -q "How short should replies be?"`
- `HERMES_HOME=/tmp/... hermes chat -Q -q "What should happen to secrets?"`
- `HERMES_HOME=/tmp/... hermes chat -Q -q "What site art must remain?"`
Expected: answers reflect stored facts

Step 3: Verify forgetting
- remove one entry
- restart fresh process
Expected: deleted fact does not reappear

Step 4: Verify correction
- replace a fact
- restart fresh process
Expected: old fact does not override corrected fact

Step 5: Commit fixes if needed

---

## Non-goals for V2

Do not add these yet:
- embeddings/vector DB
- automatic speculative memory extraction from every turn
- opaque LLM-written summaries as canonical truth
- cloud dependency for core persistence

If V2 works, those can be V3 experiments.

---

## Acceptance criteria

1. `memory add/replace/remove` remains stable for callers.
2. durable facts survive across fresh Hermes processes.
3. corrected/forgotten facts do not reappear in prompt injection.
4. startup prompt includes compact high-value memory, not a full dump.
5. legacy markdown memory is migrated and still exported for compatibility.
6. tests prove end-to-end persistence, dedupe, correction, forgetting, retrieval ranking, and prompt budget behavior.

---

## Recommended execution order

1. Store + tests
2. Supersede/tombstone semantics
3. Retrieval scoring
4. Tool backend swap
5. Startup injection wiring
6. Migration + doctor
7. Reinforcement loop
8. Disposable-home proof

---

## Decision needed before implementation

Choose one:
- A. Safe local V2 only: SQLite typed memory + retrieval, no Honcho changes
- B. Hybrid V2: local SQLite typed memory as source of truth, optional Honcho mirror later

Recommendation: A first. Build the part we control. Then mirror outward if it still earns a place.
