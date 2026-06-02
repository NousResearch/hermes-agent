# Holographic memory vs session_search evaluation

Task: t_941e01ed
Profile: researcher_hermes_maintenance
Date: 2026-06-01

## Executive recommendation

Treat `session_search`/`state.db` transcripts as the recall ground truth and add an automated regression harness that samples transcript-derived facts into a temporary holographic memory database, then compares `session_search` discovery against `fact_store` search/probe/reason results.

The immediate reliability risk is not HRR math alone. The highest-confidence root causes are:

1. Profile/db-path ambiguity: the holographic plugin is configured with `db_path: ~/.hermes/memory_store.db`, and `MemoryStore` expands `~` against process `HOME`, not `HERMES_HOME`. In this worker, `HERMES_HOME` is `.../profiles/researcher_hermes_maintenance`, but the active `fact_store` tool exposed only two facts from an apparent coordinator-home-scoped database. Relevant facts exist in `/home/engs2272/.hermes/memory_store.db` but were not visible to this worker's tool.
2. Ingestion coverage gaps: `session_search` can find assistant/tool transcript decisions that were never converted into facts. Holographic `auto_extract` only scans user messages with narrow preference/decision regexes, while the required ground-truth items often live in assistant summaries, tool outputs, diffs, Kanban comments, or task handoffs.
3. Query/retrieval fragility: `fact_store search` is FTS5-candidate gated with AND semantics. Multi-term operational queries such as `Matrix E2EE device session recovery` or `Kanban comments blocked workers executable dispatch` miss facts unless their tokens are carefully chosen. HRR `probe`/`reason` can return unrelated facts when the store is tiny, producing false positives.
4. Built-in MEMORY/USER is too small and curated for this use case. It can hold a few high-signal policy facts, but cannot be the source of truth for operational decisions like aggregate lineage counts.

## Evidence-first reproduction commands

All commands below are read-only except for creation of this report.

### Session-search ground truth across profile-scoped state DBs

```bash
cd /home/engs2272/.worktrees/t_941e01ed
python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from hermes_state import SessionDB
from tools.session_search_tool import session_search
profiles = {
  'coordinator': '/home/engs2272/.hermes/profiles/coordinator_hermes_maintenance/state.db',
  'responsible_kanban_semantics': '/home/engs2272/.hermes/profiles/responsible_kanban_semantics/state.db',
  'responsible_memory_rollout': '/home/engs2272/.hermes/profiles/responsible_memory_rollout/state.db',
  'pa_yunuen': '/home/engs2272/.hermes/profiles/pa_yunuen/state.db',
}
queries = [
  'lineage status',
  'working waiting',
  'blocked waiting dormant',
  'Matrix E2EE device session recovery',
  'Kanban comments blocked workers executable dispatch',
]
for name, path in profiles.items():
    db = SessionDB(Path(path))
    print('\n##', name)
    for q in queries:
        data = json.loads(session_search(query=q, limit=3, sort='newest', role_filter='user,assistant,tool', db=db))
        print(q, '=>', [(r.get('session_id'), r.get('match_message_id')) for r in data.get('results', [])])
PY
```

### Holographic/fact-store comparison without mutating memory

```bash
cd /home/engs2272/.worktrees/t_941e01ed
python - <<'PY'
import sys
sys.path.insert(0, '.')
from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever
queries = [
  'Lineage status working waiting blocked dormant',
  'working 2 waiting 1 aggregate lineage counts',
  'Matrix E2EE device session recovery',
  'Kanban comments blocked workers executable dispatch',
  'blocked waiting dormant semantics',
]
for label, dbp in [
  ('root', '/home/engs2272/.hermes/memory_store.db'),
  ('active_tool_apparent', '/home/engs2272/.hermes/profiles/coordinator_hermes_maintenance/home/.hermes/memory_store.db'),
  ('researcher_nested', '/home/engs2272/.hermes/profiles/researcher_hermes_maintenance/home/.hermes/memory_store.db'),
]:
    store = MemoryStore(db_path=dbp)
    retriever = FactRetriever(store)
    print('\nDB', label, 'facts', len(store.list_facts(limit=10000)))
    for q in queries:
        res = retriever.search(q, min_trust=0, limit=5)
        print(q, '=>', [(r['fact_id'], round(r.get('score', 0), 3)) for r in res])
    store.close()
PY
```

## Evaluation set and observed results

Legend:
- SS = `session_search` ground truth.
- FS(active) = the `fact_store` tool available to this worker.
- FS(root) = direct read-only query of `/home/engs2272/.hermes/memory_store.db`.
- Built-in = injected/flat MEMORY/USER for `researcher_hermes_maintenance`.

| Ground truth item | SS evidence | FS(active) | FS(root) | Built-in MEMORY/USER | Diagnosis |
|---|---|---:|---:|---:|---|
| Generic `Self status:` + `Lineage status:` labels replace role-specific labels and old `Blocker status` wording | coordinator state DB session `20260601_163207_b8b43e`, message 3228: task was created because `session_search` found the prior decision and fact_store did not. Also responsible_memory_rollout session `20260601_155550_e701ff`, message 1164 shows facts 134-137 in root memory_store. | Miss. `fact_store search Lineage status...` returned 0 in this worker. `probe`/`reason` returned only two unrelated rollout marker facts. | Hit for narrower queries. `Lineage status` returns facts 125, 134, 135, 136, 137; `Lineage status working waiting blocked dormant` returns 137/135/136. | Miss in researcher profile. No status-label entries in current MEMORY/USER. | Primary miss is profile/db-path visibility; secondary issue is query specificity. |
| Deterministic aggregate lineage counts, e.g. `working: 2 | waiting: 1 | blocked: 0 | dormant: 3` | responsible_kanban_semantics session `20260601_163311_dce124`, message 1032: implementation summary says deterministic aggregate Lineage status rendering, counts order `working | waiting | blocked | dormant`, sample output `Lineage status: WORKING — working: 2 | waiting: 1 | blocked: 0 | dormant: 3`. | Miss. | Miss. Root has label facts but not aggregate-count semantics/details. | Miss. | Ingestion gap: detailed assistant/task handoff was not saved as a fact. This is the concrete failure Yunuen reported. |
| `blocked` vs `waiting` vs `dormant` semantics | responsible_kanban_semantics session `20260601_163311_dce124`, message 1032: headline precedence and compatibility aliases; root memory fact 123 says blocked requires human intervention, dormant/on-hold means no specific work is prevented. | Miss. | Partial hit: query `blocked waiting dormant semantics` returns fact 123. | Miss. | Partial ingestion exists in root store but is invisible to this worker and lacks the aggregate renderer details. |
| Matrix E2EE device/session recovery observations | pa_yunuen session `20260601_090858_cfe5068c`, message 2020: canonical workflow says no-session/Megolm/Olm warnings mean routing may be correct but E2EE room-key sharing is broken; verify/trust bot device and import recovery key where available. Root facts 60, 61, 80 record key sharing and recovery-key/device self-signing observations. | Miss. | Partial hit only with narrow queries: `Matrix E2EE`, `room key sharing`, or `device recovery`. Multi-term query `Matrix E2EE device session recovery` returned 0. | Partial: researcher MEMORY has project Matrix E2EE policy, but not device/session recovery observations. | Query fragility plus incomplete normalized tags/entities. |
| Kanban comments on blocked workers are not executable dispatch | responsible_kanban_semantics state DB session `20260601_111852_c11788`, message 495; session `20260601_155550_b58160`, message 791: `Comments are durable context, not executable dispatch requests. A comment added after a task has blocked does not wake the assignee; only a new ready task or explicit unblock/re-dispatch enters the dispatcher queue.` Coordinator session `20260601_163207_b8b43e`, message 3226 added this to flat memory. | Miss. | Miss. | Miss for researcher profile. The fact was in coordinator MEMORY at message 3226, not in researcher MEMORY. | Cross-profile flat memory and fact-store propagation gap. Also no root fact captured for this policy. |

## Exact active-tool false negatives/false positives

From this worker's `fact_store` tool:

- `fact_store(action='list', min_trust=0)` returned only two facts, both rollout verification markers:
  1. `Hermes Maintenance holographic memory rollout verification marker for task t_96dfd135...`
  2. `Hermes Maintenance holographic rollout profile verification marker: coordinator_hermes_maintenance task t_96dfd135`
- `fact_store(action='search', query='Lineage status working waiting blocked dormant', min_trust=0)` returned 0.
- `fact_store(action='search', query='Matrix E2EE device recovery', min_trust=0)` returned 0.
- `fact_store(action='search', query='Kanban comments blocked workers executable dispatch', min_trust=0)` returned 0.
- `fact_store(action='search', query='working 2 waiting 1 aggregate lineage counts', min_trust=0)` returned 0.
- `fact_store(action='probe', entity='Lineage status')` returned both rollout marker facts with scores around 0.27/0.25. These are false positives: neither contains Lineage status semantics.
- `fact_store(action='reason', entities=['Lineage status', 'working waiting'])` also returned the same two marker facts, another false positive.

## Implementation findings

### Built-in memory

`tools/memory_tool.py` stores curated flat entries in `MEMORY.md` and `USER.md`, with a frozen snapshot injected into the system prompt at session start. Writes are durable immediately but do not update the current system prompt snapshot. Capacity is intentionally small (`memory_char_limit` and `user_char_limit`). This makes built-in memory appropriate for stable, compact policy/preferences, not transcript-grounded recall.

Current researcher built-in memory contains Matrix provisioning policy and user workflow preferences only. It does not contain status-label replacement, aggregate lineage counts, blocked/waiting/dormant semantics, or Kanban-comment dispatch semantics.

### Holographic/fact_store memory

Relevant code:

- `plugins/memory/holographic/__init__.py`
  - `system_prompt_block()` only reports count and instructions; it does not inject facts.
  - `prefetch(query)` runs `retriever.search(query, limit=5)` and injects matching facts only if FTS finds candidates.
  - `sync_turn()` is a no-op.
  - `on_session_end()` calls `_auto_extract_facts()` only when `auto_extract` is truthy.
  - `_auto_extract_facts()` scans user messages only and only narrow patterns like `I prefer`, `we decided`, `the project uses/needs/requires`.
  - `on_memory_write()` mirrors built-in memory adds into fact_store.
- `plugins/memory/holographic/store.py`
  - Facts are stored in SQLite with FTS5, simple regex entity extraction, trust score, and optional HRR vector.
  - Entity extraction mostly captures capitalized multi-word phrases, quoted terms, and AKA patterns. It will not reliably create entities for `Lineage status`, `E2EE`, `t_...`, `working: 2`, or code-like terms unless quoted.
- `plugins/memory/holographic/retrieval.py`
  - `search()` is FTS5-candidate gated. If `facts_fts MATCH ?` returns no candidates, Jaccard/HRR never run.
  - FTS5 uses AND semantics for multi-word queries.
  - `probe()` and `reason()` score all HRR vectors when numpy is available; on very small unrelated stores this can still return top-N facts even when none are semantically relevant.

### session_search

Relevant code:

- `tools/session_search_tool.py`
  - `session_search(query=...)` calls `SessionDB.search_messages()` against `state.db` FTS5.
  - Discovery returns actual message snippets plus anchored windows and bookends; no LLM summarization.
  - It deduplicates by session lineage and skips hidden `source='tool'` sessions, but can include tool-role messages when `role_filter='user,assistant,tool'`.
- `hermes_state.py`
  - `SessionDB` stores messages and sessions in profile-scoped `state.db` with FTS5.
  - `search_messages()` sanitizes FTS5 queries and supports temporal sort.

This makes session_search a stronger ground truth for prior conversation content than memory systems, because it indexes the raw transcript and does not depend on an agent deciding what to remember.

## Root-cause hypotheses, confidence, and validation

1. Profile/db-path mismatch: high confidence.
   - Evidence: config uses `db_path: ~/.hermes/memory_store.db`; holographic `initialize()` replaces `$HERMES_HOME` but leaves `~`; `MemoryStore` then calls `Path(db_path).expanduser()`. The root store has 140 facts, while the active tool visible to this worker has only 2. Relevant facts exist in root memory_store but not in active tool results.
   - Validation: start two Hermes profiles with distinct `HOME` and `HERMES_HOME`; run `fact_store list`; verify whether `~/.hermes/memory_store.db` resolves to OS HOME or profile HERMES_HOME. Change config to `$HERMES_HOME/memory_store.db` in a temp profile and confirm isolation.

2. Ingestion misses assistant/tool/Kanban-derived decisions: high confidence.
   - Evidence: aggregate lineage counts are present in session_search message 1032 but absent from root and active fact stores. Kanban-comment dispatch semantics are present in session_search diff/message 495/791 and coordinator flat-memory write 3226 but absent from active researcher memory and root facts.
   - Validation: in a temp profile, run a short session whose assistant states a decision in final response and tool output, then end the session with `auto_extract=true`; inspect memory_store for extracted facts.

3. Query matching is too brittle: high confidence.
   - Evidence: root store returns Matrix facts for `Matrix E2EE`, `room key sharing`, and `device recovery`, but returns 0 for `Matrix E2EE device session recovery`. `Kanban comments blocked workers executable dispatch` returns 0 despite transcript ground truth.
   - Validation: run query matrix against temp fact corpus with exact, partial, synonym, and code-token queries; compare FTS-only, OR-expanded FTS, tag-expanded search, and fallback Jaccard/embedding/HRR candidate generation.

4. HRR probe/reason lacks a relevance floor: medium-high confidence.
   - Evidence: active store with only two unrelated rollout marker facts returns both facts for `probe('Lineage status')` and `reason(['Lineage status', 'working waiting'])` rather than empty.
   - Validation: seed a temp store with unrelated facts, probe unknown entities, and assert empty results below calibrated score thresholds.

5. Random sampling should be used for evaluation, not runtime recall: medium confidence.
   - Evidence: session_search can discover raw transcript facts across many sessions; random samples would reveal ingestion/ranking regressions that curated examples miss. But random injection into live memory would add noise and stale facts.
   - Validation: build a deterministic seeded sampler over session_search hits, review/redact samples, and run it as an offline CI benchmark.

## Recommended fixes ordered by risk

1. Low risk: change profile configs/default docs to use `$HERMES_HOME/memory_store.db`, not `~/.hermes/memory_store.db`; in plugin initialization, expand `~/.hermes/...` relative to `HERMES_HOME` when running under a profile, or warn loudly when `db_path` resolves outside `HERMES_HOME`.
2. Low risk: add an explicit diagnostic command/log line in `fact_store list` or provider startup showing the resolved memory_store path and fact count. This would have exposed the active-tool/root-store split immediately.
3. Medium risk: add a regression harness that takes a YAML/JSON corpus of `session_search` queries, expected snippets/facts, and expected fact_store hits. Run it against a temp `HERMES_HOME` and temp `memory_store.db` so it never mutates real memory.
4. Medium risk: improve retrieval fallback. If FTS5 returns no candidates, run a safe fallback over recent/high-trust facts using token overlap and/or OR-expanded FTS before returning empty. Also tag facts with normalized aliases (`lineage-status`, `status-semantics`, `matrix-e2ee`, `kanban-dispatch`) during ingestion.
5. Medium risk: add minimum score thresholds for HRR `probe` and `reason`, plus an `empty-if-below-threshold` behavior to prevent false positives.
6. Medium-high risk: broaden ingestion beyond user messages. Candidate sources should include assistant final responses, Kanban completion/block comments, and tool outputs that are already structured handoffs. Require redaction and source metadata; never ingest secrets or raw logs wholesale.
7. Higher risk: automatic session_search-to-memory sampling. Do this only offline or behind explicit review, because raw transcripts can include stale decisions, secrets in tool output, and task-local details that should not become durable memory.

## Proposed automated comparison harness

Add a test/utility like `scripts/evaluate_memory_recall.py`:

Input corpus item:

```yaml
- id: lineage_aggregate_counts
  profile_state_db: /path/to/state.db
  session_query: "working waiting"
  expected_session_snippet: "working: 2 | waiting: 1"
  fact_queries:
    - "Lineage status working waiting blocked dormant"
    - "working 2 waiting 1 aggregate lineage counts"
  expected_fact_substrings:
    - "working: 2 | waiting: 1"
  expected_builtin_substrings: []
```

Harness behavior:

1. Create a temp `HERMES_HOME` and temp memory_store; never point at live profile DBs for writes.
2. For each corpus item, run `session_search` on the specified read-only `state.db` and assert the expected snippet is recoverable.
3. Seed the temp fact store either from known expected facts or from an extractor under test.
4. Run `FactRetriever.search`, `probe`, and `reason` for each query.
5. Classify:
   - true positive: expected fact/snippet returned in top K,
   - false negative: session_search finds it but fact_store does not,
   - false positive: fact_store returns unrelated facts above threshold,
   - profile visibility mismatch: expected fact exists in one DB but active provider resolves another path.
6. Emit JSONL plus a markdown summary with paths, fact IDs, scores, query strings, and sanitized snippets.
7. Include a seeded random sampler from session_search hits. The sampler should select N sessions/queries per profile using a fixed seed and review/redact before adding them to the stable corpus.

## Unknowns

- Whether the active runtime should intentionally share one root holographic store across profiles or keep strict profile-local stores. The code/config currently does both depending on `HOME`, `HERMES_HOME`, and `db_path`.
- Whether uncommitted local changes in the main checkout already partly address memory-write metadata/source tagging. The worktree branch inspected here contains the bridge paths, but live memory facts in root contain source metadata not produced by the legacy `HolographicMemoryProvider.on_memory_write()` signature shown in this worktree.
- The desired retention policy for operational decisions like aggregate lineage counts: they are too detailed for flat MEMORY but useful for regression and task recall.

## Source list

- `/home/engs2272/.worktrees/t_941e01ed/tools/session_search_tool.py` — implementation of transcript search, discovery, scroll, lineage dedupe, and role filtering.
- `/home/engs2272/.worktrees/t_941e01ed/hermes_state.py` — SQLite session store and FTS5 message search implementation.
- `/home/engs2272/.worktrees/t_941e01ed/tools/memory_tool.py` — built-in flat memory persistence, injection snapshot, capacity, and write behavior.
- `/home/engs2272/.worktrees/t_941e01ed/plugins/memory/holographic/__init__.py` — holographic provider initialization, prefetch, auto-extract, fact_store tool handlers, memory-write mirroring.
- `/home/engs2272/.worktrees/t_941e01ed/plugins/memory/holographic/store.py` — fact schema, FTS5 table, entity extraction, trust scores, HRR vector storage.
- `/home/engs2272/.worktrees/t_941e01ed/plugins/memory/holographic/retrieval.py` — FTS/Jaccard/HRR retrieval pipeline, probe/reason behavior.
- Profile session DBs queried read-only: coordinator_hermes_maintenance, responsible_kanban_semantics, responsible_memory_rollout, pa_yunuen, researcher_hermes_maintenance.
- Memory DBs queried read-only: `/home/engs2272/.hermes/memory_store.db`, `/home/engs2272/.hermes/profiles/coordinator_hermes_maintenance/home/.hermes/memory_store.db`, `/home/engs2272/.hermes/profiles/researcher_hermes_maintenance/home/.hermes/memory_store.db`.
