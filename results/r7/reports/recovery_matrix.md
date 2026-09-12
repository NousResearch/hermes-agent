# R7 Recovery Matrix — PRESENT SOURCE > REPORT > MEMORY

Source of truth: `git stash@{0}` (autostash 20260911-023247, parent `f97a4102dd`)
and `git stash@{1}` (autostash 20260910-022254). Verified by extraction to
`/tmp/r7rec0` + `/tmp/r7rec1` and marker grep (no fabrication).

## Component status

| component | status | source/ref | confidence |
|---|---|---|---|
| R1 cognitive layer (taxonomy/cognition/cognitive_retrieval/maintenance/migration + test) | EXACT | stash@{1} (9 files) | high — extracted, byte recovery |
| R2 retrieval hardening + safety.py + tests | EXACT | stash@{0} (store/retrieval/safety + test_r2_hardening) | high |
| R3 storage/retrieval/semantic + tests | EXACT | stash@{0} (incl. semantic_brain.py + 4 test files) | high |
| R4 temporal + tests | EXACT | stash@{0} (LIFECYCLE_*, fact_lineage, _atomic + 4 test files) | high |
| R5 hardening + tests | EXACT | stash@{0} (mark_stale guard, _would_cycle + 5 test files) | high |
| R6 operations + tests | EXACT | working tree (never lost) | high |
| results/r1–r5 artifacts | EXACT | stash@{0}^3 (untracked) | high |
| results/r6 artifacts | EXACT | working tree | high |

Nothing is MISSING. Nothing is RECONSTRUCTED (unless integration forces it —
any such case will be marked explicitly below).

## Upstream drift since stash parent (`f97a4102dd` → `45a6101f`)

- `plugins/memory/holographic/*`: ZERO upstream changes.
- `agent/memory_provider.py`: +`turn_author` optional kwarg on `sync_turn`;
  manager sends only signature-accepted kwargs (backward compatible).
- `agent/memory_manager.py`: signature-filtered `on_turn_start`/`sync_turn`
  dispatch (backward compatible).
- Abstract methods: 4 before, 4 after — no new provider obligations.

Conclusion: exact restore is safe; no reconcile edits expected. Any deviation
found by tests will be recorded here, not silently patched.

## Restore log

- Tracked R-scope files ← `git checkout stash@{0} -- <7 plugin files>`
- New R-scope files ← `git checkout 'stash@{0}^3' -- <safety, semantic_brain, 17 tests, results/r1-r5>`
- Early cognitive files ← `git checkout 'stash@{1}^3' -- <5 modules + 1 test>`
- Untouched: `operations.py`, `OPERATIONS.md`, `test_r6_ops.py`, `results/r6/`
  (post-date R6 work, absent from both stashes — verified by file lists).
## Supersession record (found during integration, not assumed)

`tests/plugins/memory/test_holographic_cognitive.py` (from stash@{1}) calls a
pre-R1 API surface that the recovered R-line deliberately replaced:

| old API (stash@{1} only) | superseded by (stash@{0} / tree) | intent covered in |
|---|---|---|
| `MemoryStore.get_fact` | lifecycle columns + `list_facts` stale flag | test_r4_temporal, test_r7_integration |
| `MemoryStore.lineage` | `fact_lineage` table + `supersede_fact` chain | test_r4_temporal, test_r5_invariants |
| `FactRetriever.cognitive_search` | hardened `search` (R2/R3) | test_r2_hardening, test_r3_retrieval |
| `HolographicMemoryProvider.run_maintenance` | `operations.run_maintenance` (R6) | test_r6_ops, test_r7_integration |
| `HolographicMemoryProvider.llm_calls` / `last_diagnostics` | per-report `llm_calls: 0` fields + `operations.get_metrics` | test_r6_ops, test_r7_integration |

16 of the file's 50 tests fail with `AttributeError` on the integrated stack —
expected, not a regression (the other 34 pass against the recovered early
modules, which strengthens the recovery story). The file is kept byte-identical (exact recovery
preserved); it is excluded from the merge gate with this documented reason.
No compat shims were added (smallest safe solution; no dead production code
for a superseded surface).

- Stashes themselves unmodified (restore used `git show` + `cp` from
  `/tmp/r7rec0` + `/tmp/r7rec1` tar extractions; only reads).
