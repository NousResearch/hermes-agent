# BUILD BRIEF: rank-2 extraction — hermes_state.py fork PURE helpers → hermes_state_ext.py

Worktree branch `feat/fork-ext-r2-state` (off fork/main). Rank-2 of
`docs/sync/2026-07-16-fork-mergeability-refactor-SPEC.md` (read FULLY; ritual IN ORDER).
REUSE `scripts/refactor_equiv/`; extend mutate.py's registry.

## Target: hermes_state.py (8,903 lines; +1,720 fork lines; 24 conflict hunks on 07-15)
DB-layer file: most fork additions are SQL-executing methods — those STAY (moving live SQL
call paths is not a pure move). Extract the PURE fork-only surface:
- `_sql_placeholders` (pure)
- `_session_list_denorm_enabled` (pure config read — lazy import stays inside the function)
- `_PLATFORM_SEARCH_ALIASES` dict + any pure query-normalization/token-classification helpers
  that consume it (grep for consumers — the tokenized session-search normalizer; keep the
  desktop-parity comment pointing at session-source.ts)
- `workspace_key` is UPSTREAM code (check merge-base) — do NOT move it
- any other pure fork helpers you find by walking the fork diff:
  git diff e0240d7bf..HEAD -- hermes_state.py | grep '^+'
New module: top-level `hermes_state_ext.py` (hermes_state.py is top-level; a fork_ext/ package
at repo root is fine too if cleaner — your call, document it).
NOTE hermes_undo.py already proves this pattern at the state layer.

Same strictness: no SQL-executing functions, no connection handling, no WAL/checkpoint logic,
no methods (only module-level functions). If the pure surface is <30 lines total, write
docs/sync/review/r2-verdict.md (honest no-go) instead of forcing it.

## Ritual (spec order)
1. Golden-capture untouched → tests/golden/state_ext/.
2. Pure move; 1-line call sites.
3. Golden-replay byte-identical.
4. ≥3 mutations; all RED; revert.
5. Targeted suites: grep tests/ for moved symbols (session-search/platform-alias tests,
   denorm-gate tests) + tests/test_hermes_state*.py targeted files.
6. fork-features.json add/migrate; lint-manifest (--vacuous-ok ok pre-landing).
7. NET budget = 10 + 2×call_sites.
8. Import audit: the new module imports NOTHING from hermes_state (one-way).

## Constraints
venv pytest targeted files green (-q -o addopts="" -p no:randomly), py_compile, commit
per-step, DO NOT push/PR, STOP (or the no-go verdict).
