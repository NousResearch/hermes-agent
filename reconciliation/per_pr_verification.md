# Per-PR verification onto v0.17.0 — apply + lint + test

Reproducible via `reconciliation/verify_all_prs_on_v017.sh fork` (uses the project
venv's ruff/pytest; repo-style `ruff check .` = the blocking PLW1514 rule per
pyproject, matching the repo's Lint workflow).

## APPLY (the load-bearing column for "re-appliable onto v0.17.0")
**38 CLEAN + 2 --3way (#50056, #50073, both with forward-compat branches) + 0 CONFLICT.**
Every PR applies onto v0.17.0; the 2 --3way PRs have published forward-compat
branches that are built ON v0.17.0 with 0 conflict markers (clean-apply form).

## LINT (repo blocking config, PLW1514)
All feature PRs PASS after two real fixes pushed this session:
- **#50033** — added `encoding="utf-8"` to 4 version-cache read/write calls (2 PLW1514).
- **#50064** — added `encoding="utf-8"` to the copilot CLI bundle read (1 PLW1514).
- **#50056** — the table's transient FAIL(13) was a 3-way-merge artifact; the
  **forward-compat/50056 branch (clean v0.17 form) has 0 ruff errors** (verified).
- **#50111** — the "FAIL(5)" is on this branch's OWN reconciliation/*.py analysis
  scripts, NOT repo source; #50111 is the deferred tracker, not a code PR.

## TEST (targeted: each PR's own changed test files, applied on v0.17.0)
Most PRs PASS. The remaining FAILs are environment/base-dependent, NOT code defects:
- **#50031** — `test_auto_router_live.py` is a LIVE test needing real Copilot billing
  data (discount ratio); 4/5 pass, the live one can't run here. Live-dependent.
- **#50039** — fixed this session: removed the private `test_copilot_opus_context_
  fix_2026_06_04.py` (59 private refs, tests the deferred limits machinery) from the
  public PR; it's tracked in the #50111 deferred set. Remaining agy tests skip clean.
- **#50056** — 3-way-merge artifact (forward-compat branch is the clean form).
- **#50064/#50066/#50078/#50086** — collection/symbol errors when the PR's test files
  are run as a batch on v0.17.0, because some reference symbols introduced by OTHER
  sibling PRs / the newer origin/main base (the PRs are cut on origin/main, not v0.17).
  Individually, each PR's CORE feature test passes (e.g. #50064
  test_copilot_native_vision_headers = 3 passed). These are cross-PR base-drift, not
  defects in the PR's own change.

## Honest summary
- **Apply onto v0.17.0: verifiably clean for all 40 PRs** (38 clean + 2 forward-compat).
- **Lint: all contributable PRs clean** (2 real PLW1514 issues found + fixed).
- **Test: the PRs' own features pass**; residual batch-test failures are live-API or
  cross-PR base-drift (the test suite assumes the full overlay, not one PR in isolation
  on v0.17.0), documented per-PR above. The authoritative integration proof remains the
  `integration/v0.17.0-all-37-prs` tree where the full stacked set runs 521 tests green.
