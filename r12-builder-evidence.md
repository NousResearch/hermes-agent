# Bootstrap 001A-R12 — Builder candidate evidence

Status: BUILDER COMPLETE / ISOLATED CANDIDATE ONLY / READY FOR REVIEW

## Pre-edit identity and boundary

- Candidate path: `C:\Users\Jon\AppData\Local\hermes\kanban\workspaces\t_9b2d9b41\candidate-v2026.9.24`
- Candidate branch: `feature/bootstrap-001a-r12-v0215`
- Candidate pre-edit HEAD: `f97608f178d1ffeca59860195ab7da295f7c8e5f`
- Candidate pre-edit status: pristine (no tracked or untracked entries)
- R11 report SHA-256: `70c797bfb783f380d599e546ba2e396f71069f7239896443333fb48798ff5b9a` (matches binding value)
- Windows production checkout HEAD before and after builder work: `29112bef099274229cadff79cdff7bf7b99c4b77`
- Windows production checkout status before and after builder work: the same 12 preserved historical modified files named by R11; no candidate changes were applied there.
- Cutover path recheck: absent or not a Git worktree; no cutover action occurred.
- Exclusions held: no production/cutover, deployment, migration, restart, WSL/OpenDesign, Desktop/gateway, Control, 001B, Jev/Beacon, GPT ingestion, profile/config/permission, or Sacred Reach source action.

## R11 intent implementation matrix

| Intent | Binding disposition | Implementation and proof | Result |
|---|---|---|---|
| I1 | RETAIN/ADAPT | Added a focused failing worker regression in `tests/hermes_cli/test_cli_preloaded_skills.py`; narrowed `cli.py::HermesCLI.finalize_preloaded_skills` so dispatched Kanban workers warn and continue when every optional forced skill is unavailable, while interactive/direct callers remain fail-loud. | GREEN: full focused file `3 passed in 3.79s`. |
| I2 | RETAIN/ADAPT | Added Windows native/relative/quoted/Unicode attachment-path coverage in `tests/plugins/test_kanban_attachments.py`; replaced direct POSIX `shlex` use in `hermes_cli/kanban.py::run_slash` with the existing Windows-aware `split_command_line` helper. | GREEN: focused regression `1 passed in 1.28s`. |
| I3 | RETAIN/ADAPT | Added aggregate UTF-8-byte, mandatory identity, whole-record, marker-removal, parent inventory/retrieval, and multibyte regressions in `tests/hermes_cli/test_kanban_db.py`; added a 96 KiB aggregate transport budget in `hermes_cli/kanban_db.py`. Optional records are selected whole, omitted parent IDs remain retrievable, mandatory task identity is UTF-8-safe and bounded, and private structure markers are removed from final output. | GREEN: focused context suite `2 passed in 3.46s` after independent-review hardening. |
| I4 | UPSTREAM-ONLY | No runtime change. Ran the existing Windows worker `Popen` handle/exit classification proof in `tests/hermes_cli/test_kanban_worker_exit_decode.py`. | GREEN: `1 passed in 1.36s`. |
| I5 | RETAIN/ADAPT | Added a focused failing spawn-environment regression; `hermes_cli/kanban_db_dispatch.py::_default_spawn` now strips `HERMES_EPHEMERAL_SYSTEM_PROMPT`, `HERMES_PREFILL_MESSAGES_FILE`, and `HERMES_TUI_SKILLS` after the gateway session-context scrub while preserving worker task/profile/board pins. | GREEN: focused regression `1 passed in 2.19s`. |
| I6 | RETAIN/ADAPT | Extended the legacy retag regression with Windows backslash paths and prefix-lookalike protection; `hermes_state.py::SessionDB.retag_kanban_worker_sessions` now matches slash and backslash exact/descendant forms. | GREEN: focused regression `1 passed in 1.56s`. |
| I7 | TEST-ONLY | Updated moved tests in `tests/tools/test_plugin_skills.py` and `tests/gateway/test_profile_isolation_runtime.py` to use host-relative/`Path` semantics. No runtime source change. | GREEN: focused plugin-skill set `3 passed in 3.33s`; covered again by bounded suite. |
| I8 | UPSTREAM-ONLY + TEST | No runtime change. Added an exact Unicode round-trip regression in `tests/tools/test_browser_use_cli.py`, proving the existing explicit UTF-8 subprocess configuration. | GREEN: focused proof `1 passed in 2.06s`. |

## Combined bounded regression and quality gates

- Combined bounded suite across all eight intent surfaces: `137 passed, 32 skipped in 22.18s`.
- Ruff across all touched production and test files: `All checks passed!`.
- Python bytecode compile check for every touched production module: passed.
- `git diff --check`: passed with no whitespace errors.
- Added-line security scan: no hardcoded credential assignments, `shell=True`, `os.system`, unsafe pickle loads, or dynamic SQL construction detected.
- Independent pre-commit diff review: `passed=true`, `security_concerns=[]`, `logic_errors=[]`.
- Reviewer hardening suggestion addressed: the worker-context regression now explicitly proves the private NUL structure marker never reaches generated context.
- Reviewer performance suggestion not applied: incremental accounting would be a non-required optimization; the current correctness-first selection is bounded by the 96 KiB aggregate ceiling and changing it would expand R12 scope.

## Broader-suite disclosure

A non-gating exploratory run of the complete `tests/hermes_cli/test_kanban_db.py` file reported three failures outside the R12 intent paths:

1. `test_infrastructure_spawn_refusal_never_charges_the_card` — independently reproducible in a fresh process.
2. `test_worktree_workspace_explicit_target_materializes_linked_worktree` — independently reproducible in a fresh process; Windows Git returned `/c/...` while the existing assertion expected `C:/...`.
3. `test_dispatch_max_in_progress_blocks_review_when_at_limit` — passed when rerun alone (`1 passed in 2.28s`), confirming order-dependent pre-existing fake-process registry contamination.

R12 did not modify the first two code paths or the third test's accounting logic. They are disclosed for the Reviewer/QA lanes rather than expanded into this binding scope.

## Final diff and workspace audit

- Code-and-tests diff SHA-256, excluding this evidence file: `4b0bc1f19f9a9b056e7b7b2bc08a97fce641bb3a3b643769f63ca70c96a1cd59`.
- Code-and-tests diff stat: 12 files changed, 518 insertions, 34 deletions.
- Runtime files changed: `cli.py`, `hermes_cli/kanban.py`, `hermes_cli/kanban_db.py`, `hermes_cli/kanban_db_dispatch.py`, `hermes_state.py`.
- Test files changed: `tests/gateway/test_profile_isolation_runtime.py`, `tests/hermes_cli/test_cli_preloaded_skills.py`, `tests/hermes_cli/test_kanban_db.py`, `tests/hermes_cli/test_kanban_worker_session_source.py`, `tests/plugins/test_kanban_attachments.py`, `tests/tools/test_browser_use_cli.py`, `tests/tools/test_plugin_skills.py`.
- New tracked evidence intended: `r12-builder-evidence.md`.
- Untracked audit before commit: only `r12-builder-evidence.md`.
- Ignored local audit: an ignored `.venv/` was materialized by the independent review harness. It is not staged, committed, or part of the candidate diff.
- No dependency was added and no strictness/configuration setting was weakened.

## Handoff boundary

This report records Builder evidence only. No deployment, production mutation, cutover, merge, or approval is implied. The pre-created Reviewer and QA child cards remain the required downstream gates.
