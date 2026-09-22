# Task 7 report — PR handoff enforcement and Hermes worktree audit

## Outcome

Implemented a fail-closed terminal receipt for dispatcher-managed Git worktrees:

- every worktree completion declares `metadata.repository_changes` as a Boolean;
- `false` is accepted only while the assigned Git checkout is clean;
- `true` requires `commit_sha`, `pushed_branch`, `repository`, `base_branch`, and `pr_url`;
- changed receipts are checked against the live 40-character HEAD, assigned/current branch, a same-head remote-tracking ref, the matching GitHub remote repository, and an ancestor remote base branch;
- rejection happens before `complete_task`, so the card remains in flight and the workspace is preserved;
- scratch tasks and non-worktree directories retain their prior completion behavior.

Worker context and the `kanban_complete` schema now describe the same receipt shape.

## Invariant coverage

Added tool-level behavior tests for:

1. committed/pushed work without a PR URL is rejected and remains running;
2. a clean exact-head pushed branch with a matching PR receipt is accepted;
3. an explicit read-only/no-change receipt is accepted without a PR;
4. a worktree completion with no change declaration is rejected;
5. worker context states both the no-change and repository-change forms.

The tests use a real temporary Git repository and bare remote. No Git behavior is mocked.

## Read-only worktree and Kanban audit

Evidence is in `docs/superpowers/evidence/2026-09-22-hermes-worktree-pr-audit.md`.

- Direct GitHub identity: `mrkillbob`.
- Registered Hermes Agent worktrees audited: 28.
- Dirty worktrees were preserved; no reset, clean, stash, commit, push, or branch rewrite was performed in them.
- Clean historical heads were covered by merged PRs #108–#124 or, for the superseded engineering-memory branch, had the same stable patch-id as the fork-base commit merged by PR #122.
- PR #125 already covers the only current open registered Hermes sync head.
- The exact unassociated heads were active/dirty WIP rather than qualifying completed changes.
- The current Kanban board had one active Hermes worktree card (`t_92e44484`), already blocked with its ephemeral path removed. Recent worktree cards were diagnostic/no-change and cleaned up. Scratch cards that named changed files had no committed Git receipt.
- No qualifying completed change required a new PR. No PR was opened and no Kanban task or receipt was mutated.

## Verification

Red evidence was observed before implementation:

- `test_complete_rejects_repository_change_without_pr_receipt` failed because the task completed instead of returning an error.
- `test_complete_rejects_worktree_without_change_declaration` failed for the same reason.
- the worker-context guidance assertion failed before the guidance changed.

Green focused command:

```text
scripts/run_tests.sh tests/tools/test_kanban_tools.py tests/hermes_cli/test_kanban_core_functionality.py \
  -k 'repository_change_without_pr_receipt or exact_repository_pr_receipt or explicit_no_repository_changes or worktree_without_change_declaration or worker_context_requires_terminal_kanban_receipt'
```

Result: **5 passed, 0 failed**.

Full affected files:

```text
scripts/run_tests.sh tests/tools/test_kanban_tools.py tests/hermes_cli/test_kanban_core_functionality.py
```

Result: **76 passed, 1 failed, 1 skipped**. All 47 tests in `tests/tools/test_kanban_tools.py` passed. The sole failure was the pre-existing, independently reproducible `tests/hermes_cli/test_kanban_core_functionality.py::test_protocol_violation_budget_not_consumed_by_other_failures`: it expected a third protocol violation to block, but the task remained ready. A fresh isolated rerun reproduced that failure. Task 7 does not alter the protocol-violation budget or dispatcher failure handling, so it was recorded rather than expanded into this repair.

Additional checks:

- `git diff --check`: pass.
- `python3 -m py_compile` for every changed Python source/test file: pass.

## Preserved unrelated state

The pre-existing modification to `task-5-report.md` was not edited or staged as part of Task 7.
