# HQ clean Priority A execution plan — 2026-07-05 19:50:45

## Purpose

Create a clean candidate branch from current `origin/main`, then port only the Priority A bugfix stacks from the divergent local `main` in a controlled order.

This plan follows the user-selected sequence `a>b>c`:

1. Preserve current state.
2. Write this execution plan.
3. Create a clean candidate branch/worktree and start with A4.

## Preservation handles

- Preserved source branch: `main`
- Source HEAD commit: `8b97f5e51`
- Annotated tag: `hq-pre-clean-candidate-main-20260705-195045`
- Backup branch: `hq/backup-main-before-clean-candidate-20260705-195045`
- External artifact backup directory:

```text
C:/Users/82109/AppData/Local/hermes/artifacts/hq-pre-clean-candidate-20260705-195045
```

Backed up reports:

```text
hq_hermes_pre_update_audit_20260705_043838.md
hq_local_ahead_74_commit_classification_20260705.md
hq_local_ahead_74_commit_inventory_20260705.json
hq_priority_a_bugfix_stack_review_20260705.md
```

## Clean candidate target

- Base: current `origin/main`
- Candidate branch:

```text
hq/clean-priority-a-20260705-195045
```

- Candidate worktree:

```text
C:/Users/82109/AppData/Local/hermes/worktrees/hq-clean-priority-a-20260705-195045
```

## Priority A port order

1. **A4 Windows Gateway restart hardening**
   - Source commit: `c1710d52ef7554da4acc20ef9e5544fa572c1ab1`
   - Expected touched paths:

```text
gateway/run.py
tests/gateway/test_restart_drain.py
```

   - Focused verification:

```bash
cd /c/Users/82109/AppData/Local/hermes/worktrees/hq-clean-priority-a-20260705-195045
export PYTHONPATH="$PWD"
python -m pytest tests/gateway/test_restart_drain.py -v --tb=short -n 0
```

2. **A5 Desktop/TUI websocket stall diagnostics**
   - Port only after A4 applies and tests pass or blocker is documented.

3. **A2 Runtime event/result boundary hardening**
   - Port before A1 because clarify callback reliability depends on event/result boundaries.

4. **A1 Clarify approval/choice UX correctness**
   - Port as one coherent final-behavior stack, not as repeated raw duplicate commits.

5. **A3 Desktop startup and prompt-submit resilience**
   - Port after core event/clarify paths are stable.

## Guardrails

- Do not mutate `main` while building the clean candidate.
- Use a separate git worktree for the candidate.
- Use `cherry-pick --no-commit` for the first porting step; do not create a commit until after verification and user approval.
- If `cherry-pick --no-commit` stages changes, unstage them after clean application unless a commit was explicitly approved.
- Stop on conflicts and report exact files; do not auto-resolve broad conflicts.
- Run focused tests for each stack before moving to the next stack.
- Do not push, merge, reset, or rebase without a separate explicit approval.

## Initial command plan

```bash
cd /c/Users/82109/AppData/Local/hermes/hermes-agent

git worktree add \
  -b hq/clean-priority-a-20260705-195045 \
  /c/Users/82109/AppData/Local/hermes/worktrees/hq-clean-priority-a-20260705-195045 \
  origin/main

cd /c/Users/82109/AppData/Local/hermes/worktrees/hq-clean-priority-a-20260705-195045

git cherry-pick --no-commit c1710d52ef7554da4acc20ef9e5544fa572c1ab1

git restore --staged .
git diff --name-status
git diff --check

export PYTHONPATH="$PWD"
python -m pytest tests/gateway/test_restart_drain.py -v --tb=short -n 0
```

## Expected result for step C

A4 should be present as uncommitted changes in the clean candidate worktree, with no staged changes and no `CHERRY_PICK_HEAD`/`MERGE_HEAD` state. If tests pass, the next approval gate is whether to commit A4 or continue porting A5 first.
