# SDD: Kanban respawn guard should only treat task-owned PRs as active PRs

## Problem

The Kanban dispatcher has a respawn guard that returns `active_pr` whenever any recent task comment contains a GitHub pull request URL. The intent is valid: avoid respawning a worker that already opened a PR for the task and might create duplicate PR spam.

The current predicate is too broad. Worker handoffs often include upstream reconnaissance such as related issues or upstream PRs. A comment that merely records an upstream/reference PR is not evidence that the worker opened a task-owned PR. Treating any PR URL as a task-owned PR leaves the task stuck in `ready`: dashboard nudge wakes the dispatcher, the dispatcher skips with `active_pr`, and the operator experiences the button as doing nothing.

## Reproduction

A ready task has a recent handoff comment containing structured evidence like:

```json
{
  "upstream_checked": [
    "https://github.com/NousResearch/hermes-agent/issues/35542 is OPEN and matches the bug",
    "https://github.com/NousResearch/hermes-agent/pull/35544 is OPEN and implements a related narrow fix"
  ]
}
```

Actual behavior: `check_respawn_guard(...) == "active_pr"` and `dispatch_once(...)` skips the task.

Expected behavior: upstream/reference PR URLs do not trigger `active_pr`; the dispatcher may spawn the ready task normally.

## Desired Behavior

### Guard active PRs when a comment explicitly says this task produced/opened a PR

Examples that should still guard:

- `PR created: https://github.com/org/repo/pull/42`
- `Opened https://github.com/org/repo/pull/42`
- structured text using task-owned labels such as `created_pr`, `opened_pr`, `submitted_pr`, or `task_pr`

Bare keys such as `pr_url` or `pull_request_url` are intentionally not sufficient by themselves because they are frequently used for upstream/reference evidence.

### Do not guard reference/upstream/evidence PR URLs

Examples that should not guard:

- JSON or prose under `upstream_checked`
- `related_prs`
- `references`
- `evidence`
- comments saying a PR was checked, reviewed, referenced, or matched upstream behavior

## Implementation Plan

1. Add a small helper in `hermes_cli/kanban_db.py` that classifies whether a comment indicates a task-owned PR.
2. Preserve the existing PR URL regex, but require explicit ownership/opening language rather than any URL match.
3. Update `check_respawn_guard` to call the helper.
4. Add regression tests covering:
   - explicit task-owned PR comments still trigger `active_pr`;
   - upstream/reference PR comments do not trigger `active_pr`;
   - dispatcher spawns ready tasks when the only PR URL is an upstream/reference PR.

## Acceptance Criteria

- Existing active-PR duplicate-prevention behavior remains for explicit worker-created PR comments.
- Upstream/reference PR comments no longer block respawn.
- Targeted Kanban tests pass.
- The PR description includes this SDD, the diagnosis, and the resolved code behavior.
