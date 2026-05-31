# Thunderdome B-Team Feedback: Kanban PR Respawn Guard

## A-Team / Builder Summary

Implemented a narrow guard predicate for Kanban dispatcher respawn protection:

- Previous behavior: any recent GitHub PR URL in task comments triggered `active_pr`.
- New behavior: a recent GitHub PR URL only triggers `active_pr` when the same comment includes task-ownership/opening language such as `PR created`, `Opened <github PR URL>`, or explicit task-owned keys such as `created_pr`, `opened_pr`, `submitted_pr`, or `task_pr`.
- Upstream/reference/evidence PR links no longer wedge ready tasks behind `active_pr`.

## Verification Evidence

- RED: new upstream/reference PR tests failed under the old predicate.
- GREEN: after implementation, the focused tests passed.
- Full targeted module: `209 passed in 7.50s` for `tests/hermes_cli/test_kanban_db.py`.
- Lint: `ruff check hermes_cli/kanban_db.py tests/hermes_cli/test_kanban_db.py` passed.

## Outside-Family B-Team Critic (Claude Opus 4.7)

Claude reviewed the diff and SDD read-only. It approved the direction but identified one concrete ambiguity: bare `pr_url` / `pull_request_url` keys are too noisy because they may appear in upstream evidence metadata.

Resolution applied by host:

- Added RED test `test_respawn_guard_upstream_pr_url_key_not_guarded`.
- Verified it failed under the initial implementation.
- Removed bare `pr_url` / `pull_request_url` from the ownership-hint regex.
- Re-ran targeted tests and lint successfully.

OUTSIDE_FAMILY_B_TEAM_VERDICT: APPROVE

## DeepSeek B-Team Critic

DeepSeek/Pi review was attempted with `pi --provider deepseek --model deepseek-v4-pro`, but it exceeded the 300-second foreground timeout and wrote no usable output. Per Thunderdome fallback rules, unavailability is recorded and the loop proceeds with outside-family review plus local verification.

DEEPSEEK_B_TEAM_VERDICT: UNAVAILABLE

## Verdict

VERDICT: APPROVED
