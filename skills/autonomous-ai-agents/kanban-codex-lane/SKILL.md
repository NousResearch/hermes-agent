---
name: kanban-codex-lane
description: Run Codex as an isolated Kanban build lane.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [kanban, codex, worktrees, autonomous-agents, prediction-market-bot]
    related_skills: [kanban-worker, codex, hermes-agent]
---

# Kanban Codex Lane

## Overview

This skill defines the lightweight Hermes+Codex dual-lane convention for Kanban workers. Hermes is always the task owner: it calls `kanban_show`, decides whether Codex is appropriate, creates or selects an isolated workspace, starts and monitors Codex, reconciles any diff, runs verification, and writes the final `kanban_complete` or `kanban_block` handoff. Codex is an input lane only. Codex output is not a task completion signal, not a trusted reviewer, and not allowed to write durable Kanban state directly.

The convention exists so a Hermes worker can use Codex for bounded implementation help without changing the dispatcher. The dispatcher must still spawn Hermes workers. A worker may optionally spawn Codex inside its own run, then accept, partially accept, or reject the lane after independent review and tests.

The skill-local helper at `scripts/codex_goal_lane.py` provides the operator controls for a concrete Hermes -> Codex builder -> Claude reviewer -> Hermes verifier lane. It intentionally lives with this skill instead of core dispatcher code so Hermes can reuse the existing worker-lane pattern while still getting `preflight`, `run`, `status`, `logs`, `stop`, `review`, and `verify` controls.

## When to Use

Use the Codex lane when all of these are true:

- The Kanban task is a coding, refactor, documentation, test, or mechanical migration task with clear acceptance criteria.
- A bounded diff can be evaluated by Hermes in one run.
- The repo can be copied or checked out in an isolated git worktree/branch.
- Hermes can run the relevant tests itself after Codex exits.
- The prompt can state all safety constraints and files that must not change.

Do not use the Codex lane when any of these are true:

- The task requires human judgment that is not already captured in the Kanban body.
- The worker lacks repo access, Codex auth, or time to reconcile the result.
- The task requires raw secret disclosure, credential rewriting, private user data export, or production order-entry changes outside the explicit objective.
- A small direct edit is faster and safer than spawning another agent.
- The task is research-only and should produce a written handoff rather than a diff.
- The worker would be tempted to mark Done based only on Codex self-report.

## Prerequisites

- `git` is available and the target repo has a valid HEAD.
- `codex` is installed and authenticated through an existing CLI session, `OPENAI_API_KEY`, or Codex OAuth state.
- `claude` is installed and authenticated through an existing Claude Code session or supported API credentials.
- The task can run in an isolated worktree/branch and has clear acceptance criteria.
- Hermes has enough time to run independent verification after Codex and Claude finish.

Credentials may be used by Codex, Claude, GitHub, package registries, browser automation, or local developer tools when the task requires them. Raw secret values must not be printed, copied into logs, committed, or rewritten unless the explicit task is credential repair. If credential repair is required, stop the lane and report the exact credential/tool that failed without exposing the secret value.

## How to Run

Use the helper for a fully tracked lane:

```bash
LANE=skills/autonomous-ai-agents/kanban-codex-lane/scripts/codex_goal_lane.py
REPO=/path/to/repo
TASK_ID=t_manual
GOAL_FILE=/tmp/codex-goal.txt

python "$LANE" preflight --repo "$REPO"
python "$LANE" run --repo "$REPO" --task-id "$TASK_ID" --goal-file "$GOAL_FILE" --mode auto --autonomy yolo
python "$LANE" status --run-id "<run_id>"
python "$LANE" logs --run-id "<run_id>"
python "$LANE" review --run-id "<run_id>"
python "$LANE" verify --run-id "<run_id>" --command "scripts/run_tests.sh tests/tools/test_x.py" --accept
python "$LANE" stop --run-id "<run_id>"
```

Use `--simulate` for dry-runs and tests when live Codex or Claude should not be invoked:

```bash
python "$LANE" run --repo "$REPO" --task-id "$TASK_ID" --goal-file "$GOAL_FILE" --mode goal --autonomy yolo --simulate
python "$LANE" review --run-id "<run_id>" --simulate --simulate-verdict pass
python "$LANE" verify --run-id "<run_id>" --simulate-pass --accept
```

The helper writes lane state under `$HERMES_HOME/codex-goal-lanes/<run_id>/` by default. Use `--state-root <dir>` for tests or a project-specific operator workspace.

## Quick Reference

| Command | Purpose |
|---|---|
| `codex_goal_lane.py preflight` | Check git, Codex, Claude, auth hints, and worktree support without printing secrets. |
| `codex_goal_lane.py run` | Create/select the isolated worktree and start or simulate the Codex builder phase. |
| `codex_goal_lane.py status` | Show persisted state, including process ownership and phase evidence. |
| `codex_goal_lane.py logs` | Read builder, review, and verifier logs. |
| `codex_goal_lane.py stop` | Stop the tracked Codex process group and mark the lane stopped. |
| `codex_goal_lane.py review` | Run or simulate non-interactive Claude Code review. |
| `codex_goal_lane.py verify` | Run Hermes-owned verification and make the accept/human-review decision. |

## Ownership Rules

1. Hermes owns the Kanban lifecycle. Codex must never call `kanban_complete`, `kanban_block`, `kanban_create`, gateway messaging, or any Hermes board CLI as a substitute for the worker.
2. Hermes owns final acceptance. Treat Codex commits/diffs as untrusted patches until reviewed and verified.
3. Hermes owns test execution. Codex may run tests, but those runs are advisory; repeat required verification from Hermes with the repo's canonical wrapper.
4. Hermes owns safety. If Codex changes safety boundaries, risk gates, live trading behavior, or secrets handling, reject the lane even if tests pass.
5. Hermes owns cleanup. Kill stuck Codex processes and remove temporary worktrees when they are no longer needed.

## Required Worktree and Branch Pattern

Never run Codex directly in a shared dirty checkout. Use a branch/worktree name that ties the lane to the Kanban task and keeps untrusted edits isolated.

Recommended variables:

```bash
TASK_ID="${HERMES_KANBAN_TASK:-t_manual}"
REPO="/path/to/repo"
BASE="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
SAFE_TASK="$(printf '%s' "$TASK_ID" | tr -cd '[:alnum:]_-')"
BRANCH="codex/${SAFE_TASK}/$(date -u +%Y%m%d%H%M%S)"
WORKTREE="/tmp/${SAFE_TASK}-codex-lane"
```

Create the isolated lane:

```bash
git -C "$REPO" fetch --all --prune
git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
git -C "$WORKTREE" status --short --branch
```

If the current Kanban workspace is already an isolated git worktree created for this task, you may create a sibling Codex branch inside it only if `git status --short` is clean except for intentional Hermes edits. Otherwise create a separate temporary worktree and cherry-pick or copy accepted commits back after reconciliation.

Cleanup after reconciliation:

```bash
git -C "$REPO" worktree remove "$WORKTREE"
git -C "$REPO" branch -D "$BRANCH"  # only after accepted commits were copied/cherry-picked or intentionally rejected
```

Keep the worktree if it is needed as an artifact for review; record it in `codex_lane.artifacts` and mention it in the handoff.

## Codex Capability Checks

Run these before spawning Codex. Missing Codex is a normal reason to skip the lane, not a task blocker if Hermes can do the task directly.

```bash
command -v codex
codex --version
codex features list | grep -i goals || true
```

If `/goal` support is required, enable or launch with the feature flag only after checking availability:

```bash
codex features enable goals || true
codex --enable goals --version
```

Authentication can be via `OPENAI_API_KEY` or the Codex CLI OAuth state (often `~/.codex/auth.json`). Do not print token files. A missing `OPENAI_API_KEY` is not proof that auth is unavailable.

## Mode Selection

Use `codex exec` for bounded one-shot edits where Codex should exit on its own:

```python
terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
```

Use Codex `/goal` only for broader multi-step work that benefits from durable objective tracking. Launch interactively in a PTY/tmux session or with `codex --enable goals` if the feature is disabled by default. Keep the goal objective self-contained: repo path, task id, safety constraints, allowed scope, acceptance criteria, tests, and commit expectations.

Example `/goal` objective text to paste into Codex:

```text
/goal Work in this repository only: <WORKTREE>. Task: <TASK_ID> <TITLE>.
Hermes owns the Kanban lifecycle; do not call Hermes kanban tools or messaging.
Create small commits on branch <BRANCH>. Follow the PMB safety constraints in the prompt.
Run the requested verification commands and report exact outputs. Stop after producing a diff and summary.
```

Do not use `--yolo` for prediction-market-bot or safety-sensitive repos. Prefer `--full-auto` inside the isolated worktree, then rely on Hermes reconciliation.

For operator-approved maximum-autonomy lanes outside PMB or other safety-sensitive repos, `--autonomy yolo` is allowed only inside the isolated worktree/branch. This maps to the current Codex CLI's full bypass flag when the helper runs live `codex exec`. Keep the blast radius external: the worktree is the sandbox, Hermes still owns final verification, and raw secret values still cannot appear in logs, diffs, commits, or handoff text.

## Prompt Construction

Use the linked template at `templates/pmb-codex-lane-prompt.md` for prediction-market-bot work. For other repos, keep the same structure and replace the PMB-specific safety block with repo-specific invariants.

Every Codex prompt must include:

- `task_id`, title, and full Kanban acceptance criteria.
- Repo path, worktree path, branch name, and allowed file scope.
- Explicit statement: Hermes owns Kanban lifecycle; Codex is an input lane only.
- Required output: concise summary, files changed, commits, tests run, and known risks.
- Prohibited actions: raw secret disclosure, external messaging, board mutation, unrelated refactors, dependency upgrades unless required.
- Verification commands Codex may run and commands Hermes will run afterward.

For PMB, include these mandatory safety constraints verbatim:

```text
PMB safety constraints:
- live-SIM is paper-only; do not add or enable live REST order entry.
- Never use market orders.
- Do not add execution crossing or bypass price/risk checks.
- Do not fake passive fills, fills, PnL, order states, or reconciliation evidence.
- Do not weaken risk gates, limits, kill switches, or fail-closed behavior.
- Keep research/selection outside the C++ hot path unless explicitly requested.
- Do not read, print, write, or require secrets/tokens/credentials.
```

## Monitoring, Timeout, and Kill Behavior

When using the helper, prefer its persisted controls:

```bash
python "$LANE" status --run-id "<run_id>"
python "$LANE" logs --run-id "<run_id>" --phase builder
python "$LANE" stop --run-id "<run_id>"
```

`stop` targets the tracked process group when the platform supports it and records the outcome in the lane state. A stopped or timed-out lane is not accepted; Hermes must inspect artifacts and either verify a safe partial result or park the task in Human Review.

Start long Codex lanes in the background with PTY and completion notification:

```python
result = terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
session_id = result["session_id"]
```

Monitor without interfering:

```python
process(action="poll", session_id=session_id)
process(action="log", session_id=session_id, limit=200)
process(action="wait", session_id=session_id, timeout=300)
```

Send a Kanban heartbeat every few minutes for lanes longer than two minutes, e.g. `kanban_heartbeat(note="Codex lane running in <WORKTREE>; waiting for tests/diff")`.

Kill conditions:

- No useful output for the task's remaining runtime budget.
- Codex requests secrets, production credentials, or external permissions.
- Codex attempts to modify files outside the worktree.
- Codex starts unrelated rewrites or dependency churn.
- Codex is still running near the worker timeout and no safe partial artifact exists.

Kill command:

```python
process(action="kill", session_id=session_id)
```

After kill, inspect `git status --short`, preserve useful patches only if safe, and record `codex_lane.result: timed_out` or `rejected` with a concrete `rejected_reason`.

## Reconciliation Checklist

Hermes must perform this checklist before accepting any Codex lane result:

- [ ] `git -C <WORKTREE> status --short --branch` shows only expected files.
- [ ] `git -C <WORKTREE> diff --stat` and `git diff` were reviewed by Hermes.
- [ ] No secrets, credentials, generated caches, unrelated data, or local artifacts are included.
- [ ] PMB safety constraints were preserved: no live REST order entry, no market orders, no execution crossing, no fake passive fills/PnL, no risk-gate weakening, no secrets.
- [ ] Codex commits are small enough to cherry-pick or squash cleanly.
- [ ] Hermes ran the canonical tests itself, using `scripts/run_tests.sh` for Hermes Agent or the repo's documented wrapper for other repos.
- [ ] Any Codex-run tests are listed separately from Hermes-run tests.
- [ ] Accepted commits/diffs were applied to the Hermes-owned workspace/branch.
- [ ] Rejected or partial work has a concrete reason and artifact path if useful.

Acceptance outcomes:

- `accepted`: Codex diff/commits were reviewed, applied, and verified.
- `partial`: Some Codex work was accepted after edits or cherry-picks; rejected parts are documented.
- `rejected`: No Codex changes were accepted; reason is documented.
- `timed_out`: Codex exceeded the lane budget; useful artifacts may or may not exist.
- `human_review`: Hermes could not safely accept or rework automatically; exact evidence is recorded.
- `rework`: Claude or Hermes found actionable implementation issues that should go back to Codex.

## kanban_complete Metadata Schema

Include this object under `metadata.codex_lane` for every task where the lane was considered. If Codex was not used, set `used: false` and explain why in `rejected_reason` or a sibling `notes` field.

```json
{
  "codex_lane": {
    "used": true,
    "mode": "auto | exec | goal | skipped",
    "run_id": "t_caa69668-20260508100000-abcdef12",
    "worktree": "/absolute/path/to/codex/worktree",
    "branch": "codex/t_caa69668/20260508100000",
    "command": "codex exec --full-auto ...",
    "reviewer": "claude-code",
    "hermes_verification": "required",
    "result": "accepted | rejected | partial | timed_out | human_review | rework",
    "accepted_commits": ["<sha1>", "<sha2>"],
    "rejected_reason": "empty when fully accepted; otherwise concrete reason",
    "tests_run": [
      {"command": "scripts/run_tests.sh tests/tools/test_x.py", "exit_code": 0, "owner": "hermes"},
      {"command": "codex-reported: npm test", "exit_code": 0, "owner": "codex"}
    ],
    "artifacts": ["/absolute/path/to/log-or-patch"]
  }
}
```

For tasks that intentionally skip Codex:

```json
{
  "codex_lane": {
    "used": false,
    "mode": "skipped",
    "worktree": null,
    "branch": null,
    "command": null,
    "result": "rejected",
    "accepted_commits": [],
    "rejected_reason": "Direct Hermes edit was smaller and safer than spawning Codex.",
    "tests_run": [],
    "artifacts": []
  }
}
```

## Common Pitfalls

1. Treating Codex self-report as verification. Always inspect the diff and rerun tests from Hermes.
2. Running Codex in the user's dirty main checkout. Always isolate in a worktree/branch.
3. Letting Codex own Kanban. Codex may summarize progress, but Hermes writes board state.
4. Forgetting PMB safety invariants in the prompt. Missing safety text is a lane setup failure.
5. Using `/goal` for quick edits. Prefer `codex exec` unless durable multi-step continuation is needed.
6. Killing a stuck lane without recording why. `rejected_reason` must explain the decision.
7. Accepting broad unrelated cleanup because tests pass. Reject or cherry-pick only the scoped changes.

## Verification Checklist

- [ ] Codex was skipped or started only after `command -v codex`, `codex --version`, and optional goals feature checks.
- [ ] Codex ran only in an isolated worktree/branch.
- [ ] Prompt included task scope, ownership rules, PMB safety constraints when applicable, and verification commands.
- [ ] Hermes reviewed `git diff` and safety-sensitive files.
- [ ] Hermes ran canonical tests independently.
- [ ] `kanban_complete.metadata.codex_lane` follows the schema above.
- [ ] Temporary processes and unnecessary worktrees were cleaned up.
