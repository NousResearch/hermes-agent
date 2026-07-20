# Gate 0 — isolated baseline, toolchain, and execution preflight

Date: 2026-07-17
Task: t_afcb9cba
Worker profile: automation-operator

## Scope guardrails applied
- No source-checkout edits performed.
- No installs/upgrades performed.
- No production code changes performed.
- Only artifact created in this task: `docs/truth-ledger/discovery/gate-0.md`.

## Baseline/worktree verification
- Expected baseline commit: `3a1a3c7e6727a31df89b61b27bad313430bdac45`
- Source checkout (`/Users/hermes/.hermes/hermes-agent`):
  - HEAD: `3a1a3c7e6727a31df89b61b27bad313430bdac45`
  - Branch: `main`
  - Status count: `696` (all `??` untracked; classes `{ '??': 696 }`)
- Shared worktree (`/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`):
  - HEAD: `3a1a3c7e6727a31df89b61b27bad313430bdac45`
  - Branch: `feat/truth-ledger-option-2`
  - Status count at preflight: `0`
- `git worktree list --porcelain` shows both linked worktrees at the same baseline commit.

## Toolchain and assignment checks
- `command -v codex`: `/Users/hermes/.local/bin/codex`
- `codex --version`: `codex-cli 0.144.5`
- `codex login status`: `Logged in using ChatGPT`
- `hermes kanban assignees`: includes `automation-operator` with `running=1, todo=30`
- Environment confirms active task/workspace:
  - `HERMES_PROFILE=automation-operator`
  - `HERMES_KANBAN_TASK=t_afcb9cba`
  - `HERMES_KANBAN_WORKSPACE=/Users/hermes/.hermes/hermes-agent/.worktrees/truth-ledger-option-2`

## Readiness checks
- Kanban board stats show active queue and this running task.
- Disk free on workspace volume: `295Gi` available (`33%` used).
- Test/plugin conventions inventory:
  - Test runner present and executable: `scripts/run_tests.sh`
  - Plugin tree present under `plugins/` with provider plugin structure (`plugin.yaml`, `__init__.py`, `provider.py`).

## Acceptance assessment
- PASS: shared worktree linked on `feat/truth-ledger-option-2` at exact baseline.
- PASS: automation-operator assignment/toolchain/profile readiness checks.
- PASS: dispatcher/kanban readiness and disk headroom checks.
- ATTENTION: source checkout dirty entry count observed as `696` (all untracked), not `695` stated in card text.
  - No tracked-file mutations were observed; source HEAD remained pinned to baseline.
  - This count discrepancy appears pre-existing to this run and should be treated as baseline-drift evidence for orchestrator review.
