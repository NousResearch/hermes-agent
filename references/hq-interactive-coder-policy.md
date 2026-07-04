# HQ Interactive Coder Policy (Draft)

Interactive coder/fullstack agent systems (Replit Agent, Bolt, Cursor, OpenHands, Aider, Devin-class tools) expand a single user request into multi-step tool trajectories: terminal, browser, file, web, git, shell. This policy maps those trajectories to HQ Hermes control, eval, and Task Score v1 boundaries.

## Why this exists

- Coder-first agents commonly execute dozens of tool calls per user turn.
- HQ's current safety/eval layer is stronger on per-tool/per-request checks than on multi-step `edit → run → test → approve → continue` trajectories.
- Without trajectory-level scoping, audit trails, rollback points, and cost gates, interactive coder loops can drift from supervised, inspectable automation.

## Scope

- Covers coder-style interactive workflows on HQ Hermes.
- Excludes pure single-shot completions and non-coding agent flows (research, messaging-only).

## Core principles

1. **Small clear tools over raw API sprawl.** Prefer scoped tool wrappers that expose only the resources/commands needed for the task.
2. **Explicit trajectory checkpointing.** Every interactive run must record reversible rollback points: file snapshot, git diff, or terminal state capture.
3. **Actor → resource scope metadata.** Tool calls should carry actor/user/agent/resource/scope/risk labels for audit id generation.
4. **Four-stage approval boundary.** Approval must be explicit at `plan → execute → approve → release`.
5. **Cost-per-task cap.** Coder tasks with repeated tool calls should respect a bounded cost estimate before continuing.
6. **No indefinite daemons or uncontrolled always-on autonomy.**

## Control / eval mapping

- `per-task trajectory` eval replaces coarse single-step labels when tool-call count exceeds threshold.
- Deterministic state diff is the canonical verification between `edit` and `test` steps.
- Rollback is by git revert or checkpoint script; uncontrolled destructive deletes remain approval-gated.

## Mapping to existing HQ artifacts

- Build harness policy references: `hq-harness-dashboard-adapter-build-packet.md`, `hq-health-dashboard-harness-integration-build-packet.md`
- Validator/adapter tests: `tests/scripts/test_hq_harness_validator.py`, `tests/scripts/test_hq_harness_dashboard_adapter.py`

## Open questions / next loop

- Exact threshold from `edit-test` count to `trajectory` mode.
- Shell/session/terminal/file/browser resource scope enum values and cleanup rules.
- Frontier model cost cap rules per task type.
