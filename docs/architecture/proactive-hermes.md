# Proactive Hermes MVP

Hermes is the primary agent. It owns planning, memory, context, policy, risk
classification, approvals, audit, and user-facing responses. OpenClaw is only
an execution gateway: Hermes may delegate an approved `DelegatedTask`, but
OpenClaw must return a `DelegatedResult` to Hermes and must not reply to KJ
directly.

## Runtime Flow

1. Hermes loads standing orders on startup, heartbeat, and user-message driven
   proactive checks.
2. Hermes evaluates tool policy before every side effect.
3. The heartbeat runner scans Obsidian, cron health, and delegated task state.
4. No anomaly and no due proactive follow-up returns `[SILENT]`; cron
   suppresses delivery while the audit log is still written.
5. An anomaly, stuck task, due progress report, or waiting-for-KJ input
   produces a notification payload for the Hermes channel.
6. Low-risk tool execution may be delegated to OpenClaw through the bridge.
7. High-risk or unknown actions stop at an approval gate.

## Standing Orders

Standing orders are markdown sections with these fields:

- `scope`: authorization boundary.
- `trigger`: `schedule`, `event`, or `condition` trigger text.
- `allowed_actions`: comma-separated low-risk actions.
- `approval_gates`: actions that must stop for KJ approval.
- `escalation_rules`: when Hermes must stop and ask.
- `output_policy`: when to notify versus remain silent.

The MVP parser is intentionally conservative and accepts simple `##` sections
with `- key: value` lines. Richer frontmatter can be added later without
changing the heartbeat contract.

## Tool Policy

`config/tool-policy.yaml` divides actions into:

- `AUTO_ALLOW`: read-only checks, Obsidian reads, audit writes, local drafts,
  local reports, and low-risk delegation.
- `CONFIRM_FIRST`: external sends, production changes, deploys, deletion,
  money movement, secret/API-key changes, and publishing.
- `DENY`: secret leakage, approval bypass, audit disablement, unauthorized
  financial access, and unconfirmed trades.

Unknown actions default to `CONFIRM_FIRST`.

## Audit and Memory

The Obsidian adapter writes append-only daily notes:

- `System/Agent Runs/YYYY-MM-DD.md`
- `System/Commitments/YYYY-MM-DD.md`
- `System/Delegated Tasks/YYYY-MM-DD.md`

Each run records timestamp, trigger type, decision, risk level, actions,
tools, result summary, follow-up, and whether KJ was notified.

## Progress and Missing-Information Follow-up

Hermes should not wait for a new user message when an active task is blocked
on missing information. The heartbeat scans commitment and delegated-task
ledgers for:

- `status: waiting_for_kj`, `blocked`, `stuck`, `failed`, or `timeout`.
- `condition: awaiting_kj_input` or `needs_kj_input`.
- `next_action` text that asks KJ to provide or confirm information.

Waiting-for-KJ items generate a concise Hermes-channel reminder on a throttled
cadence. Open active commitments and delegated tasks can also produce periodic
progress reports, but only after the configured progress interval. The throttle
state is written to `System/Proactive State/heartbeat-state.yaml`, keeping
Obsidian append-only logs separate from small operational state.

When Hermes itself asks KJ for missing information in a final assistant
response, the gateway records a `waiting_for_kj` commitment before delivery.
This keeps "please provide the photos/details/confirmation and I will continue"
from becoming a dead-end chat message. The heartbeat later notices the record
and asks whether the requested information is ready.

## Config Guidance

Do not set `approvals.mode: off` for proactive Hermes. Recommended runtime
settings are:

```yaml
agent:
  tool_use_enforcement: auto
approvals:
  mode: smart
  cron_mode: deny
terminal:
  backend: docker
cron:
  provider: ""
```

Use `terminal.backend: local` only when the environment is already sandboxed or
the operator accepts the risk. The setup script reports recommendations but
does not mutate `~/.hermes/config.yaml`.
