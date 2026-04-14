# Low-Noise Interaction Policy v1

Date: 2026-03-21
Status: active (Wave 8 C1 Operator Cockpit UX)
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose
To enforce strict signal-to-noise ratio in the operator's primary channel (e.g., Telegram). Watchdogs and background subagents that successfully complete their tasks should not notify the operator unless they fail, drift out of limits, or require explicit manual approval.

## 1) Notification Rules

### 1. Silent Successes
- Routine heartbeat checks (`openclaw gateway status`).
- Successful execution of `verify_then_resume.sh` with a `READY` status.
- Continuity checkpoint creations that do not exceed the `1 hour` drift threshold.
- Reconciliations of `autopilot_delegated_ingress` that don't emit a blocking error.
- Note: transitions defined as mandatory triggers in `docs/ops/execution_meaningful_event_reporting_doctrine_v1.md` are **not** silent successes.

### 2. Degraded Warnings (Batch/Daily)
- Soft warnings (e.g., `web_capture_scheduler_stale`, `idle_lane_autospawn_trace_stale`) should not trigger immediate notifications unless they stack up. They should be delivered via a batched daily report or a grouped alert.
- **Rule:** Do not immediately ping on `caution` statuses unless requested by an operator explicitly via a `/status` or `/health` command.

### 3. Immediate Alerts (Blockers & Failures)
- When a `safe-to-act` mutation gate returns `forbidden`.
- When an SLO error budget is depleted (`failing` state in A6).
- When an explicit `a6_observability_failed` blocker halts a Wave execution.
- **Rule:** The notification MUST be a Unified Cockpit Action Card (`cockpit_action_card_design_v1.md`), containing the failure and the remediation hint.

## 2) Enforcement
The `cockpit_alert_router.py` must intercept raw logs/snapshots and decide whether to route the alert to Telegram based on these rules.

Personal OS coupling (`XP-301`):
- `docs/ops/personal_os_scope_boundary_contract_v1.md` is the canonical boundary authority for Personal OS refusal/escalation and approval semantics.
- Any Personal OS interaction classified as `E1..E4` escalation must bypass silent-success behavior and surface as an explicit blocked/escalated action card.
