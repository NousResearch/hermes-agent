# Subagent milestone review

An opt-in bundled plugin for evidence-bearing intermediate reports and explicit
parent-reviewed timeout renewal. It does not add tools to the core toolset and
requires no service, credentials, or network dependency.

## Enable

Merge into the active profile's `config.yaml`, preserving other enabled plugins:

```yaml
plugins:
  enabled: [subagent-progress]
  entries:
    subagent-progress:
      allow_gateway_injection: true

delegation:
  child_timeout_seconds: 600
  reviewed_timeout: true
```

Restart the session after enabling it. The plugin's `subagent_progress` toolset
can be managed through `hermes tools`; include it when explicitly restricting a
child's toolsets. Existing conversations keep their cached tool schemas.

`reviewed_timeout` defaults to off. With it off, upstream's timeout remains an
**inactivity** budget that renews on runtime activity. The plugin can still save
milestones, but there is no reviewable lease. With it on, a positive, finite
`child_timeout_seconds` is required. Only the direct owning parent's explicit
approval grants a fresh window of that duration from the approval time. Unused
time is not accumulated. Heartbeat stale-worker cancellation remains independent.

## Workflow

1. A child calls `report_progress` after a meaningful milestone or when blocked,
   supplying completed work, evidence references, and the next step. A blocker
   requiring a decision also sets `needs_decision: true`.
2. The parent independently inspects the referenced evidence against the original
   goal. Reports are untrusted claims, not proof or user instructions.
3. The parent calls `review_subagent_progress` with `approve`, `steer`, or `stop`,
   a reason, and evidence references. Approval requires `evidence_checked`.
   The runtime checks ownership and freshness, not whether the claimed inspection
   was actually performed; the parent is responsible for evidence quality.

Reports, steering, silence, tool activity, and heartbeats never renew a reviewed
lease. Self-review, ancestor/sibling review, reset sessions, expired/stopped
children, duplicate reviews, and superseded checkpoints cannot renew it. Verified
compression descendants retain ownership. Schema correction runs inside the
same supervised worker and lease, not a new timeout window.

## Delivery and lifecycle

Reports are committed before notification to
`$HERMES_HOME/state/subagent-progress.sqlite3`. The database stores goals, report
payloads, review decisions, supervision observations, and delivery receipts. Do
not put secrets in reports. The database is local, not outbound telemetry, and
survives restart; live delegation and renewable leases do not survive restart.

Gateway wakes require separate injection consent. They contain a checkpoint ID,
are pinned to the original conversation, and queue non-preemptively while the
parent is busy. They do not interrupt its tool execution or start a concurrent
parent turn. Stale/consumed/reviewed wakes are discarded without losing queued
user messages.

A busy parent can also receive checkpoint context in the **same turn**, appended
to its next fresh top-level tool result before canonical session persistence.
Previously persisted messages, system prompts, and nested `execute_code` JSON
results are not rewritten. This path consumes a checkpoint only after the exact
tool message is confirmed durable; absent databases, failed flushes, and failed
spill writes leave it pending. Oversized context retains a retrievable full-text
reference, and receipt callbacks use the bounded hook runner. Ordinary start-of-
turn context hooks retain their existing behavior.

Scheduling, adapter admission, durable context delivery, model consumption,
explicit review, and user-facing message delivery are distinct states. Neither
an admission receipt nor durable context delivery renews the child's window.
A long-running tool or model call can still delay review past expiry; there is
no preemption or automatic extension while the parent is busy.

The CLI displays milestones and can read persisted checkpoints at its next
context boundary. Pinned autonomous wakes are supported only by the messaging
gateway, not CLI/TUI/desktop injection hosts; unsupported hosts fail closed.
Do not enable a short reviewed budget unless the parent can review within it.

A periodic supervisor records activity without treating it as evidence of
progress. At most one alert per approval window is requested after half the
window has elapsed. Alerts cannot themselves be approved. Notifications retry
at most three times in the running process. Unloading cancels supervision and
pending display work. Terminal child status preserves the last milestone for
the parent, including after child-session compression.

## Tests

```bash
scripts/run_tests.sh tests/plugins/subagent_progress tests/tools/test_delegate_reviewed_deadline.py tests/tools/test_delegate_reviewed_wait.py tests/tools/test_delegate_schema_deadline.py tests/gateway/test_plugin_review_delivery.py tests/gateway/test_plugin_orphan_review.py
```

Tests use isolated homes, local SQLite, real runtime imports, and in-memory
adapters. No real model or messaging service is required.
