# Gateway Monitoring

Service health monitoring plus structured operational diagnostics for the
Hermes gateway daemon, exported over OTLP/HTTP to an operator-configured
endpoint (OpenTelemetry Collector, DataDog, or any OTLP receiver).

This plane is content-free by construction. It exports gateway and cron
lifecycle state, platform connector health, and content-free warning/error
diagnostics. It never exports prompts, messages, tool arguments or results,
job names, destinations, schedules, raw errors, session history, usage
analytics, audit logs, or detailed execution traces. Run/model/tool trajectory
capture is a separate plane served by the NeMo Relay integration
(`plugins/observability/nemo_relay/`) and its Hermes-owned subscribers.

## What gets exported

| Signal | OTLP route | Content |
| --- | --- | --- |
| Gateway gauges | `/v1/metrics` | `hermes.gateway.up/state/busy/drainable/active_agents/background_work/background_delegations/restart_requested`, `hermes.platform.up/degraded` with bounded `error_code` attributes |
| Health/lifecycle events | `/v1/traces` | `gateway.lifecycle` state transitions (`starting -> running -> draining -> stopped`, `startup_failed`, exit), `gateway.health_snapshot`, platform state changes |
| Diagnostics | `/v1/logs` | Warning/error gateway events with a constant body and bounded subsystem, severity, error class, and error code attributes; rendered log messages are never exported |
| Cron scheduler gauges | `/v1/metrics` | Ticker heartbeat and last-success age (omitted when unavailable), a monotonic catch-up-occurrence count from the scheduler's stale-window branch, enabled/running job counts, and overdue count derived from persisted `next_run_at` plus the scheduler's existing grace rule |
| Cron execution lifecycle | `/v1/traces` | Durable `claimed/running/completed/failed/unknown` states, bounded source and error class, opaque hashed job key, elapsed duration when timestamps exist, and delivery outcome when the scheduler knows it; terminal states make a fail-open flush attempt that can delay completion by up to one second |

Signals carry `service.name`, version, supervision mode, and a stable one-way
hash of the install id so an operator can distinguish instances without
exporting account/profile identity or the raw install identifier.

`hermes.gateway.active_agents`, `hermes.gateway.background_work`, and
`hermes.gateway.background_delegations` are complementary. `active_agents`
counts foreground message turns plus in-flight cron jobs plus API runs — the
work the gateway drains on shutdown. `background_work` counts detached work that
`active_agents` never includes: backgrounded `delegate_task` subagents,
`terminal(background=true)` processes, and kanban workers; it is
**task-granular** — a fan-out batch of N subagents counts as N — so it reflects
real concurrent subagent load. `background_delegations` counts only async
delegation **units** (each `delegate_task` dispatch is one, a fan-out batch is
one), matching the async pool's capacity accounting; alert it against
`delegation.max_concurrent_children` to see slot pressure. Sum `active_agents`
and `background_work` for total live work per instance; use
`background_delegations` for pool-saturation.

## Enabling

```yaml
# config.yaml
monitoring:
  gateway_health_export:
    enabled: true
  export:
    otlp:
      enabled: true
      endpoint: http://collector-host:4318/v1/traces   # metrics/logs derive
      headers_env: {}   # header name -> ENV VAR NAME (values never stored)
```

Check the posture any time:

```bash
hermes monitoring status
```

The OpenTelemetry SDK is an optional extra (`pip install 'hermes-agent[otlp]'`),
lazy-installed on first use. When the SDK is missing or the endpoint is down,
the gateway runs unaffected: metric collection and ordinary event export stay
off the hot path, while terminal cron events make one bounded fail-open flush
attempt of up to one second so the final state is less likely to be lost.

Works identically under systemd/launchd/s6 supervision, containers, tmux, or
a plain `hermes gateway run`: the exporter lives in the gateway process, so
no sidecar, agent, or collector is required on the host.

## Collecting into DataDog

Run a customer-owned OpenTelemetry Collector and forward:

```yaml
# otel-collector config
receivers:
  otlp:
    protocols:
      http:
exporters:
  datadog:
    api:
      key: ${env:DD_API_KEY}
service:
  pipelines:
    metrics:   {receivers: [otlp], exporters: [datadog]}
    traces:    {receivers: [otlp], exporters: [datadog]}
    logs:      {receivers: [otlp], exporters: [datadog]}
```

Point `monitoring.export.otlp.endpoint` at the collector. Alerts belong on
`hermes.gateway.up`, `hermes.platform.up`, and `hermes.platform.degraded`.

## Generic fleet queries and alerts

The exact syntax depends on the customer's observability backend. The examples
below use PromQL-style expressions and intentionally avoid vendor-specific
routing, destinations, or customer inventory.

Group fleet views by the opaque `service.instance.id` resource attribute. A
process that has died cannot emit its own zero, so every deployment needs both
explicit-state and missing-series detection.

```promql
# Explicit gateway failure.
hermes_gateway_up == 0

# Box disappeared or stopped exporting. Choose a window longer than the
# configured export interval and collector retry allowance.
absent_over_time(hermes_gateway_up[5m])

# Locally owned bridge is explicitly down.
hermes_platform_up == 0

# Scheduler thread is stale even though the gateway may still be alive.
hermes_cron_scheduler_heartbeat_age_seconds > 180

# Ticker loops but has not completed a successful tick recently.
hermes_cron_scheduler_last_success_age_seconds > 300

# One or more jobs are beyond their existing scheduler grace window.
hermes_cron_jobs_overdue > 0

# Catch-up counter increased, proving at least one stale occurrence was
# collapsed and run once after a delay.
increase(hermes_cron_scheduler_catch_up_occurrences[15m]) > 0
```

Cron execution lifecycle records arrive as `hermes.cron_execution` spans.
Alert or derive events from bounded attributes such as:

```text
hermes.status = failed|unknown
hermes.delivery_outcome = failed|not_configured
hermes.error_class = auth_failed|rate_limited|timeout|network_error|
                     dispatch_failed|interrupted|empty_response|
                     invalid_config|unknown
```

Recommended operator views:

1. one row per `service.instance.id` with gateway and configured local-platform
   state;
2. scheduler heartbeat, last-success age, running count, overdue count, and
   catch-up increase;
3. a cron lifecycle feed keyed only by opaque `hermes.job_key`;
4. separate alerts for box absence, local bridge down, scheduler stale, cron
   failed/unknown, delivery failure, and overdue/catch-up activity.

Keep alert thresholds and routing in deployment-owned configuration. Do not add
job names, prompts, outputs, schedules, destinations, raw errors, profile names,
or account identity merely to make a dashboard easier to read.

## Release-validation scenarios

Before accepting a deployment, force and verify all five cases through the real
collector and backend:

1. **Cron success:** observe `claimed -> running -> completed`, duration, and a
   truthful delivery outcome.
2. **Cron failure:** observe `failed` plus a bounded error class, with no raw
   exception or content in the decoded OTLP payload.
3. **Cron interruption:** stop the owning gateway during execution, restart it,
   and observe recovery to `unknown`.
4. **Locally owned bridge outage:** break one native connector, observe its
   bounded down/retrying/fatal state and recovery, and verify unaffected boxes
   remain healthy.
5. **Killed gateway:** terminate one canary, verify missing-series detection,
   restart it, and confirm the same opaque instance identity returns.

Hermes Agent-owned Relay transport health remains in scope. A separate gateway
or connector service remains authoritative for any shared connected-platform
state that it owns and should export that state through its own telemetry path.

For every scenario, verify the signal and alert clear on recovery, other boxes
remain unaffected, collector failure stays fail-open, and decoded metrics,
spans, logs, and resource attributes remain content-free.

## Local smoke test (no Docker)

```bash
# terminal 1: capture collector on :4318
python scripts/observability/otel_capture_collector.py \
  --host 127.0.0.1 --port 4318 --log /tmp/hermes_otel_capture.jsonl

# terminal 2: drive the real exporter through lifecycle transitions,
# a fatal platform, and a structured warning event, then flush
python scripts/observability/gateway_health_export_probe.py \
  --endpoint http://127.0.0.1:4318/v1/traces \
  --log /tmp/hermes_otel_capture.jsonl --wait 8
# exit 0 prints: {"requests": 6, "paths": ["/v1/logs", "/v1/metrics", "/v1/traces"]}
```

## Maintaining and extending this plane

This plane is a **fixed, enumerated, content-free vocabulary** by design. Adding
a signal is not just "emit a new metric" — every new name and attribute must be
declared in each layer that enforces the bounded vocabulary, or it is silently
dropped downstream. Follow the checklist for the change you are making. The
golden rule: **a new signal that is emitted but not declared in every layer
looks like a code bug but is a vocabulary-registration bug — nothing errors, the
signal just never arrives.**

### Content-free invariant (applies to every change)

Before adding anything, confirm it cannot carry content. Numbers, booleans,
ages, durations, monotonic counts, and one-way hashes are safe. **Never** add an
attribute that can hold a job name, prompt, output, schedule, destination, raw
exception text, file path, profile name, account id, or free-form string. When
you must key a record to a job/entity, hash it (`sha256(...)[:24]`, see
`_job_key` in `agent/monitoring/cron_health.py`) — never emit the raw id. All
string attributes that could touch user input must pass through
`redaction.redact_for_export` and be truncated (see `_span_attrs` in
`agent/monitoring/otlp_exporter.py`).

### Adding a new gauge/metric

1. Emit it in the snapshot builder (`agent/monitoring/gateway_health.py`
   `build_gateway_health_snapshot`, `cron_health.py` `build_cron_health_snapshot`,
   or a sibling reader wired into `_read_runtime_snapshot` in
   `gateway_health_export.py`). Best-effort: never let a reader raise into the
   collection loop — wrap it and log a **content-free WARNING with the exception
   TYPE name only** (the pattern the cron and background-work readers use), so a
   future regression is visible instead of silently dropping the signal.
2. Register the dotted metric name in the observable-gauge `metric_names` list in
   `gateway_health_export.py::_start_metric_provider`. **A gauge that is emitted
   in the snapshot but not registered here is never observed.**
3. Add the export-table row and an alert example in this file.
4. If the deployment fronts the exporter with an OpenTelemetry Collector that
   uses a metric-name allowlist (a `filter/...` processor with `name != "..."`
   guards), add the new name there too — otherwise the collector drops it before
   the backend. This is not repo code, but it is the single most common reason a
   correctly-emitted new metric never appears; call it out in the PR so the
   deploying operator updates their collector config.

### Adding a new subsystem (a new family of signals)

Mirror the cron pattern (`cron_health.py` + its wiring): put the read/projection
logic in its own module, expose one `build_<subsystem>_health_snapshot()` that
returns bounded `GatewayMetric`s (and events if any), and extend it into
`_read_runtime_snapshot` with the same best-effort try/except-WARNING guard.
Then do the "adding a metric" checklist for each new name, and the "adding an
attribute" checklist for each new event attribute. Add a release-validation
scenario below for the subsystem's failure mode.

### Extending the error-class / status / source / state vocabularies

These are the closed enums that keep the plane bounded. Extend the SET, then the
classifier, never one without the other:

- **Cron** (`agent/monitoring/cron_health.py`): `_KNOWN_STATUSES`,
  `_KNOWN_SOURCES`, `_KNOWN_DELIVERY_OUTCOMES`, and the `classify_cron_error`
  keyword buckets. Anything not in the set is coerced to `unknown` on the way
  out, so a new value that is not added to the set is invisible.
- **Gateway/platform** (`agent/monitoring/gateway_health.py`):
  `_KNOWN_GATEWAY_STATES`, `_KNOWN_PLATFORM_STATES`, and `classify_gateway_error`.

Rules: keep the vocabulary SMALL and operationally meaningful (an error class
should map to an operator action, not to an exception subclass); a new bucket
must match on a stable keyword, not on message text that could vary; update the
`hermes.error_class = ...` list in this file's alert section and the enum's unit
test so the contract is asserted, not frozen as a count.

### Adding a content-free attribute to an existing event/span

Add the key to the emitter's per-kind `keep_by_kind` allowlist in
`agent/monitoring/otlp_exporter.py::_span_attrs` (unlisted keys are dropped), run
it through redaction if it is ever string-shaped, and — as with metrics — if the
deployment's collector has a span-attribute `keep_keys(...)` allowlist, add the
attribute there too or it is stripped in transit.

### Verify the whole chain, not just emission

Emitting is necessary but not sufficient. Confirm the signal survives all the
way to the backend, because the enums, the `metric_names` registration, the
emitter attribute allowlist, and any collector allowlist each drop unlisted
values with no error:

```bash
hermes monitoring status                 # posture
python scripts/observability/gateway_health_export_probe.py \
  --endpoint http://127.0.0.1:4318/v1/traces \
  --log /tmp/cap.jsonl --wait 8          # drive the real exporter
```

Decode the captured OTLP payload and assert the new name/attribute is present
AND that no content leaked. When a real collector sits in front, add its
allowlist entries and re-verify against the backend, not just the local capture.

## Boundaries and roadmap

The `hermes monitoring` CLI intentionally exposes `status` only. This first
release covers only Hermes Agent-owned service-health and operational-diagnostic
signals, including Hermes Agent-owned Relay transport health. Team Gateway's
authoritative shared connector/platform state is explicitly out of scope, as
are product analytics, audit/quality reporting, and detailed execution traces.
Shared client usage metrics and enterprise trace telemetry are being designed on
the NeMo Relay integration with their own consent, policy, and export
boundaries; this monitoring plane stays narrow so an operator can enable it
without touching any content-bearing signal. The telemetry surface may be
reorganized as that lands.
