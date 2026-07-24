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
| Gateway gauges | `/v1/metrics` | `hermes.gateway.up/state/busy/drainable/active_agents/restart_requested`, `hermes.platform.up/degraded` with bounded `error_code` attributes |
| Health/lifecycle events | `/v1/traces` | `gateway.lifecycle` state transitions (`starting -> running -> draining -> stopped`, `startup_failed`, exit), `gateway.health_snapshot`, platform state changes |
| Diagnostics | `/v1/logs` | Warning/error gateway events with a constant body and bounded subsystem, severity, error class, and error code attributes; rendered log messages are never exported |
| Cron scheduler gauges | `/v1/metrics` | Ticker heartbeat and last-success age (omitted when unavailable), a monotonic catch-up-occurrence count from the scheduler's stale-window branch, enabled/running job counts, and overdue count derived from persisted `next_run_at` plus the scheduler's existing grace rule |
| Cron execution lifecycle | `/v1/traces` | Durable `claimed/running/completed/failed/unknown` states, bounded source and error class, opaque hashed job key, elapsed duration when timestamps exist, and delivery outcome when the scheduler knows it; terminal states make a fail-open flush attempt that can delay completion by up to one second |

Signals carry `service.name`, version, supervision mode, and a stable one-way
hash of the install id so an operator can distinguish instances without
exporting account/profile identity or the raw install identifier.

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
a plain `hermes gateway run` — the exporter lives in the gateway process, so
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
