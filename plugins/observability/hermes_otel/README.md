# Hermes OTel Observability Plugin

Built-in Hermes plugin that emits structured observability spans for every LLM
API call. Phase 1 writes spans as JSONL to a local file; Phase 2 will add OTLP
HTTP export to a self-hosted OpenTelemetry Collector.

## What is collected per API call

- Provider/model called (provider, model, base_url, api_mode)
- Token counts (input, output, cache_read, cache_write, reasoning, prompt, total)
- Estimated cost (USD, status, source)
- Credits state when available (remaining USD/micros, paid_access, used_fraction)
- Rate-limit state when available (RPM/TPM remaining/limit)
- Errors (error_type, message, status_code, failover_reason, retry_count)
- Session metadata (session_id, turn_id, api_request_id, platform, profile)

No message content is exported.

## Enable

```bash
hermes plugins enable observability/hermes_otel
```

Or via `hermes tools` → **Hermes OTel**.

The plugin is opt-in and disabled by default. When disabled it is completely
inert; when enabled it still fails open on any runtime error.

## Verify

```bash
hermes plugins list                 # observability/hermes_otel should show enabled
hermes chat -q "hello"              # then inspect ~/.hermes/logs/otel-spans.jsonl
```

## Configuration

Optional environment variables (set in `~/.hermes/.env` or via `hermes tools`):

```bash
HERMES_OTEL_ENABLED=true                  # master switch (default: true when plugin enabled)
HERMES_OTEL_OUTPUT_PATH=...               # override JSONL path
HERMES_OTEL_FLUSH_SECONDS=5               # in-memory flush interval
HERMES_OTEL_MAX_BUFFER=100                # buffer cap before flush
HERMES_OTEL_SAMPLE_RATE=1.0              # 0.0-1.0 sampling
```

## Disable

```bash
hermes plugins disable observability/hermes_otel
```

## Phase 2 preview

The collector/Grafana stack will live on the Lyons server. The plugin will gain
an OTLP HTTP exporter targeting `http://localhost:4318`, with the collector
fanning out to Prometheus (metrics) and Loki (logs/spans) for Grafana
dashboards.
