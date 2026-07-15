# OpenTelemetry OTLP Observability Plugin

This bundled observer exports privacy-safe Hermes trajectories as OpenTelemetry
traces over OTLP. It is disabled by default and fails open if configuration,
the optional SDK, or the collector is unavailable.

## Install and enable

```bash
pip install "hermes-agent[otel]"
hermes plugins enable observability/otel

export HERMES_OTEL_ENABLED=1
export HERMES_OTEL_ENDPOINT=http://localhost:4318/v1/traces
export HERMES_OTEL_PROTOCOL=http
export HERMES_OTEL_SERVICE_NAME=hermes-agent
```

For OTLP/gRPC, use a gRPC endpoint and protocol:

```bash
export HERMES_OTEL_ENDPOINT=http://localhost:4317
export HERMES_OTEL_PROTOCOL=grpc
```

Optional collector headers are a JSON object. Keep credentials in the
environment rather than repository files:

```bash
export HERMES_OTEL_HEADERS='{"authorization":"Bearer ..."}'
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HERMES_OTEL_ENABLED` | `0` | Set to `1` to initialize tracing. |
| `HERMES_OTEL_ENDPOINT` | OTel SDK default | OTLP trace endpoint. |
| `HERMES_OTEL_PROTOCOL` | `http` | `http` or `grpc`. |
| `HERMES_OTEL_SERVICE_NAME` | `hermes-agent` | OTel resource service name. |
| `HERMES_OTEL_HEADERS` | `{}` | JSON object of OTLP headers. |

The provider uses the OTel SDK batch span processor. Hermes calls
`provider.shutdown()` from `on_session_finalize` to flush queued spans.

## Span model

- `hermes.session` roots each conversation run.
- `hermes.turn` represents one user turn.
- `hermes.llm_request` represents one provider API attempt.
- `hermes.tool.<tool_name>` represents one tool execution.
- `hermes.subagent` represents delegated child work.
- `hermes.approval` represents a dangerous-command approval decision.

Correlation attributes include stable Hermes session, turn, request, tool, and
subagent IDs. Request spans include model, provider, token usage, status, and
duration metadata.

## Privacy

The plugin never emits raw user messages, conversation history, API request or
response bodies, tool arguments or results, command text, goals, summaries,
error messages, API keys, or headers. It exports only stable IDs, safe labels,
counts, lengths, durations, token usage, status fields, and non-content
length summaries for commands and delegated goals.

## Verify

```bash
hermes plugins list
hermes chat -q "hello"
```

Then inspect your collector backend for the `hermes-agent` service. To disable
export without removing the plugin, set `HERMES_OTEL_ENABLED=0`.
