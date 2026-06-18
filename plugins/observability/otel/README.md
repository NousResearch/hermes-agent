# OpenTelemetry (OTLP) Observability Plugin

A native **OpenTelemetry** observability plugin for Hermes. It exports Hermes's
built-in observability events over OTLP in three complementary ways:

1. **GenAI-convention spans** (`gen_ai.*`) — the standard trace model, so Hermes
   telemetry flows to *any* OpenTelemetry backend (an OpenTelemetry Collector,
   Jaeger, Grafana Tempo, Honeycomb, …) with no vendor lock-in.
2. **Dashboard-shaped OTLP log records** — the `session_start` / `api_response`
   / `tool_result` event model that agent-coding dashboards (the kind Claude
   Code / Codex / Copilot populate) aggregate on, so a Hermes session shows up
   in the Activity / Sessions / Leaderboards views like any other coding agent.
3. **Local-model cost** — when the backend returns no price (Ollama / HF / vLLM
   local models), cost is estimated from the model's parameter size, so spans
   and log records carry a real `cost_usd` instead of `$0`.

Bundled with Hermes but **opt-in** — it only loads when you explicitly enable
it. It slots in beside `observability/langfuse` and `observability/nemo_relay`
and makes no changes to Hermes core. Without the OTel SDK the hooks no-op
silently (the plugin fails open).

## Quickstart

```bash
# 1. Install the OTel SDK + OTLP HTTP exporter (the only hard dependency)
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# 2. Enable the plugin
hermes plugins enable observability/otel

# 3. Send a turn, then look for an "invoke_agent" trace in your OTel backend
hermes chat -q "hello"
```

It works out of the box against a local collector on `http://localhost:4318`.

### Configure (optional)

All configuration is optional. Set these in `~/.hermes/.env`:

```bash
HERMES_OTEL_SERVICE_NAME=agent.coding.hermes   # service.name (backends group on this)
HERMES_OTEL_ENDPOINT=http://localhost:4318     # OTLP HTTP endpoint
HERMES_OTEL_AGENT_NAME=hermes                  # gen_ai.agent.name
HERMES_OTEL_CAPTURE_CONTENT=false              # attach prompt/response text (privacy-gated)
HERMES_OTEL_USER_ID=                           # stamp user.id on the resource
```

Standard OTel env vars are honoured as fallbacks: `OTEL_SERVICE_NAME`,
`OTEL_EXPORTER_OTLP_ENDPOINT`. Prompt / response / tool I/O is attached **only**
when `HERMES_OTEL_CAPTURE_CONTENT=true` (privacy-gated, off by default).

### Verify / disable

```bash
hermes plugins list                 # observability/otel should show "enabled"
hermes plugins disable observability/otel
```

## Architecture

```
invoke_agent        per Hermes turn   (gen_ai.conversation.id = session id)
  ├── chat          per LLM API call  (tokens, cost, finish reason, TTFT)
  └── execute_tool  per tool call     (gen_ai.tool.name / .type, errors)
```

`gen_ai.system = hermes` on every span; `service.name` defaults to
`agent.coding.hermes`. Spans and dashboard log records are emitted in parallel
(a logs failure never disables spans). Identity (`service.name`, `user.id`, and
any `team.id` / `organization.id` from `OTEL_RESOURCE_ATTRIBUTES`) is placed on
the OTel **resource** so per-user / per-org leaderboard aggregations work.

The mapping logic is pure, Hermes-agnostic, and unit-tested with in-memory
exporters (no Hermes, no network) under `tests/plugins/test_otel_plugin.py`. The
**full reference lives inline, next to the code** — each module's docstring is
the source of truth:

| Module | What its docstring documents |
|--------|------------------------------|
| `emitter.py` | GenAI **span** model (`invoke_agent` / `chat` / `execute_tool` + attributes) |
| `log_emitter.py` | dashboard **log-record** model (`session_start` / `api_response` / `tool_result` — the field contract dashboards aggregate on) |
| `cost.py` | local-model **cost estimation** (model-size → price-tier methodology) |
| `provider.py` | tracer / logger **bootstrap** (genai_otel preferred, plain OTel SDK fallback) |
| `__init__.py` | the only Hermes-coupled module: `register(ctx)` + lifecycle hooks |
| `plugin.yaml` | manifest |

## Advanced — GPU / CO2 / eval metrics

Installing `genai-otel-instrument` instead of the plain OTel SDK unlocks on-prem
**GPU / energy / CO2** metrics and an inline **eval suite** (PII, toxicity, bias,
prompt-injection, restricted-topics, hallucination) — scored in-process, on-prem,
with no data leaving the host, no SaaS backend, and no separate eval service.
It is driven entirely by `genai_otel`'s own `GENAI_*` environment variables;
**no plugin code change is required**.

→ See **[ADVANCED.md](./ADVANCED.md)** for the full setup: why `genai_otel` vs a
vanilla OTel SDK (with a capability comparison), the install extras, the
environment recipe, and exactly what attributes/metrics you get.
