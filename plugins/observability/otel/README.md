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

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it. It slots in beside `observability/langfuse` and
`observability/nemo_relay` and makes no changes to Hermes core.

## Span model

```
invoke_agent      per Hermes turn   (gen_ai.conversation.id = session id)
  ├── chat        per LLM API call  (tokens, cost, finish reason, TTFT)
  └── execute_tool  per tool call   (gen_ai.tool.name / .type, errors)
```

`gen_ai.system = hermes` on every span; `service.name` defaults to
`agent.coding.hermes`. Prompt/response content is attached **only** when
`HERMES_OTEL_CAPTURE_CONTENT=true` (privacy-gated, off by default).

## Log-record model

Emitted in parallel with the spans (best-effort — a logs failure never disables
spans), to the OTLP **logs** endpoint (`<endpoint>/v1/logs`):

| `event.name`    | Per          | Key attributes |
|-----------------|--------------|----------------|
| `session_start` | Hermes turn  | `session.id`, `model` |
| `api_response`  | LLM API call | `model`, `input_tokens`, `output_tokens`, `cost_usd`, `finish_reason`, `ttft_ms` |
| `tool_result`   | tool call    | `tool_name`, `tool_type`, `status_code`, `decision_type` |

Identity (`service.name`, `user.id`, and any `team.id` / `organization.id` from
`OTEL_RESOURCE_ATTRIBUTES`) is placed on the OTel **resource** so the
per-user / per-org leaderboard aggregations work. Same privacy gate: prompt /
response / tool I/O is attached only with `HERMES_OTEL_CAPTURE_CONTENT=true`.

## Cost estimation

Local backends report no price, so `cost.py` mirrors the
`genai-otel-instrument` methodology: parse the parameter size from the model
name (`llama3.1:8b` → 8 B, `qwen3:0.6b` → 0.6 B, `smollm2:360m` → 0.36 B), map
it to a per-1K-token price tier, and compute
`cost = in_tok/1000 * promptPrice + out_tok/1000 * completionPrice`. It is used
**only** as a fallback when the agent itself reported no cost; an
agent-provided `cost_usd` always wins. When the size can't be inferred, cost is
omitted (no fake `$0`).

## Enable

```bash
# OTel SDK + OTLP HTTP exporter (the only hard dependency)
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

hermes plugins enable observability/otel
```

Optionally install `genai-otel-instrument` to use the richer reference emitter
(adds token/cost/latency conventions and optional on-prem **GPU / energy / CO2**
metrics plus **eval / guardrail** scoring); the plugin auto-detects and prefers
it when present:

```bash
pip install genai-otel-instrument
# …or, for the full on-prem story (GPU/CO2 + PII/toxicity/bias/injection/… eval):
pip install "genai-otel-instrument[gpu,evaluation]"
```

See **[Full setup — GPU / CO2 / eval metrics](#full-setup--gpu--co2--eval-metrics-genai_otel)**
below for the env-var recipe.

Without the OTel SDK the hooks no-op silently — the plugin fails open.

## Configure

All configuration is optional; the plugin works out of the box against a local
collector on `http://localhost:4318`. Set these in `~/.hermes/.env`:

```bash
HERMES_OTEL_SERVICE_NAME=agent.coding.hermes   # service.name (backends group on this)
HERMES_OTEL_ENDPOINT=http://localhost:4318     # OTLP HTTP endpoint
HERMES_OTEL_AGENT_NAME=hermes                  # gen_ai.agent.name
HERMES_OTEL_CAPTURE_CONTENT=false              # attach prompt/response text (privacy-gated)
HERMES_OTEL_USER_ID=                           # stamp user.id on the resource
```

Standard OTel env vars are honoured as fallbacks: `OTEL_SERVICE_NAME`,
`OTEL_EXPORTER_OTLP_ENDPOINT`.

## Full setup — GPU / CO2 / eval metrics (`genai_otel`)

When `genai-otel-instrument` is installed the plugin routes through it (see
`provider.py`), which unlocks on-prem **GPU/energy/CO2** metrics and a
**Galileo-equivalent eval suite** (PII, toxicity, bias, prompt-injection,
restricted-topics, hallucination). **No plugin code change is required** — these
are driven entirely by `genai_otel`'s own `GENAI_*` environment variables, read
inside `genai_otel.instrument()` (which the plugin already calls). Everything
below is opt-in and degrades gracefully if a dependency is missing.

### Why `genai_otel` (vs a vanilla OTel SDK or other GenAI instrumentors)

The plugin works against a **plain OTel SDK** (the fallback) — that alone gives
you GenAI-convention spans, tokens, latency, finish reasons, and the dashboard
log records, portable to any OTLP backend. Routing through
`genai-otel-instrument` adds four signals that are **genuinely unique** — neither
a raw OTel SDK nor the common GenAI instrumentation libraries (OpenLLMetry /
Traceloop, OpenInference / Arize, OpenLIT, Langfuse) emit them:

| Signal | `genai_otel` | Vanilla OTel SDK | Other GenAI libs |
|--------|:---:|:---:|:---:|
| **On-prem GPU metrics** — per-GPU utilization, power, memory, temperature, clocks, throttle state (nvidia-ml-py / amdsmi) | ✅ | ❌ | ❌ |
| **Energy + CO2** — `energy_kwh`, `co2_emissions_gco2e`, region-aware (codecarbon) | ✅ | ❌ | ❌ |
| **Local-model cost** — Ollama / HF / vLLM cost estimated from parameter size when the backend returns no price | ✅ | ❌ (`$0`) | ❌ (priced cloud APIs only) |
| **Inline eval / guardrails** — PII, toxicity, bias, prompt-injection, restricted-topics, hallucination scored as span attributes + metrics at instrumentation time | ✅ in-process | ❌ | ❌ (need a separate eval pipeline) |

Everything runs **in-process and on-prem** — no data leaves the host, no SaaS
observability backend, and no separate eval service. The token / latency / cost
fields stay standard OTel GenAI conventions (portable to any OTLP backend); the
four signals above are extra attributes/metrics a conventions-aware backend can
use or safely ignore. All of it is opt-in and degrades gracefully when a
dependency is missing.

### 1. Install the extras

```bash
pip install "genai-otel-instrument[gpu,evaluation]"   # GPU: nvidia-ml-py + codecarbon; eval: presidio + spaCy + detoxify
python -m spacy download en_core_web_lg                # presidio PII NER model (compliance-grade PERSON/LOCATION/NRP)
```

### 2. Environment recipe

Set these alongside the `HERMES_OTEL_*` vars above (in `~/.hermes/.env` or the
process environment, **before** Hermes launches):

```bash
# --- MANDATORY for eval: detectors score gen_ai.prompt / gen_ai.completion, which the
#     plugin only stamps when content capture is on (genai_otel's own GENAI_ENABLE_CONTENT_CAPTURE
#     is the WRONG lever — it gates genai_otel's instrumentors, not the plugin's spans). ---
HERMES_OTEL_CAPTURE_CONTENT=true

# --- MANDATORY to avoid double-counting: Hermes calls Ollama through the openai SDK, and
#     genai_otel's DEFAULT instrumentor set includes 'openai', which would emit a SECOND
#     span (+ duplicate token/cost metrics) per model call. Replacing the list with a
#     no-op ('mcp') drops 'openai'. GPU/CO2/eval are independent of this list and still run. ---
GENAI_ENABLED_INSTRUMENTORS=mcp

# --- On-prem GPU + energy + CO2 (sustainability story) ---
GENAI_ENABLE_GPU_METRICS=true       # default true — utilisation, power, memory, temp (NVIDIA via nvidia-ml-py)
GENAI_ENABLE_CO2_TRACKING=true      # energy_kwh + CO2 gCO2e via codecarbon
GENAI_CO2_COUNTRY_ISO_CODE=IND      # carbon-intensity region — set your deployment country
GENAI_CODECARBON_LOG_LEVEL=error    # keep codecarbon quiet

# --- Eval / guardrails — scored on every captured prompt + response ---
GENAI_ENABLE_PII_DETECTION=true
GENAI_PII_MODE=redact               # redact the evaluation copy (data-sovereignty); raw gen_ai.prompt still carries text
GENAI_ENABLE_TOXICITY_DETECTION=true
GENAI_ENABLE_BIAS_DETECTION=true
GENAI_ENABLE_PROMPT_INJECTION_DETECTION=true
GENAI_ENABLE_RESTRICTED_TOPICS=true
GENAI_ENABLE_HALLUCINATION_DETECTION=true
```

### 3. What you get (verified 2026-06-18 against `genai-otel-instrument` 1.3.3)

- **Span attributes** (every captured turn): `evaluation.pii.{prompt,response}.*`
  (detected, entity_types, *_count, score, redacted), `evaluation.toxicity.*`
  (score + categories — e.g. toxicity 0.93 / insult 0.84), `evaluation.bias.*`,
  `evaluation.hallucination.*` (score, claims, citations, hedge_words),
  `evaluation.prompt_injection.*`, `evaluation.restricted_topics.*`, plus
  `gen_ai.usage.cost.*` (model-size pricing for local Ollama models).
- **Metrics**: GPU (`utilisation / power / memory / temperature`), energy and
  CO2 (`co2_emissions_gco2e`, `energy_consumed_kwh`, `power_consumption_watts`),
  and eval score gauges — all tagged with `service.name = agent.coding.hermes`.

### 4. How it works / gotchas

- Eval needs **no plugin change**: `instrument()` registers a *global*
  `EvaluationSpanProcessor` + wraps the OTLP exporter with an
  `EvaluationEnrichingSpanExporter` whenever any `GENAI_ENABLE_*_DETECTION` is
  set; the detectors read `gen_ai.prompt` / `gen_ai.completion` off **any** span,
  so they score the plugin's own `invoke_agent` / `chat` spans.
- **detoxify** downloads a ~500 MB model from HuggingFace on first toxicity eval
  — pre-warm on an internet-connected host, or set
  `GENAI_ENABLE_TOXICITY_DETECTION=false` for fully air-gapped installs.
- The eval **score metrics** only reach a Prometheus/Timescale backend if its
  keep-list includes `genai_evaluation_*` (the eval *span attributes* are
  unaffected and are the most complete source).
- **Data sovereignty**: `HERMES_OTEL_CAPTURE_CONTENT=true` exports raw
  prompt/response text on spans; `GENAI_PII_MODE=redact` redacts the eval copy.
  Confirm your PII-before-storage policy covers `gen_ai.prompt` / `gen_ai.completion`
  before enabling content capture in a regulated environment.

## Verify

```bash
hermes plugins list                 # observability/otel should show "enabled"
hermes chat -q "hello"              # then check your OTel backend for an
                                    # "invoke_agent" trace with chat/execute_tool children
```

## Disable

```bash
hermes plugins disable observability/otel
```

## Files

```
plugins/observability/otel/
  __init__.py      Hermes plugin: register(ctx) + lifecycle hooks (the only Hermes-coupled module)
  emitter.py       Hermes-agnostic OTel GenAI span builder
  log_emitter.py   Hermes-agnostic OTel log-record builder (dashboard event model)
  cost.py          local-model cost estimation (model size → price tier)
  provider.py      tracer + logger bootstrap (genai_otel preferred, plain OTel SDK fallback)
  plugin.yaml      manifest
```

The Hermes coupling is isolated to `__init__.py`; the span-mapping
(`emitter.py`), log-record-mapping (`log_emitter.py`), and cost (`cost.py`)
logic is pure and unit-tested with in-memory exporters (no Hermes, no network)
under `tests/plugins/test_otel_plugin.py`.
