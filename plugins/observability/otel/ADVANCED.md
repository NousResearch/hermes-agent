# Advanced setup — GPU / CO2 / eval metrics (`genai_otel`)

> Optional deep-dive for the [`observability/otel`](./README.md) plugin. The
> README quickstart gets you GenAI-convention spans + dashboard log records
> against any OTLP backend. This guide covers the on-prem **GPU / energy / CO2**
> metrics and the **eval / guardrail** suite you unlock by routing through
> `genai-otel-instrument`. Everything here is opt-in and degrades gracefully if
> a dependency is missing.

When `genai-otel-instrument` is installed the plugin routes through it (see
`provider.py`), which unlocks on-prem **GPU/energy/CO2** metrics and a
**Galileo-equivalent eval suite** (PII, toxicity, bias, prompt-injection,
restricted-topics, hallucination). **No plugin code change is required** — these
are driven entirely by `genai_otel`'s own `GENAI_*` environment variables, read
inside `genai_otel.instrument()` (which the plugin already calls).

```bash
pip install genai-otel-instrument
# …or, for the full on-prem story (GPU/CO2 + PII/toxicity/bias/injection/… eval):
pip install "genai-otel-instrument[gpu,evaluation]"
```

The plugin auto-detects and prefers `genai_otel` when present; otherwise it
falls back to a plain OTel SDK.

## Why `genai_otel` (vs a vanilla OTel SDK or other GenAI instrumentors)

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

## 1. Install the extras

```bash
pip install "genai-otel-instrument[gpu,evaluation]"   # GPU: nvidia-ml-py + codecarbon; eval: presidio + spaCy + detoxify
python -m spacy download en_core_web_lg                # presidio PII NER model (compliance-grade PERSON/LOCATION/NRP)
```

## 2. Environment recipe

Set these alongside the `HERMES_OTEL_*` vars from the README (in `~/.hermes/.env`
or the process environment, **before** Hermes launches):

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

## 3. What you get (verified 2026-06-18 against `genai-otel-instrument` 1.3.3)

- **Span attributes** (every captured turn): `evaluation.pii.{prompt,response}.*`
  (detected, entity_types, *_count, score, redacted), `evaluation.toxicity.*`
  (score + categories — e.g. toxicity 0.93 / insult 0.84), `evaluation.bias.*`,
  `evaluation.hallucination.*` (score, claims, citations, hedge_words),
  `evaluation.prompt_injection.*`, `evaluation.restricted_topics.*`, plus
  `gen_ai.usage.cost.*` (model-size pricing for local Ollama models).
- **Metrics**: GPU (`utilisation / power / memory / temperature`), energy and
  CO2 (`co2_emissions_gco2e`, `energy_consumed_kwh`, `power_consumption_watts`),
  and eval score gauges — all tagged with `service.name = agent.coding.hermes`.

## 4. How it works / gotchas

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
