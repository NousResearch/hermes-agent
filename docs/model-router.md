# Hermes staged model router

Hermes includes an opt-in, turn-boundary router ported from the architecture of
[`pi-smart-router`](https://github.com/beettlle/pi-smart-router). It keeps the
existing `Candidate`/`route_turn` API and adds a staged `RouterPipeline`.

## Safety invariants

- `off` is the default and preserves the configured model.
- `suggest` computes and records a recommendation but retains the current model.
- `auto` selects once before `AIAgent` construction; it never changes the model
  during the tool loop, preserving prompt-cache and tool-history stability.
- Candidates are explicit declarations. Provider resolution stays in the
  gateway; failures and stage exceptions fall back safely.
- No network, credentials, embedding download, or filesystem probe is used by
  default. Local routing is explicitly opt-in.
- Telemetry stores no prompt text, only a bounded djb2 hash and prompt length.

## Pipeline

The adapted pipeline is sequential with early exit:

1. `force_model` — explicit override
2. `loop_escalation` — repeated tool failures can escalate a pinned session
3. `context_fit` — remove models whose declared window cannot fit the request
4. `low_intensity` — cheap adequate selection for low-intensity turns
5. `session_pin` — hold a warm session pin; break on compaction, overflow, or
   cache-breakeven / score-margin conditions
6. `triage` — deterministic trivial/complex/ambiguous classification with
   adversarial sanitization and cyclomatic scan
7. `local_zero` — optional LM Studio/Ollama health probe
8. `triage_cloud_fallback` — trivial turns use the economical tier
9. `hydra_match` — deterministic requirement vector, capability shortfall gate,
   and multi-objective scoring
10. `safe_default` — healthy configured safe tier
11. `context_overflow` — largest fitting fallback, otherwise current model

Pi-only subsystems are deliberately not copied: hardware/battery probe,
speculative prewarm, adaptive reasoning effort, workload heat, isotonic
calibration, price fetching, planning delegate, Gemini history guard, and
provider stream failover.

## Configuration

Use the normal configuration writer, not manual edits:

```bash
hermes config set model_router.mode suggest
```

Example:

```yaml
model_router:
  mode: suggest                 # off | suggest | auto
  candidates:
    - model: gpt-5.6-luna
      provider: openai-codex
      tier: economical
      reasoning: true
      vision: true
      context_window: 272000
      quality: 0.90
      cost: 0.20
      cost_per_1m: 0.20          # optional; only for cache economics
      est_latency_ms: 500
      verbosity: 1.0
    - model: gpt-5.6-sol
      provider: openai-codex
      tier: frontier
      reasoning: true
      vision: true
      context_window: 272000
      quality: 1.00
      cost: 0.70
  frugality:
    lambda_cost: 0.5
    lambda_latency: 0.1
    lambda_verbosity: 0.15
  loop_escalation: {threshold: 3}
  session_pin: {enabled: true, dwell_turns: 3, switch_margin: 0.25}
  safe_tier: economical
  local_zero: {enabled: false, endpoints: [], model: null, timeout_ms: 1500}
  hydra: {enabled: true, embeddings: false}
  telemetry: {enabled: true, db_path: null}
```

Tier aliases `zero-tier`, `economical-cloud`, and `frontier-cloud` are accepted
for compatibility with Pi. `quality` and `cost` are routing heuristics, not
billing guarantees. Keep the pool small and verify exact provider/model IDs
before switching from `suggest` to `auto`.

## Inspection

```bash
hermes router status
hermes router history --limit 20
hermes router history --session SESSION_ID
hermes router stats
```

The default database is `$HERMES_HOME/state/model_router/router.db`; it is
SQLite WAL mode, owner-only, and pruned to roughly 10,000 routing rows.
Telemetry and state failures are non-fatal.

## Development

The compatibility tests remain in `tests/agent/test_model_router.py`. Staged
pipeline tests belong under `tests/agent/model_router/`. Run the legacy gate
and package smoke tests with:

```bash
pytest tests/agent/test_model_router.py tests/agent/model_router -q
```
