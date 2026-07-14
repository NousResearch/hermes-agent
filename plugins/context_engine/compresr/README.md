# context_engine/compresr

**Compaction-time context compression** for Hermes, powered by
[Compresr](https://compresr.ai) (YC W26).

A drop-in replacement for the built-in `compressor` context engine. Instead of
asking an auxiliary LLM to write an abstractive summary of the middle of the
conversation when the context-window threshold is crossed, this engine sends
those turns to Compresr's query-specific compression API (`latte_v1`/`latte_v2`),
which scores spans against the current goal and keeps only the answer-bearing
ones — in sub-second time rather than a multi-second LLM summarization.

## Design

It subclasses `ContextCompressor` and overrides **only** the "compress the
middle window" step (`_generate_summary`) plus `update_model` (to recompute
token budgets once the real model is known). Everything else — old-tool-result
pruning, head/tail protection, boundary alignment, tool_call/tool_result
sanitization, historical-media stripping, anti-thrashing accounting — is
inherited verbatim. So an A/B against the built-in engine is a clean comparison
of the *core compaction strategy* only.

The query handed to Compresr is Hermes's own derived focus topic
(`_derive_auto_focus_topic`), so compression is goal-aware. Because Compresr is
an external service, this engine defaults to preserving the current transcript
when summary generation fails instead of dropping middle turns with a
deterministic placeholder handoff. A short cooldown avoids hammering a failing
endpoint. During a sustained Compresr outage, long conversations may stop
compacting until the API recovers or the user starts a fresh session; this is a
deliberate fail-open tradeoff to avoid losing context.

## Data sent to Compresr

When enabled, Hermes sends the middle conversation window selected for
compaction to the configured Compresr API endpoint. Do not enable this engine
unless that third-party processing is acceptable for your deployment.

## Activation

```yaml
# ~/.hermes/config.yaml
context:
  engine: compresr
```
```bash
# ~/.hermes/.env
COMPRESR_API_KEY=cmp_...
```
Or run `hermes setup` and choose a Compresr option under **Context Compression**.

## Configuration

| env / `compresr:` key | default | meaning |
|---|---|---|
| `COMPRESR_API_KEY` | — | **required** `cmp_…` key, read from `.env` only |
| `COMPRESR_BASE_URL` / `base_url` | `https://api.compresr.ai/api` | API base |
| `COMPRESR_MODEL` / `model` | `latte_v2` | `latte_v1` \| `latte_v2` |
| `COMPRESR_TARGET_RATIO` / `target_ratio` | — | override (Compresr Nx / removal semantics) |
| `COMPRESR_TIMEOUT` / `timeout` | `60` | request timeout (s) |

`target_compression_ratio` semantics: a value in (0, 1] is a *removal* fraction
(0.8 → remove ~80%); a value > 1 is an Nx target (5 → ~1/5). When unset, Hermes's
keep-fraction `f` is mapped to the Nx factor `1/f` (0.2 keep → 5×).

## Relationship to `tool_output_compresr`

This handles compaction-time compression of the conversation. Its per-turn
complement, [`tool_output_compresr`](../../tool_output_compresr), compresses
large *tool outputs* as they arrive. They compose; enable either or both.

## Tests

`tests/test_compresr_context_engine.py` — contract test pinning the
`_generate_summary` override signature and the success → prefixed-body /
error → `None` fallback round-trip, so an upstream refactor of the parent's
private seam fails loudly in CI rather than silently degrading.
