# Usage formats and limits

The helper reads one JSON object per non-blank line. These are illustrative
fixtures, not live measurements. A full API response is unnecessary; export its
usage object only. Use one shape per input file.

## Chat Completions (`--format chat`)

```json
{"prompt_tokens": 2000, "prompt_tokens_details": {"cached_tokens": 1500}}
```

Prompt totals include cached input. Fallback cache-read paths, in order, are
`cache_read_input_tokens`, `prompt_cache_hit_tokens`, and `cached_tokens`.
Fallback prompt totals use `input_tokens` if `prompt_tokens` is absent.
Cache-write paths are `prompt_tokens_details.cache_write_tokens`,
`prompt_tokens_details.cache_creation_input_tokens`, `cache_creation_input_tokens`,
and `cache_write_tokens`. A missing write counter does not prevent the read ratio
because the prompt total is already inclusive.

## Responses (`--format responses`)

```json
{"input_tokens": 2000, "input_tokens_details": {"cached_tokens": 1500}}
```

Cache writes use `input_tokens_details.cache_write_tokens`, then
`input_tokens_details.cache_creation_tokens`. The prompt total is inclusive.

## Native Anthropic Messages (`--format anthropic`)

```json
{"input_tokens": 500, "cache_read_input_tokens": 1500, "cache_creation_input_tokens": 0}
```

Uncached `input_tokens` excludes cache reads and writes. All three counters must
be present to construct a denominator. Missing write telemetry therefore makes
this call unmeasurable even if a read counter is present. An Anthropic-style model
through a Chat proxy is not necessarily native Messages usage: select the shape
of the exported counters, not the model vendor's name.

## Report interpretation

- Each selected counter must be a non-negative JSON integer (not a boolean,
  float, string, or negative value). Missing and null fields remain unknown.
- The first present, non-null candidate wins, including zero. Hermes's runtime
  normalizer uses first-nonzero fallback; this diagnostic deliberately preserves
  explicit zero so a conflicting alias cannot turn it into a claimed hit.
- For inclusive totals, reported cache buckets exceeding the total are rejected:
  the API shape or exported values need investigation, not silent clamping.
- The aggregate denominator excludes calls missing either prompt totals or read
  counters. Zero-total calls have no individual ratio and add zero to totals.
- The output includes numeric buckets only; arbitrary input fields are discarded.
  Malformed input fails the whole report, identifies the line, and does not echo
  sensitive content. Do not use this as a parser for untrusted, huge files: it
  retains one numeric entry per call in memory.
- Coverage and token-weighted reuse are not billing estimates or proof of cache
  retention. The helper does not infer session boundaries, model changes,
  provider-specific anomalies, or idle expiry from counters alone.

## Product evidence

Verify these against the installed Hermes version when diagnosing behavior:

- `agent/usage_pricing.py::normalize_usage`: Chat, Responses, and native
  Anthropic usage shapes; native input excludes cached buckets. Also documents
  the MiniMax-M3 native cache-read-floor caveat.
- `agent/turn_usage.py::record_response_usage`: canonical counters and API-call
  logging, including cache reads/writes and serving-provider metadata.
- `agent/prompt_cache_scope.py`: logical conversation scope, compression lineage,
  and isolation across new conversations/forks.
- `plugins/model-providers/openrouter/__init__.py::build_extra_body`: sends sticky
  `session_id` alongside configured provider preferences. It does not establish
  that `provider.order` always disables cache affinity.
- `agent/prompt_caching.py`: route-aware marker support and TTL policy.

The architecture page already exists at
https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching.
Consult current provider documentation for size thresholds, retention, and prices;
those values are deliberately not embedded in this skill.
