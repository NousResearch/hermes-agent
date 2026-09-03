# Docs: OpenRouter provider_routing passthrough

The `provider_routing` block in `config.yaml` now supports **general
passthrough**: any key that is not one of the six typed preference keys
(`sort`, `only`, `ignore`, `order`, `require_parameters`,
`data_collection`) is forwarded verbatim into the OpenRouter request's
`provider` object.

## Why

OpenRouter's provider preference object gains fields faster than any
wrapper can type them: `zdr` (Zero Data Retention), `quantizations`,
`max_price`, `preferred_min_throughput`, `preferred_max_latency`,
`allow_fallbacks`, `enforce_distillable_text`, provider-specific
`headers`, and more (see the
[provider routing docs](https://openrouter.ai/docs/features/provider-routing)).
Until now Hermes silently dropped any such key, so users could not opt
into routing guarantees — most notably ZDR — without code changes.

With passthrough, current and future OpenRouter routing fields work
without a Hermes release per new field.

## Configuration

```yaml
# ~/.hermes/config.yaml
provider_routing:
  # Typed keys keep their existing attribute-based semantics.
  sort: throughput
  # Everything else is passed through verbatim:
  zdr: true                      # Zero Data Retention routing
  quantizations: [fp8, fp16]     # endpoint quantization filter
  max_price:
    prompt: 0.5                  # USD per 1M prompt tokens
    completion: 1.5              # USD per 1M completion tokens
  preferred_min_throughput: 50   # tokens/s per 1M-prompt endpoints
  allow_fallbacks: false         # strict: only listed providers
```

### Semantics

- **Typed keys win on conflict.** A passthrough value for a typed key is
  ignored (logged at `DEBUG`): the typed config keys remain the
  authoritative spelling of those six preferences.
- **Unknown keys warn once per request-build**: each unrecognized key
  logs a `WARNING` (`provider_routing: forwarding unrecognized key ...`)
  so typos surface immediately — a mistyped `zdr` should never route
  silently to a non-ZDR provider.
- **Scoped to the provider object.** Only `extra_body["provider"]` is
  touched; nothing else in the request body changes.
- **Fail-closed on ZDR.** With `zdr: true`, OpenRouter routes only to
  ZDR-capable endpoints and errors if none exist rather than silently
  degrading to a data-retaining provider.

### Scope

OpenRouter only. The typed keys continue to work for the Nous profile;
passthrough keys are an OpenRouter-specific extension of the provider
preference object and are guarded separately (cf. #89430: the Nous API
400s on unknown provider keys).

### What it covers (OpenRouter provider-object fields)

- `zdr` — zero-data-retention routing (this PR's motivating case)
- `quantizations`, `max_price` — replace one-off PRs #91800 and #72102/#72118/#94742
- `preferred_min_throughput`, `preferred_max_latency` — SLA-style routing
- `enforce_distillable_text`, `partition`, `allow_fallbacks` — policy controls
- any future field OpenRouter adds — zero Hermes-side changes needed

## Precedent

- [Nanocoder](https://github.com/Nanocoder/AI-CLI) OpenRouter provider:
  typed fields + shallow-merge `extra_body` passthrough.
- #68679 adds ZDR via a policy-aware catalog + Desktop toggle —
  complementary: it is the ZDR *feature*; this is the generic *transport*
  that would also have carried it (and retires the one-off field PRs).

## Related PRs / issues

- #68679 — request-time ZDR enforcement (open, complementary)
- #17247 — closed in favor of #68679
- #91800 — quantization passthrough (one-off)
- #72102 / #72118 / #94742 — max_price passthrough attempts (one-offs)
- #89430 / #99830 — Nous API 400s on unknown provider keys (why this is
  OpenRouter-scoped)
- #32757 — original zdr preference request
