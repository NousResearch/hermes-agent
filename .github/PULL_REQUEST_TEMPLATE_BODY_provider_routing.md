# OpenRouter provider_routing: pass unknown keys through to the provider object

## Summary

Unrecognized keys in the config `provider_routing` block are now forwarded
verbatim into the request's OpenRouter `provider` preference object, so
current and future OpenRouter routing fields (`zdr`, `quantizations`,
`max_price`, `preferred_min_throughput`/`preferred_max_latency`,
`allow_fallbacks`, `enforce_distillable_text`, …) work without a Hermes
release per new field.

Motivating case: **Zero Data Retention**. Today a user who needs ZDR
routing has no way to ask for it — the key is silently dropped and their
prompts flow to whichever endpoint OpenRouter picks. With passthrough,
`zdr: true` in config.yaml reaches OpenRouter and enforces ZDR-only
routing (fail-closed: OpenRouter errors rather than degrading to a
data-retaining provider).

## Design

- Typed keys (`sort`, `only`, `ignore`, `order`, `require_parameters`,
  `data_collection`) keep their existing attribute semantics and **win on
  conflict** — passthrough never overrides them (DEBUG-logged).
- Each forwarded key logs a `WARNING` so typos surface immediately (a
  mistyped `zdr` must never route silently to a non-ZDR provider).
- Passthrough is **scoped to `extra_body["provider"]`** — nothing else in
  the request body is touched, and prompt-cache invariants hold (the merge
  happens at request build from static config, not mid-conversation).
- **OpenRouter-scoped**: the Nous profile is unaffected (cf. #89430 — the
  Nous API 400s on unknown provider keys).
- Wired across all agent-creation sites: CLI, gateway (messaging +
  background tasks), TUI/desktop, subagent delegation, and background
  review forks (routing pins preserve prompt-cache locality).

Full docs with config examples: `docs/provider_routing_passthrough.md`
(added in this PR).

## Testing

- New behavior-contract tests
  (`tests/providers/test_provider_routing_passthrough.py`): extract
  semantics, typed-wins-on-conflict, malformed-key handling, build-path
  merge, empty/no-op cases.
- E2E with real imports against a temp `HERMES_HOME`: config
  `{sort: throughput, zdr: true, max_price: {...}}` → built provider
  object `{zdr: true, max_price: {...}}`, typed `only` wins.
- Full `tests/agent` + `tests/providers` suites pass.
- Live-verified: `zdr: true` in config.yaml routes OpenRouter requests to
  ZDR-capable providers only (DeepInfra for GLM/V4 models), HTTP 200.

## Related

- Complementary to **#68679** (request-time ZDR via policy-aware catalog +
  Desktop toggle): that is the ZDR *feature*; this is the generic
  *transport* that also carries every other OpenRouter provider field. My
  usage notes/endorsement: https://github.com/NousResearch/hermes-agent/pull/68679#issuecomment-5527160159
- Retires the one-off field PRs: **#91800** (quantization),
  **#72102 / #72118 / #94742** (max_price).
- Context: #17247 (closed in favor of #68679), #32757 (zdr request),
  #89430 / #99830 (Nous 400s on unknown provider keys — scoping rationale).
- Precedent: Nanocoder's typed-fields + `extraBody` shallow-merge.
