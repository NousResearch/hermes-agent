# OpenAI Service Tier Support for Hermes

Date: 2026-04-04

## Goal

Hermes should support OpenAI `service_tier` as a normal Hermes feature on
Responses API requests.

The important part is not just "make `priority` work." The important part is
to make it work in a way that matches how Hermes already handles provider
selection, config, request shaping, and downstream integrations.

If we do this well, a downstream system such as Paperclip can turn on
`priority` by writing ordinary Hermes `config.yaml` values. It should not need
to carry a hardcoded patch, a wrapper-only environment variable, or a second
coordination layer around Hermes.

## Short Recommendation

The clean architecture is:

- Add `model.request_options.service_tier` to Hermes config.
- Parse and normalize it once in `hermes_cli/runtime_provider.py`.
- Create a real Codex/Responses adapter module, probably
  `agent/codex_responses_adapter.py`.
- Move Responses request validation and request-body shaping into that module.
- Make both `run_agent.py` and `agent/auxiliary_client.py` call the same shared
  builder instead of each shaping their own Responses payload.
- Keep the feature opt-in.
- If the option is unset, Hermes should send no `service_tier` field.
- Only inject the field on known OpenAI Responses routes.
- Record the requested tier and the applied tier when the backend returns it.

That is the smallest design that is clean, Hermes-native, and easy to consume
from other systems.

## Why This Is The Right Shape

Hermes already has a clear pattern for model routing:

- `hermes_cli/config.py` owns the persisted config shape.
- `hermes_cli/runtime_provider.py` resolves provider, base URL, API mode, and
  credentials for CLI, gateway, cron, and helpers.
- `run_agent.py` executes the main request path.
- `agent/auxiliary_client.py` handles independent lightweight model calls.
- `agent/anthropic_adapter.py` shows the intended direction for
  provider-specific request logic: isolate it instead of scattering it across
  the runner.

`service_tier` should fit into that same structure.

It should not be:

- a Paperclip-only feature
- a special environment variable that bypasses config
- a hardcoded `priority` branch inside `run_agent.py`
- a duplicated patch in both the main agent path and the auxiliary path

That kind of patch works for a one-off proof. It is the wrong permanent shape
for a fork.

## What Hermes Looks Like Today

Today the relevant pieces are:

- `hermes_cli/config.py`
  - This is the main persisted config surface for model settings.
  - Hermes already stores `provider`, `default`, `base_url`, and `api_mode`
    under the model configuration.
- `hermes_cli/runtime_provider.py`
  - This already resolves the active runtime provider for CLI, gateway, cron,
    and helpers.
  - It is the right place to normalize any provider-specific request options.
- `run_agent.py`
  - This already knows how to run `codex_responses`.
  - `_preflight_codex_api_kwargs()` currently hardcodes the allowlist for the
    Responses request body.
  - `_run_codex_stream()` and `_run_codex_create_stream_fallback()` send the
    actual requests through `client.responses.*`.
- `agent/auxiliary_client.py`
  - This has a second Codex-specific request adapter that builds a Responses
    payload for auxiliary calls.
  - That is already a duplication seam.
- `agent/anthropic_adapter.py`
  - The module comment explicitly says it follows the same pattern as a
    codex-responses adapter, but Hermes does not currently have that adapter as
    a real shared module.
- `hermes_state.py`
  - Hermes already persists billing and usage metadata.
  - It does not currently persist service-tier metadata.

The main architecture problem is simple:

- Hermes has the right config and routing layers already.
- It does not yet have one shared Responses adapter surface.
- Because of that, `service_tier` would be easy to patch in the wrong places
  and hard to keep correct over time.

## Recommended Config Surface

The recommended config shape is:

```yaml
model:
  provider: openai-codex
  default: gpt-5.4
  api_mode: codex_responses
  request_options:
    service_tier: priority
```

This is the best fit for Hermes as it exists today.

Why this shape is better than a new `providers:` tree:

- Hermes already treats `model` as the persisted home for the active inference
  route.
- `service_tier` is a request policy for the active route.
- Adding one small `request_options` block is much cheaper than introducing a
  second provider-config architecture for one field.
- This leaves room for future transport-level request fields without bloating
  the top-level model block.

### Semantics

- Unset:
  - Hermes does not send a `service_tier` field at all.
  - This preserves today's behavior exactly.
- `auto`:
  - Hermes sends `service_tier: "auto"`.
- `default`:
  - Hermes sends `service_tier: "default"`.
- `priority`:
  - Hermes sends `service_tier: "priority"`.
- `flex`:
  - Hermes sends `service_tier: "flex"`.

I would keep the field optional and default it to unset, not to `priority`.

That matters because this should be an opt-in billing and latency decision, not
an invisible change in Hermes defaults.

## Compatibility Rules

Hermes should not send `service_tier` on every OpenAI-compatible request just
because the field exists in config.

The safe rule is:

- Only inject `service_tier` when `api_mode == "codex_responses"`.
- Only inject it for known OpenAI-backed Responses routes.
- Start with the routes we actually understand:
  - `openai-codex`
  - direct `api.openai.com` Responses routes
- Do not assume third-party OpenAI-compatible proxies support the field.

That means the feature should be gated by the resolved runtime route, not just
by the configured provider name.

If the user sets `service_tier` on a route that Hermes does not know supports
it, Hermes should:

- keep the config value
- warn clearly in config check or debug logs
- skip sending the field on that route

It should not silently send the field to every custom endpoint and hope for the
best.

## Proposed Internal Types

Hermes should treat this as structured request metadata, not as an ad hoc
string passed around by hand.

Suggested type:

```python
ServiceTier = Literal["auto", "default", "priority", "flex"]

@dataclass(frozen=True)
class ModelRequestOptions:
    service_tier: ServiceTier | None = None
```

Suggested resolved runtime shape:

```python
{
    "provider": "...",
    "api_mode": "...",
    "base_url": "...",
    "api_key": "...",
    "request_options": {
        "service_tier": "priority",
    },
}
```

The key point is that `runtime_provider.py` should return normalized request
options alongside the provider route, so the rest of Hermes does not have to
re-parse config or guess compatibility rules.

## Proposed Module Boundary

Create a real shared Responses adapter module:

- `agent/codex_responses_adapter.py`

This module should own:

- `service_tier` validation
- compatibility checks for known supported routes
- request-body construction for Responses API calls
- the allowlist for supported request fields
- preflight normalization
- response metadata extraction, including any returned service-tier readback

The core design rule is:

- `run_agent.py` should not own the Responses request schema
- `agent/auxiliary_client.py` should not own a second copy of the same schema

They should both call the adapter.

## Main Agent Path

For the main agent path, the flow should be:

1. `config.yaml` stores `model.request_options.service_tier`
2. `hermes_cli/runtime_provider.py` parses and normalizes it
3. `AIAgent` receives normalized request options as part of runtime resolution
4. `agent/codex_responses_adapter.py` builds the final request body
5. `run_agent.py` executes the request without re-implementing field rules

In practice, this means:

- shrink `_preflight_codex_api_kwargs()` or replace it with a shared adapter
  call
- keep `run_agent.py` responsible for orchestration, retries, and streaming
- move request-shape ownership out of `run_agent.py`

That is both cleaner and closer to the adapter pattern Hermes already uses for
Anthropic.

## Auxiliary Client Path

The auxiliary path needs the same treatment.

Today `agent/auxiliary_client.py` has a Codex adapter shim that constructs its
own Responses payload. That is exactly the kind of duplication that will drift.

The right design is:

- keep the auxiliary client's public behavior the same
- replace its hand-built Responses request construction with the shared adapter
- let the auxiliary path opt into `service_tier` only when the resolved
  auxiliary route supports it

I would not build a second auxiliary-only service-tier system.

If later Hermes wants task-specific service tiers for vision, summarization, or
compression, that can be added as a separate config extension. It should not
block the first clean implementation.

## Validation and Error Behavior

Hermes should validate this in two places.

### Config-time validation

`hermes config check` should reject invalid values and warn on unsupported
route combinations.

Examples:

- valid:
  - `priority`
  - `auto`
- invalid:
  - `fast`
  - `high`
- warn and ignore on unsupported route:
  - `service_tier: priority` with `api_mode: chat_completions`
  - `service_tier: priority` on an unknown custom provider

### Runtime validation

At request build time, the shared adapter should decide whether the resolved
route supports the field.

If it does:

- include the field in the request body

If it does not:

- omit the field
- log one clear debug message

The runtime should not crash just because a stale config value followed a model
switch onto an unsupported provider.

## State and Observability

`service_tier` is request metadata, not model identity. Hermes should treat it
that way.

The minimum useful observability is:

- request debug dumps include the field when present
- the runner records the requested service tier on the current request
- if the backend returns the applied tier, Hermes records that too

I would not make the session database the first or only source of truth for
this. Session rows are coarse-grained. A session can contain more than one
request, and future fallbacks or overrides may produce mixed service tiers
inside one session.

The practical design is:

- primary truth: per-request debug and trajectory metadata
- optional convenience summary: last requested tier and last applied tier on
  the session row

If we do add session-level fields, use dedicated names like:

- `requested_service_tier`
- `applied_service_tier`

Do not overload billing fields to carry this.

## CLI and Config UX

The first version does not need a new slash command.

The clean first version is:

- config-only
- visible in `hermes config`
- validated in `hermes config check`
- preserved by model-switch flows when the provider route stays compatible

That means:

- `hermes_cli/config.py` needs the new config field
- `hermes_cli/main.py` and `hermes_cli/model_switch.py` should preserve
  `model.request_options` when switching within the same compatible provider
  family

I would avoid adding a special `/service-tier` command until the core adapter
and config shape are stable.

## File Plan

These are the files I would expect to touch in the fork.

New file:

- `agent/codex_responses_adapter.py`
  - shared request builder
  - field validation
  - compatibility rules
  - response metadata extraction

Existing files:

- `hermes_cli/config.py`
  - add `model.request_options`
  - validate allowed values
  - support migration for existing configs cleanly
- `hermes_cli/runtime_provider.py`
  - parse and normalize `service_tier`
  - return it as part of resolved runtime config
- `run_agent.py`
  - stop owning the Responses request schema directly
  - call the shared adapter
  - record requested and applied service-tier metadata
- `agent/auxiliary_client.py`
  - replace local Responses request shaping with the shared adapter
- `hermes_cli/main.py`
  - preserve config on model-switch and setup paths
- `hermes_state.py`
  - optional session-level summary fields if we decide they are worth storing
- `tests/...`
  - config validation
  - runtime provider normalization
  - request builder behavior
  - auxiliary path behavior

## Recommended Rollout

### Phase 1: Internal plumbing

- Add config support
- Add runtime normalization
- Add the shared Responses adapter
- Route `run_agent.py` through it

### Phase 2: Remove duplication

- Route `agent/auxiliary_client.py` through the same adapter
- Make request dumps and debug output show the field cleanly

### Phase 3: Nice operator behavior

- Add config-check warnings for unsupported routes
- Preserve the field across compatible model switches
- Add docs

### Phase 4: Optional visibility improvements

- Record last requested and applied tiers in session summaries if useful
- Surface the active tier in status or usage views if we decide operators need
  it

## Downstream Integration

This design is good for downstream systems because it keeps Hermes in charge.

For example, a downstream system should be able to write:

```yaml
model:
  provider: openai-codex
  default: gpt-5.4
  request_options:
    service_tier: priority
```

and stop there.

That is exactly what we want.

The downstream system should not have to:

- patch `run_agent.py`
- patch `agent/auxiliary_client.py`
- inject a special wrapper-only flag
- teach users a second control plane just to get priority processing

## Alternatives I Would Reject

### 1. Hardcode `priority` in `run_agent.py`

This is fine for a local proof. It is not a real feature design.

Why I would reject it:

- it bakes billing policy into code
- it does not match Hermes config-driven behavior
- it leaves the auxiliary path inconsistent

### 2. Add a Paperclip-only environment variable

This solves the immediate integration problem but makes Hermes less coherent.

Why I would reject it:

- it bypasses Hermes config
- it creates wrapper semantics instead of a Hermes feature
- it is harder to reason about in CLI, gateway, cron, and tests

### 3. Introduce a new top-level `providers:` config tree just for this

This is heavier than necessary for one request option.

Why I would reject it:

- it creates a second config architecture for the main model path
- it is more work to document, migrate, and preserve through model switches
- Hermes already has a natural home for this under `model`

## Final Recommendation

If we want to carry this in a Hermes fork, the best long-term design is:

- a first-class Hermes config field
- a shared Codex/Responses adapter module
- one normalized request-options path through runtime provider resolution
- one request builder used by both the main runner and auxiliary clients
- an opt-in `service_tier` field that is only sent on known supported routes

That gives us a fork that is small, understandable, and easy to upstream later.

It also gives downstream systems a clean story:

- write normal Hermes config
- let Hermes own the request semantics
- avoid permanent wrapper patches
