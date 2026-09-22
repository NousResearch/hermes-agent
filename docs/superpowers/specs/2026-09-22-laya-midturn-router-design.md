# Laya Mid-Turn Model-and-Effort Router Design

## Goal

Add a disabled-by-default, fail-closed mid-turn router that may choose an allowlisted model-and-reasoning route only after a completed tool round. It must not rewrite system prompts, tools, transcript history, or profile/session state.

## Scope and safety boundary

The existing `model_router` remains disabled by default and continues to own all controller I/O. This change adds `model_router.mid_turn.enabled`, also defaulting to `false`. Mid-turn routing runs only after `run_tool_round()` has completed, persisted, and returned `continue`; it never runs before the first model request, after failed/policy-halted tools, or inside retry/fallback handling.

The controller receives a bounded, redacted state object with the original goal and the most recent completed tool-result text. It receives route IDs and descriptions only. It never receives provider names, model IDs, credentials, raw transcript objects, tool schemas, or unbounded output. Hermes resolves the controller's chosen ID through the existing configured route allowlist.

`planning` and `exception` controller choices are not applied directly at the mid-turn boundary. They are mapped to `mid_turn.strong_route` (required to name an allowlisted route); controller failure, malformed values, low confidence, and route incompatibility leave the active route unchanged. There is no provider fallback.

## Safe opt-in subset

A single live `AIAgent` owns a provider-specific client, transport, prompt-cache policy, compression state, fallback state, and a system prompt whose bytes may be model-specific. `switch_model()` intentionally changes persistent state and invalidates the cached prompt; using it during a tool loop would violate the core caching invariant.

Therefore this implementation supports only a same-runtime mid-turn change: the selected route must resolve to the active concrete provider, base URL, API mode, API-key identity, and credential-pool identity. Within that runtime, the loop changes only the outbound model ID, selected reasoning configuration, and selected request overrides for the next model request. Provider/transport/credential changes are rejected before mutation. The previous model/effort/overrides are restored in a `finally` at the turn boundary, so gateway-cached agents do not carry a router choice to a later user turn.

This deliberately does not claim seamless cross-provider switching. Cross-provider mid-turn routing needs a transaction-safe runtime/client snapshot and a way to prove the cached system prompt is compatible with both route identities; that is follow-up work.

## Configuration

```yaml
model_router:
  enabled: false
  routes: {}
  mid_turn:
    enabled: false
    tool_outcome_max_chars: 2048
    strong_route: exception
```

The root `model_router.enabled` gate remains required. Empty/invalid values fail closed. The new nested options are normal config data, not environment variables and are not activated by this change.

## Data flow

1. A model emits tool calls.
2. Hermes persists and executes the tool calls normally.
3. If the round completes, the loop calls `maybe_apply_midturn_route()` before the next iteration begins.
4. The helper collects only current-round `tool` content, redacts and bounds it, then asks the local loopback controller for an allowlisted ID.
5. Hermes validates confidence and target resolution; a strong/exceptional decision is pinned to `strong_route`.
6. Hermes applies a route only when the resolved provider/base URL/API mode equals the current runtime. The next API call uses that route's model and reasoning effort.
7. `run_conversation()` restores the saved request-local model/effort/overrides in `finally`, before the cached agent can serve another turn.

## Invariants and tests

Behavior tests will prove: disabled mode makes no controller call; the controller state includes bounded redacted completed tool output; a routine compatible route changes the next request model and effort; planning/exception and failures pin or fail closed without arbitrary fallback; incompatible provider/API targets leave runtime unchanged; and one end-to-end tool round preserves the original system prompt bytes, tool schema object, transcript role sequence, and restores the cached agent's base model after the turn.

## Non-goals

No production config edit, setup wizard activation, network controller deployment, model catalog assertion, provider credential changes, prompt/tool schema mutation, or arbitrary fallback path is included.
