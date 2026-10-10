# Bug report and technical specification: auxiliary model aliases

## Summary

Configured `model_aliases` are resolved by the main agent, cron, and delegated-child startup paths, but a task configured as `auxiliary.<task>.provider: auto` previously read `model.default` literally. If `model.default` was an alias, the auxiliary client sent the alias label to the provider as the wire model ID.

This is a functional inconsistency: aliases are user-facing routing abstractions and must never reach a provider request as a model ID.

## Reproduction

```yaml
model:
  provider: openrouter
  default: zdr-flash
model_aliases:
  zdr-flash:
    model: z-ai/glm-5.3-flash
    provider: openrouter
    base_url: https://openrouter.ai/api/v1
auxiliary:
  vision:
    provider: auto
```

Call `vision_analyze`.

### Actual behavior before this change

The automatic auxiliary route selected `openrouter` with model `zdr-flash`. OpenRouter rejected the request with HTTP 400 because `zdr-flash` is not a provider model ID.

### Expected behavior

The auxiliary route resolves the configured alias first, then requests the concrete route: model `z-ai/glm-5.3-flash`, alias endpoint, and alias credential scope. URL-bearing aliases may be represented internally as `custom`; the provider label is not the contract. The concrete model, endpoint, and credential source are.

## Design

1. In `agent.auxiliary_client._main_route_target`, load `model_aliases` through the normal readonly config loader.
2. Resolve only a matching configured alias with `resolve_startup_model_route`; do not reinterpret arbitrary model names through the global alias registry.
3. Replace the inherited main model, provider, base URL, and credential only with fields returned by that route; retain live runtime values where the alias leaves a field blank.
4. Perform this before the title fast-model preference and MoA handling, so those later steps operate on a concrete provider/model route.
5. Keep explicit `auxiliary.<task>` configuration authoritative. This change applies only to `provider: auto` inheriting the main route.
6. If alias/config resolution unexpectedly raises, retain the existing route and log at debug level. Normal configured aliases must not take that path.

## Acceptance criteria

- An auto auxiliary task with a configured main-model alias sends the alias's concrete model and endpoint to `resolve_provider_client`.
- The literal alias is never used as the provider wire model.
- A main model that is not a configured alias retains existing auxiliary behavior.
- Explicit per-task auxiliary routes remain unchanged.
- The regression test is isolated with a temporary `HERMES_HOME` and exercises the real config and startup-alias resolver.

## Regression coverage

`tests/agent/test_vision_routing.py::TestMainModelAliasRouteForAuxiliary::test_auto_route_expands_main_model_alias` verifies the OpenRouter-style configuration above and asserts the concrete model and alias endpoint reach the auxiliary provider resolver.
