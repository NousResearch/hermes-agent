---
name: smart-model-routing
description: Route Hermes turns across configured models.
version: 0.1.0
author: Waylish (@Waylish), with Hermes Agent/Codex
license: MIT
prerequisites:
  plugins: [smart-model-routing]
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [models, routing, plugin, productivity]
    category: productivity
    requires_toolsets: [smart_model_routing]
    homepage: https://github.com/Waylish/hermes-smart-model-routing
---

# Smart Model Routing

Use this skill when the user wants to configure, test, tune, or explain
per-turn model routing across multiple Hermes model APIs.

## Requirements

- Expected plugin: `smart-model-routing`.
- Expected plugin tool: `smart_model_route_preview`.
- Expected config block: `smart_model_routing`.
- Automatic model switching requires Hermes native `resolve_model_route` support
  or an equivalent core integration. Without that, use preview-only behavior and
  do not claim that Hermes is switching models automatically.
- Keep API keys out of `config.yaml`. Use `api_key_env` and store secrets in
  Hermes' normal secret environment file.

## Workflow

1. Inspect the `smart_model_routing` config block.
2. Confirm `enabled: true`.
3. Confirm every intended tier has `provider`, `model`, and either `api_key_env`
   or inherited provider credentials.
4. Use `smart_model_route_preview` on representative prompts.
5. Compare the returned tier, reason, provider, model, and reasoning settings
   with the user's intended routing policy.
6. Tune thresholds first, then keywords, then tier model choices.
7. Report whether the result is preview-only or automatic switching is active.

## Recommended Tiers

| Tier | Use for | Typical model |
|---|---|---|
| `cheap` | Greetings, short questions, low-risk text | `deepseek-v4-flash` |
| `standard` | Normal explanations and medium prompts | `kimi-k2.6` |
| `code` | Code, tracebacks, refactors, tests, repo work | `kimi-for-coding` |
| `reasoning` | Multi-step analysis, strategy, research, proof, forecasting | `deepseek-v4-pro` |
| `long_context` | `@file` references, whole documents, large prompts | `kimi-k2-thinking-turbo` |
| `media` | Images or non-text attachments | A vision-capable primary model, such as `MiniMax-M3` |

Missing tiers should fall back to Hermes' primary `model:` config.

## Preview Cases

Use cases like these when validating a config:

| Prompt | Expected tier | Why |
|---|---|---|
| `hello` | `cheap` | Short text |
| `Explain TCP congestion control in practical terms.` | `standard` | Medium explanation |
| `Fix this traceback and add tests: ...` | `code` | Code/debug keyword |
| `Analyze 2026-2028 AI chip supply chain risk...` | `reasoning` | Strategic analysis |
| `Summarize @docs/spec.md and identify contradictions.` | `long_context` | Context reference |
| A message with an attached image | `media` | Non-text input |

## Tuning Order

1. Tune `max_simple_chars`, `max_simple_words`, and `max_simple_lines`.
2. Tune `max_standard_chars`, `max_standard_words`, and `max_standard_lines`.
3. Tune `code_keywords`, `reasoning_keywords`, and `long_context_keywords`.
4. Tune tier `model`, `provider`, and `reasoning` settings.

If a prompt routes too cheaply, lower the relevant threshold or add a keyword.
If a prompt over-routes to an expensive model, raise the threshold or remove an
overbroad keyword.

## Response Format

When reporting a route, include:

- selected tier
- trigger reason
- provider and model
- reasoning settings, if any
- whether the route is preview-only or automatic switching is active
- fallback behavior when the tier is missing or credentials are unavailable
