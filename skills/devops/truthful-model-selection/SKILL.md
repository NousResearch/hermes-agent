---
name: truthful-model-selection
description: Diagnose and harden Hermes model selection so the chosen model is truly executed, fallbacks are explicit, and Telegram model switching remains honest under rate limits.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [model-routing, fallback, telegram, rate-limits, hermes]
    related_skills: [systematic-debugging, hermes-agent]
---

# Truthful Model Selection

Use this when Hermes appears to "switch models" in UI or Telegram, but runtime behavior still follows hidden fallback chains or stale provider config.

## Goal

Guarantee that:
1. selected_model is the user's requested truth
2. executed_model is what actually ran
3. fallback only happens when policy explicitly allows it
4. Telegram `/model` always changes control state honestly, even during rate limits
5. background jobs keep running independently from chat/provider turbulence

## Root problem pattern

Typical symptom cluster:
- UI says "Model switched to X"
- runtime still prints "switching to fallback provider"
- retries hit a stale model like `MiniMax-Text-01`
- user correctly experiences this as a lie

Common architectural cause:
- global `fallback_model` / `fallback_providers` are loaded automatically at boot
- session `/model` changes visible state but does not clear or override runtime fallback chain
- selected model and executed model are not tracked separately
- stale fallback entries survive in config or caches

## Investigation checklist

1. Inspect where fallback chain is built.
   - Search for `_fallback_chain`, `_fallback_model`, `_load_fallback_model`, `fallback_providers`, `fallback_model`.
2. Confirm whether gateway boot loads global fallback config automatically.
3. Confirm whether `/model` changes only display/session metadata or also runtime execution state.
4. Search logs for these strings:
   - `Rate limited — switching to fallback provider`
   - `Non-retryable error (HTTP 404)`
   - stale model names (for example `MiniMax-Text-01`)
5. Verify whether rate-limit handling and model-selection messaging are handled by different code paths.
6. Inspect provider list sources used by Telegram, TUI, setup, and runtime.
   - In Hermes, check whether the picker reads `providers` while runtime resolution still reads `custom_providers`.
   - A real failure mode observed in production: `~/.hermes/config.yaml` stores named local endpoints under `providers:` (for example `ollama-pluto`, `ollama-elibook`), but `runtime_provider.py` still does `config.get("custom_providers")` and returns `None` when that key is absent or not a list.
   - When these structures diverge, local endpoints can appear in menus yet fail at execution time, forcing manual `ssh` + `hermes setup model` repair.
7. Verify the provider picker and resolver use one canonical schema.
   - Backward-compatible migration is fine, but UI, setup wizard, Telegram `/model`, and runtime execution must all read the same effective provider list.
   - If the codebase is mid-migration, add an adapter layer instead of maintaining parallel code paths.
8. Audit provider-menu omissions separately from runtime support.
   - A provider may already exist in `providers.py` overlays yet still be absent from the switch menu.
   - Concrete case: `openai-codex` existed as a provider overlay with `transport="codex_responses"`, but the picker/fallback tables did not expose the expected OpenAI Codex model family.
   - When users report a missing provider group, inspect both overlay registration and the picker/setup fallback model tables.
9. For local OpenAI-compatible endpoints, verify model discovery order:
   - explicit `models`
   - `default_model` or `model`
   - full probe of `/v1/models` to collect all advertised model ids
   - only then a single-model auto-detect fallback
   - if the backend is local/Ollama-compatible, allow a non-empty placeholder API key such as `ollama` instead of treating empty remote-provider credentials as a reason to fall back to OpenRouter
   - important production finding: relying on auto-detect only when exactly one model is exposed breaks Telegram/TUI switching for multi-model Ollama endpoints and leaves provider groups showing `0` models
10. When a switch targets a local provider, ensure runtime does not silently re-enter remote credential resolution.
   - If the chosen provider is `ollama-pluto` or `ollama-elibook`, the executor should pin `base_url` to that endpoint and skip OpenRouter/OpenAI API-key fallback logic.
   - after a manual `/model` switch, explicitly clear any stale gateway/session fallback state such as `_effective_model`, `_effective_provider`, and cached agent `_fallback_model`; otherwise the UI can report a successful switch while execution still follows an earlier degraded route and keeps surfacing rate-limit behavior
11. Audit picker visibility separately from raw credential detection.
   - Hermes-only providers like `openai-codex` should remain visible when they are the current provider or configured provider, even if one credential-detection path fails.
   - keep the configured/current model visible in the picker even when the curated catalog is stale or incomplete, so users can still reselect known-good runtimes from Telegram/TUI.

## Design rules to enforce

### 1. Separate selected and executed model
Every response or task execution should carry:
- `selected_model`
- `selected_provider`
- `executed_model`
- `executed_provider`
- `execution_reason` = `direct | degrade | override | resume`

If executed != selected, surface it explicitly.

### 2. Make fallback policy explicit
Support a small policy surface:
- `strict`: never fallback
- `degrade`: may fallback only within approved ladder
- `override`: system may temporarily reroute, but must report it clearly

Never allow hidden fallback when policy is `strict`.

### 3. Quarantine dead fallback entries
If a model returns `404 not_found`, mark it unusable for the current session/job immediately.
Do not retry it again in the same execution path.

### 4. Health-gate the fallback ladder
Before using fallback providers/models:
- validate config entries at boot
- probe model availability where feasible
- disable invalid entries before runtime

A safe Telegram ladder is:
1. requested model/provider
2. approved same-provider sibling (only if policy allows)
3. approved remote fallback
4. local Pluto Ollama
5. local Elitebook Ollama

### 5. Separate control plane from worker plane
Telegram chat should set execution policy, not host long-running execution.
Long-running tasks must run as background jobs/subagents with pinned model policy and notify status back to chat.

## Implementation outline

1. Add a canonical execution policy object per session/job.
   Include:
   - preferred provider/model
   - fallback policy
   - allowed fallbacks
   - retry budget
   - local escape hatch list
2. Ensure `/model` updates that policy object.
3. On each turn/job start, resolve executor from policy.
4. Before sending user-facing "switched" messages, verify the runtime executor actually changed.
5. Emit one truthful status line when degradation occurs.
6. If no healthy executor exists, fail honestly:
   - `Switch failed; no healthy fallback available`
   rather than pretending success.

## Acceptance criteria

A fix is not done until all five pass:
1. `/model` in Telegram always changes `selected_model` state.
2. `executed_model` is logged or surfaced for every answer/job.
3. Silent fallback is impossible.
4. Stale/404 models do not remain in active fallback chains.
5. Active tasks continue as background jobs despite chat/provider issues.

## Pitfalls

- Do not treat UI confirmation as evidence that runtime switched.
- Do not keep a global fallback chain active after a user explicitly picked a model unless policy allows it.
- Do not retry a provider/model that already returned 404.
- Do not bind task survival to the Telegram request/response lifecycle.

## Useful searches

Use Hermes file search instead of grep:
- `_fallback_chain`
- `_fallback_model`
- `_load_fallback_model`
- `fallback_providers`
- `fallback_model`
- `Rate limited — switching to fallback provider`
- stale model identifiers such as `MiniMax-Text-01`

## Outcome

A truthful system may still degrade under rate limits, but it never lies about what happened. That distinction is critical when users depend on model guarantees and uninterrupted task execution.
