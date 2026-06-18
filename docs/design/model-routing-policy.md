# Rule-Based Model Routing Policy for Known Hermes Task Slots

Status: design proposal (not yet implemented)
Author: drafted for maintainers

## Summary

Hermes already supports several model-selection surfaces:

- one main model/provider for the active agent session;
- auxiliary task model overrides (`compression`, `web_extract`, `vision`, approval, title generation, etc.);
- per-cron model overrides;
- per-delegation model overrides;
- fallback providers for rate-limit / capacity failures.

Those pieces work, but users still have to wire the same cost/quality policy in many places by hand. The common failure mode is expensive frontier models being used for routine side tasks — page summarization, title generation, context compression, cron cards, approval classification — because those tasks inherit the main model.

This proposal adds a native, config-driven **`model_routing` policy layer**: named cost/quality tiers plus deterministic defaults/rules for known Hermes task slots.

This is not the same as saved model presets. Presets let a user manually switch between saved model configurations. A routing policy lets Hermes deterministically resolve the right model tier for known task classes while preserving safety and observability.

## Motivation

High-quality models are valuable for hard reasoning, coding, architecture, sensitive changes, and final review. They are wasteful for many mechanical or bounded tasks.

Examples where users currently burn premium model quota unnecessarily:

- `auxiliary.web_extract` summarizing pages with the main model;
- `auxiliary.compression` summarizing context with an expensive reasoning model;
- session-title generation using the main model;
- `approval` / command-risk scoring using the main model;
- LLM-backed cron cards inheriting the main model;
- delegated subagents inheriting a premium parent model for narrow subtasks;
- profiles/agents using the same expensive model even when their work is mostly extraction or formatting.

Hermes has the primitives to avoid this, but the configuration is scattered. A single policy layer would make cost-aware operation discoverable, safer, visible, and testable without lowering quality for the main agent.

## Proposed configuration shape

```yaml
model_routing:
  enabled: true
  mode: shadow # shadow | enforce

  tiers:
    premium:
      provider: openai-codex
      model: gpt-5.5
      reasoning_effort: medium
      thinking: fast

    free_text:
      provider: nous
      model: nvidia/nemotron-3-ultra:free
      reasoning_effort: low

    free_extract:
      provider: nous
      model: stepfun/step-3.7-flash:free
      reasoning_effort: low

  defaults:
    harry: premium
    dobby: premium
    researcher: free_extract
    cron: free_text
    auxiliary: free_text

  rules:
    - name: high_risk
      if_any:
        - auth
        - credentials
        - provider_config
        - deployment
        - destructive_change
      route: premium

    - name: hard_coding
      if_any:
        - debugging
        - refactor
        - architecture
        - tests_failing
      route: premium

    - name: extraction
      if_any:
        - summarize_page
        - web_extract
        - transcript_summary
        - format_digest
      route: free_extract

    - name: simple_text
      if_any:
        - title_generation
        - formatting
        - cron_card
        - rewrite
      route: free_text
```

The exact model IDs above are illustrative, not requested defaults. The important part is the indirection:

1. define named routing tiers;
2. bind defaults and known task classes to tiers;
3. preserve premium routing for high-risk or hard-reasoning work;
4. allow users to start in `shadow` mode before enforcing cheaper routes.

## Modes

### `shadow`

Hermes computes the route it would choose, logs or surfaces the decision, but keeps the current model behavior unchanged.

This gives users and maintainers safe observability before changing runtime behavior.

### `enforce`

Hermes applies the resolved tier for supported known task slots.

Unsupported or ambiguous tasks should fall back to current behavior rather than guessing.

## Resolution order

For any known task slot, resolve the model in this order:

1. Explicit per-call or per-job override.
2. Safety rule requiring `premium`.
3. Matching `model_routing.rules[]` route.
4. `model_routing.defaults.<agent-or-slot>` route.
5. Existing `auxiliary.<task>` / cron / delegation config.
6. Main model.
7. Existing fallback-provider behavior on capacity/rate-limit failures.

This preserves backwards compatibility: if `model_routing` is absent or disabled, behavior is unchanged.

## Known task slots that are safe to route first

These task types are already explicit in Hermes. They do not require semantic classification of arbitrary user messages:

| Slot | Why it is safe to route |
|---|---|
| `auxiliary.title_generation` | short formatting task |
| `auxiliary.approval` | bounded classifier task, with safety fallback to premium if uncertain |
| `auxiliary.web_extract` | explicit extraction/summarization task |
| `auxiliary.compression` | explicit context-summary task |
| `auxiliary.vision` | explicit image-analysis task |
| `auxiliary.session_search` | explicit session-search synthesis task |
| cron jobs | job has its own config and often a narrow prompt/script |
| delegated tasks | caller can pass intent/toolsets/model explicitly |
| named profiles/agents | profile identity can provide a conservative default tier |

Arbitrary main-user-message routing should be a later design. It is harder because Hermes must decide before the main model runs whether the task is simple, high-risk, coding, architecture, extraction, etc.

## Relationship to existing / adjacent work

This proposal is complementary to model presets/profiles.

- A saved model preset stores a manual snapshot of `model` + `auxiliary` assignments.
- A `model_routing` policy defines deterministic route tiers and rules for known task classes.

For example, a user might save a preset called `cheap-weekend`, but inside that preset the `model_routing` policy still says high-risk/auth/deployment work must route to `premium`.

Related upstream work to consider when designing the final shape:

- model presets / lightweight model profiles;
- web model routing editors;
- delegation-specific model/provider overrides.

Those surfaces can become consumers or editors of the same routing policy rather than competing mechanisms.

## UX / observability

A routing layer should be visible to the user. Suggested surfaces:

- `hermes config show` shows tiers, defaults, rules, and current mode.
- Dashboard Models page shows routing tiers and task bindings.
- Usage insights group token/cost by route/task slot where possible.
- `shadow` mode shows “would route X to Y” without changing runtime behavior.
- When an expensive main model is used for a known cheap side task, Hermes can optionally warn or suggest a route binding.

## Safety and cache constraints

- Keep routing decisions outside the model tool schema.
- Do not mutate model/tool configuration during an active model call loop.
- Avoid changing the system prompt or tool schemas mid-conversation.
- Main session model changes should follow existing `/model` semantics.
- Known task routing should happen at the call site before invoking the auxiliary/cron/delegation model.
- High-risk categories should conservatively route to `premium`.
- Ambiguous tasks should fall back to current behavior in the first iteration.

## Testing plan

Unit tests:

- Tier resolution returns provider/model/effort correctly.
- Unknown tier names fail loudly with a helpful config error.
- Explicit per-call overrides beat route bindings.
- Safety rules beat cheaper default routes.
- Missing or disabled `model_routing` preserves current behavior.

Integration tests with temp `HERMES_HOME`:

- Configure two tiers and bind `auxiliary.web_extract`; assert the auxiliary request resolves to the bound provider/model in `enforce` mode.
- Configure `shadow` mode; assert routing decisions are observable but active behavior is unchanged.
- Configure `auxiliary.compression`; assert compression model resolution uses the route without changing the main model.
- Configure a cron default; assert a job without explicit model resolves to the configured cron tier while a job with explicit model keeps its override.
- Configure delegation default; assert explicit delegation overrides still win.

Regression/safety tests:

- No same-role message alternation regressions.
- No prompt-cache-breaking mutation during a conversation loop.
- No fallback-provider behavior regression on rate-limit/capacity errors.
- High-risk route examples never resolve to free/cheap tiers in `enforce` mode.

## Open questions for maintainers

1. Should this live under `model_routing`, or be folded into the existing `auxiliary`, `fallback`, and model-profile config surfaces?
2. Should named tiers support `reasoning_effort`, `thinking`, `service_tier`, and provider-specific `extra_body`, or only provider/model initially?
3. Should Hermes ship suggested presets such as `premium`, `balanced`, `free_text`, `free_extract`, or only expose the mechanism?
4. Should `rules.if_any` use internal task labels, tool names, profile names, user-facing strings, or a small enum?
5. Should main-message auto-routing be considered later, and if so, should it be rule-based, classifier-based, user-mode-based, or agent-requested escalation?
6. Should the dashboard surface warnings when known side tasks inherit an expensive main model?

## Why this belongs in core instead of a user prompt

Prompts and SOUL rules can ask agents to be cost-aware, but known task slots are resolved by Hermes infrastructure before or outside normal assistant reasoning. A native policy layer would make routing deterministic, testable, visible, and reusable across CLI, gateway, cron, delegation, and dashboard flows.

## Suggested first implementation slice

Start with the non-controversial, cache-safe part:

1. Add `model_routing.tiers` resolution helper.
2. Add `shadow` mode observability for known auxiliary slots.
3. Allow auxiliary slots to reference a tier by rule/default in `enforce` mode.
4. Add tests proving backwards compatibility, safety precedence, and override order.
5. Expose the resolved route in config/status output.

Cron/delegation defaults and main-message classification can follow after maintainers settle the API shape.
