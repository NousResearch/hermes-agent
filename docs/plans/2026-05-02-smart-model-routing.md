# Smart model routing for simple turns

## Goal

Route obviously simple, low-risk user turns to a configured cheap model while preserving the existing primary model for coding, tool-heavy, contextual, multimodal, and long-running work.

## Architecture

- Add a small shared routing module under `agent/` with pure functions:
  - classify whether a user message is safe to route cheaply;
  - build an alternate route from `smart_model_routing.cheap_model`;
  - resolve provider credentials through `hermes_cli.runtime_provider.resolve_runtime_provider`.
- Call the helper from both CLI and gateway `_resolve_turn_agent_config()` so Telegram/Discord/etc. and terminal behavior match.
- Keep `/fast` service-tier request overrides independent from model routing.
- Keep the agent cache safe by including the routed model/provider/base_url/api_mode in the existing route signatures.

## Initial policy

Route to cheap model only when all are true:

- `smart_model_routing.enabled` is true;
- a cheap model is configured;
- current turn is plain text;
- no prior conversation history unless explicitly disabled via config;
- message is short (`max_simple_chars`, `max_simple_words`);
- message is not a slash command, not an `@file`/`@diff` context reference, not a code block, not an image/media indicator;
- message does not contain obvious coding/deploy/research/tooling verbs or dangerous-operation terms.

## Non-goals

- No LLM classifier in the first slice: it would spend tokens to save tokens.
- No mid-agent switching after tool results.
- No routing for cron/API-server/Feishu-comment special paths in this first slice unless they already share gateway `_resolve_turn_agent_config()`.

## Tasks

1. Add RED tests for classification and cheap route resolution.
2. Add helper implementation.
3. Wire CLI and gateway wrappers, passing history where available.
4. Document config keys.
5. Validate targeted tests and a lightweight integration smoke.

## Validation

- `python -m pytest tests/agent/test_smart_model_routing.py -q`
- targeted CLI/gateway tests if added
- `python -m pytest tests/hermes_cli/test_config_validation.py tests/gateway/test_session_model_override_routing.py -q` when routing code touches those paths
