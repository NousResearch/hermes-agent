# first-turn-copywriter

Project-local plugin template that uses `pre_llm_call` plus `is_first_turn` to:

1. Call `delegate_task` with the exact subagent goal `帮我生成100个字的文案`
2. Read the child agent's `summary`
3. Inject that text into the main model's current-turn user-message context

Usage:

- Enable project plugins before starting Hermes: `HERMES_ENABLE_PROJECT_PLUGINS=true`
- Start Hermes from this repository root
- On the first turn of a new session, the hook runs automatically

Notes:

- `delegate_task` needs a parent agent context, so this template is intended for CLI sessions where `ctx.dispatch_tool()` can resolve the active agent.
- To make the copy stricter or longer, edit `_SUBAGENT_GOAL` in `__init__.py`.