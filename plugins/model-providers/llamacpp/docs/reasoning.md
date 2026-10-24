# Reasoning control

- reasoning_effort mapped onto the served chat template through the
  request's chat_template_kwargs, clamped to the levels the template
  actually accepts (parsed from /props). A cold or uninspectable
  template passes the configured value through verbatim.
- Thinking off (reasoning_effort none, or enabled false) emits a JSON
  boolean enable_thinking: false; omitted only when the served template
  provably has no toggle.
- Per-model chat_template_kwargs passthrough from Hermes config. The
  reserved reasoning keys (reasoning_effort, enable_thinking) are
  dropped from the passthrough, so the template-aware mapping cannot be
  bypassed from that channel.
- Per-model or session-wide reasoning_budget_tokens, emitted only when
  the server build is confirmed to support the field (>= b8287, read
  from /props build_info).

## Config surface

    providers.llamacpp.models.<id>      per-model metadata:
      chat_template_kwargs              extra Jinja variables for the template
      reasoning_budget_tokens           integer >= -1 (-1 = explicitly off)
    agent.reasoning_effort              session-wide effort
    agent.reasoning_overrides           per-model effort map
    agent.reasoning_budget_tokens       session-wide budget

The legacy custom_providers list form (entries with base_url and a
models list) normalizes to the same surface.

## Examples

    config-examples/qwen38-27b.yaml              thinking model behind llama-swap
    config-examples/deepseek-v4-flash-0731.yaml  DeepSeek V4 Flash 0731

Copy an example into a fresh config.yaml, adjust the api URL, and start
hermes. Each example documents the request-body effects to expect. Both
configs were validated against a live llama-swap deployment, with the
request bodies captured through a local proxy to confirm what actually
reaches the server.
