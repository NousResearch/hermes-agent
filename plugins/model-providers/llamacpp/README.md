# llamacpp model provider plugin for Hermes

Serves llama.cpp endpoints - a bare llama-server or a llama-swap proxy -
as a first-class Hermes provider named "llamacpp" (alias "llama-swap").
Local endpoint, no API key.

What it adds over the generic custom provider:

- Endpoint detection and a read-only server probe (probe.py). On
  llama-swap the probe never touches a model-scoped route for a model
  that is not resident: llama-swap starts models on those routes, and a
  metadata probe must never load a model as a side effect.
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
- Replayed assistant messages keep reasoning_content, so llama-server's
  --reasoning-preserve flag reuses the prompt cache across turns.
- Status bar surfacing: server-reported tokens per second (from the
  response timings block) and llama-swap residency (from /running).
- Opt-in prompt warmer: primes the server's prompt cache with the exact
  session preamble at session open, so the first user turn processes
  only the new tokens. Default off. The warm request starts the model
  on llama-swap exactly like a real first turn would - that is the
  point, but it is why the gate is strict (boolean true only).

## Setup

The plugin ships bundled with hermes-agent and registers on start; there
is nothing to install. A user plugin of the same name at
~/.hermes/plugins/model-providers/llamacpp overrides this bundled copy
(last-writer-wins), so the profile can be replaced without editing the
checkout.

Point it at your server and pick a model:

    model:
      default: <model-id>
      provider: llamacpp
    providers:
      llamacpp:
        api: http://<host>:<port>/v1

Or configure only the providers block and select interactively:

    hermes model

The llamacpp provider row lists the server's live model catalog;
picking a model writes the selection to your config. No API key is
needed. See Examples for complete configs including reasoning control.

## Config surface

    providers.llamacpp.api              endpoint base URL (.../v1)
    providers.llamacpp.models.<id>      per-model metadata:
      chat_template_kwargs              extra Jinja variables for the template
      reasoning_budget_tokens           integer >= -1 (-1 = explicitly off)
      prompt_warmer                     boolean, wins over agent.prompt_warmer
    agent.reasoning_effort              session-wide effort
    agent.reasoning_overrides           per-model effort map
    agent.reasoning_budget_tokens       session-wide budget
    agent.prompt_warmer                 opt-in prompt warmer

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
