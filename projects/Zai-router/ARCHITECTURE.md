# Architecture

## Request path

```text
Desktop model picker / CLI
  -> provider: zai-indirect
  -> ProviderProfile
  -> api_mode: anthropic_messages
  -> base URL: https://api.z.ai/api/anthropic
  -> agent/anthropic_adapter.py
  -> Z.ai Anthropic Messages endpoint
```

The profile exposes only `glm-5.2` and does not run an OpenAI-style `/models` probe.

## Provider separation

`zai-indirect` is deliberately separate from `zai`:

- `zai` continues to use Z.ai's OpenAI-compatible PaaS route.
- `zai-indirect` uses the Anthropic Messages route.
- The providers have distinct identifiers and can use distinct credential namespaces.
- Endpoint-specific behaviour is activated only for the exact Z.ai Anthropic endpoint.

## Picker discovery

Bundled provider registration extends Hermes' canonical provider registry. Once a supported credential is available, model catalogue generation returns one distinct row:

```text
Z.ai Indirect
  glm-5.2
```

The terminal cache can be refreshed with:

```bash
hermes model --refresh
```

The gateway must be restarted after source or provider-registration changes.

## Project-directory boundary

`projects/Zai-router/` is the portable project home, not the runtime module path. Hermes' loader expects provider code under `plugins/model-providers/`, and the shared Anthropic transport remains under `agent/`.
