# Deterministic model router

Hermes has an opt-in, turn-boundary model router. It is disabled by default.
The router never changes the model during a tool loop, and it does not perform
credential discovery or network probes.

Example configuration:

```yaml
model_router:
  mode: auto # off, suggest, or auto
  candidates:
    - model: gpt-5-mini
      provider: openai-codex
      reasoning: true
      vision: true
      context_window: 400000
      quality: 0.8
      cost: 0.3
    - model: kimi-k3
      provider: kimi-coding
      reasoning: true
      vision: true
      context_window: 1048576
      quality: 1.0
      cost: 0.8
```

`off` preserves the configured model. `suggest` keeps the configured model and
records the best candidate in the route metadata. `auto` selects the best
eligible candidate once before constructing `AIAgent`; provider resolution
failures and capability mismatches fall back to the current model.

Candidates are explicit declarations. Their provider must already be
configured and authenticated; the router does not infer availability from a
model name.