# Optional Burn Tool Router Integration

Hermes can now run a local Rust/Burn classifier as an **observe-only pre-router** for each user turn.

The goal is to collect cheap routing telemetry before any model call without risking tool starvation. In this first integration, the router logs what it would choose but does **not** change the live tool surface.

## Why this exists

LLM tool schemas get large fast. A tiny local classifier can predict the likely tool family before the model request and eventually help Hermes:

- reduce schema bloat for obvious turns,
- bias tool ordering/toolset exposure,
- collect tool-routing labels from real traffic,
- keep full fallback when confidence is low.

## Safety model

Default config is disabled:

```yaml
routing:
  burn_router:
    enabled: false
    mode: observe
    binary: ""
    model: ""
    confidence_threshold: 0.72
    timeout_seconds: 0.25
```

Modes:

- `observe`: log category/confidence only. No behavior change.
- `hint`: return high-confidence `enabled_toolsets` hints to callers. Current conversation loop does not consume this yet.
- `narrow`: reserved for future experiments; treated as advisory and should keep fallback.

Failures are safe:

- missing binary/model → skip,
- timeout → skip,
- malformed JSON → skip,
- non-zero router exit → skip.

## Local smoke test

```bash
export HERMES_BURN_ROUTER_ENABLED=true
export HERMES_BURN_ROUTER_MODE=observe
export HERMES_BURN_ROUTER_BINARY=/path/to/hermes-burn-tool-router/target/release/hermes-burn-tool-router
export HERMES_BURN_ROUTER_MODEL=/path/to/hermes-burn-tool-router/tool_router.safetensors

python - <<'PY'
from agent.burn_router import BurnRouterConfig, get_burn_router_hint
cfg = BurnRouterConfig.from_config({'routing': {'burn_router': {'enabled': True}}})
print(get_burn_router_hint('search X for trending Base coins', cfg))
PY
```

## Next step

After observe logs are collected, add a labeler that compares predicted category against actual first tool/toolset used. If precision is high enough, enable `hint` mode for obvious cases like `x_search`, `file`, `web`, and `terminal`.
