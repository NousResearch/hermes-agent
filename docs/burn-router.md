# Burn Router Plugin

The Burn Router integration is packaged as an optional Hermes plugin. It shells
out to a local Rust/Burn sidecar (`hermes-burn-tool-router`) before a model call
and logs the route prediction for telemetry.

The first implementation is deliberately conservative:

- disabled unless the plugin is enabled
- observe-only by default
- fail-safe: missing binary/model, subprocess errors, timeouts, or malformed JSON
  fall back to normal Hermes behavior
- does **not** change the live tool surface or block model/tool access

## Enable

Add the bundled plugin to `plugins.enabled` and configure the local binary/model:

```yaml
plugins:
  enabled:
    - burn_router
  entries:
    burn_router:
      enabled: true
      mode: observe          # observe | hint | narrow; current hook is telemetry-only
      binary: /path/to/hermes-burn-tool-router
      model: /path/to/router-model
      confidence_threshold: 0.72
      timeout_seconds: 0.25
```

Environment variables override config:

- `HERMES_BURN_ROUTER_ENABLED`
- `HERMES_BURN_ROUTER_MODE`
- `HERMES_BURN_ROUTER_BINARY`
- `HERMES_BURN_ROUTER_MODEL`
- `HERMES_BURN_ROUTER_CONFIDENCE`
- `HERMES_BURN_ROUTER_TIMEOUT`

## Sidecar contract

Hermes invokes:

```bash
$HERMES_BURN_ROUTER_BINARY predict "<user message>" "$HERMES_BURN_ROUTER_MODEL"
```

Expected stdout JSON:

```json
{
  "category": "file",
  "confidence": 0.91,
  "time_us": 123,
  "all": {"file": 0.91, "terminal": 0.04}
}
```

`category` is mapped to Hermes toolset families for telemetry. In observe mode
those mapped toolsets are logged as an empty list because the plugin is not yet
allowed to narrow access.

## Verify with a real sidecar

Unit tests use mocks so CI does not require Rust/Burn artifacts. If you have the
real sidecar and model, run the optional integration smoke test:

```bash
HERMES_BURN_ROUTER_TEST_BINARY=/path/to/hermes-burn-tool-router \
HERMES_BURN_ROUTER_TEST_MODEL=/path/to/tool_router.safetensors \
python -m pytest tests/plugins/test_burn_router_plugin.py -q -o 'addopts='
```

This executes the configured binary with the plugin's real subprocess path and
checks the JSON contract against representative Hermes routes (`x_search`,
`file`, `media_generation`, and `cron`). Without those env vars the integration
case is skipped.

## Logs

Successful predictions emit an info log like:

```text
burn_router prediction: {'category': 'file', 'confidence': 0.91, 'time_us': 123, 'enabled_toolsets': [], 'mode': 'observe'}
```

Failures are debug-level only and do not affect the user turn.

## Why plugin instead of core?

This is experimental routing telemetry, not a mandatory runtime dependency. A
plugin keeps the Hermes core path unchanged, lets users opt in explicitly, and
creates a safer path for future router experiments without imposing Rust/Burn
requirements on every install.
