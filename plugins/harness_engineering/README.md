# Harness Engineering Plugin

User-installable Hermes plugin for Harness / Agenting Engineering soft preflight.

## Install into a Hermes profile

```bash
mkdir -p ~/.hermes/plugins
cp -R plugins/harness_engineering ~/.hermes/plugins/harness_engineering
```

Restart Hermes WebUI/gateway/CLI after installation.

## Modes

Set the default in `~/.hermes/config.yaml`:

```yaml
harness_engineering:
  preflight_mode: advisory  # advisory | strict | off
```

For per-process overrides, `HERMES_HARNESS_PREFLIGHT` takes precedence:

```bash
export HERMES_HARNESS_PREFLIGHT=advisory  # default soft rewrite
export HERMES_HARNESS_PREFLIGHT=strict    # intake-required reminder
export HERMES_HARNESS_PREFLIGHT=off       # disabled
```

The plugin registers `/intake`, `hermes harness ...`, and `pre_gateway_dispatch`.
