# BurnBar Cloud platform plugin

This plugin adds BurnBar Cloud as a Hermes messaging platform. It is a
gateway adapter only: messages, attachments, approvals, and runtime state flow
through the BurnBar Hermes Gateway API; this first platform-plugin layer does
not add end-to-end encryption.

- device-code setup against BurnBar's Hermes Gateway API
- human-in-the-loop oversight (supervised / autonomous gating of slash-confirms)
- runtime status / model-catalog publication for the BurnBar model picker
- event polling with durable cursor persistence
- Hermes replies through `/messages`
- typing state through `/typing`
- native attachment delivery through `/attachments/init`
- standalone cron delivery with `deliver=burnbar`

## Configuration

`hermes gateway setup` writes these values after the device-code flow:

- `BURNBAR_API_BASE_URL` - gateway base URL.
- `BURNBAR_ACCESS_TOKEN` - scoped bearer token from the approved device grant.
- `BURNBAR_HOME_CHANNEL` - default BurnBar destination for cron and notification delivery.
- `BURNBAR_ALLOW_ALL_USERS` / `BURNBAR_ALLOWED_USERS` - sender allowlist controls.

## Setup

```bash
hermes gateway setup
hermes gateway restart
hermes gateway status
```

Choose `BurnBar Cloud`, approve the displayed device code in BurnBar, then
restart the gateway and send a message from BurnBar. To point the adapter at a
non-default gateway, set `BURNBAR_API_BASE_URL` before running setup.

## Tests

The plugin ships a deterministic, dependency-light test suite that loads the
adapter via `tests/gateway/_plugin_adapter_loader.load_plugin_adapter("burnbar")`
(no `sys.path` tricks — the `tests/gateway/conftest.py` guard enforces this):

```bash
scripts/run_tests.sh tests/gateway/test_burnbar_plugin.py
```

It exercises:

- plugin registration + `Platform("burnbar")` dynamic resolution
- config / env-enablement / yaml-precedence
- `/events` mapping to `MessageEvent` and `model_switch`
- `/messages` send happy path + error → `SendResult(success=False)`
- `/attachments/init` + signed upload
- cursor round-trip
- oversight (`/state`) refresh + autonomous auto-approve, runtime-status payload
