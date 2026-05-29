# Local E2E Evidence: `/dobby status` and `/dobby help`

Date: 2026-05-29

Scope: local/staging only. No live Discord gateway, remote host, network
service, shell tool, or LLM path is used.

## Covered Scenarios

- `/dobby status` is dispatched through `BasePlatformAdapter.handle_message`,
  `GatewayRunner._handle_message`, and the adapter send path using a synthetic
  Discord source.
- `/dobby help` follows the same local gateway path.
- `HERMES_HOME` is a temp directory containing only synthetic config-shaped
  data.
- The synthetic user is authorized through `DISCORD_ALLOWED_USERS`; the config
  also carries the synthetic allowed channel.
- The send callback is a local fake adapter that records the response.
- Command hooks are spied and must not run.
- A stale running-agent entry is present; cleanup and interrupt paths must not
  mutate it.
- Session store methods and agent/chat handlers are spied and must not run.
- Secret-shaped synthetic values are present in loaded config-shaped data:
  OpenAI-style key, Discord token shape, and webhook secret shape. None may
  appear in sent output.

## Commands and Results

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -o addopts='' tests/gateway/test_dobby_command_center_e2e.py
```

Result: `2 passed in 0.13s`.

```bash
PYTHONDONTWRITEBYTECODE=1 HERMES_HOME="$(mktemp -d)" pytest -o addopts='' tests/gateway/test_dobby_command_center.py tests/gateway/test_dobby_command_center_e2e.py
```

Result: `12 passed, 1 warning in 0.16s`.

```bash
PYTHONDONTWRITEBYTECODE=1 HERMES_HOME="$(mktemp -d)" pytest -o addopts='' tests/productization/dobby_package
```

Result: `50 passed, 1 warning in 11.01s`.

## Live Canary Blockers

- This evidence intentionally avoids the real Discord gateway and does not prove
  Discord API registration, permissions, rate limits, or message delivery.
- Before live canary, run an approved Discord staging app with non-secret test
  credentials and confirm the same `/dobby status` and `/dobby help` responses
  in an allowlisted channel.
