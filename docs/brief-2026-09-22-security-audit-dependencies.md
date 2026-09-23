# Remove security-audit findings from supported Hermes environments

## Problem

`hermes security audit` on the current supported installation reports known vulnerabilities in packages installed by Hermes:

- `httpx2==2.7.0` and `httpcore2==2.7.0`
  - secure WebSocket traffic may be sent without TLS through SOCKS proxies
  - streaming decompression can amplify memory use
  - SSE buffering and multipart/header issues
- `tornado==6.5.7`
  - request parsing can stall the event loop or amplify memory use
- stale environments may still retain `setuptools==79.0.1`, although current source pins 83.0.0

The audit lists fixed floors of `httpx2/httpcore2 >= 2.12.0`, `tornado >= 6.5.8`, and `setuptools >= 83.0.0`.

## Required behavior

1. Fresh installs and normal `hermes update` runs must resolve to versions without these findings.
2. Exact pins and `uv.lock` must stay consistent across core, dev, MCP, and computer-use extras.
3. Updating `httpx2` must remain compatible with `mcp==2.0.0` and Hermes's direct `httpx2` imports.
4. Updating Tornado must remain compatible with `python-telegram-bot[webhooks]` and any direct Hermes use.
5. Existing stale Hermes environments must be refreshed by the supported updater rather than requiring manual package surgery.
6. Do not suppress advisories, add ignore rules, or weaken the audit.

## Tests first

Add or update a focused dependency-contract test that fails against the vulnerable pins/lock and proves:

- every Hermes-declared `httpx2` pin meets the fixed floor;
- the lock resolves `httpx2` and `httpcore2` at safe versions;
- Tornado resolves at 6.5.8 or newer;
- setuptools remains at 83.0.0 or newer where Hermes controls it.

Then update the minimum exact pins and regenerate the lock using the repository's supported workflow.

Likely owner files:

- `pyproject.toml`
- `uv.lock`
- focused dependency/security audit tests
- updater dependency-refresh logic only if a normal update would otherwise leave vulnerable installed packages behind

## Acceptance

- The new contract test fails before the change and passes afterward.
- `uv sync` or the repository's equivalent dependency installation succeeds.
- Focused MCP, webhook/Telegram import, updater, and security-audit tests pass.
- `hermes security audit` against a clean environment made from the changed branch reports none of the named advisories.
- The fix does not hide unrelated audit findings.

## Scope

One dependency remediation only. No dependency-manager rewrite, unrelated package upgrades, or security-audit redesign. Target about 80 production/configuration lines plus generated lock changes and focused tests.

Do not merge, deploy, restart gateways, or modify a user's installed environment.
