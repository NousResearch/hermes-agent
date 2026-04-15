# Brokerage Paper-Trading Guide

This subsystem lets Hermes turn a chat request into a deterministic brokerage workflow:

1. The user asks for a trade in natural language.
2. Hermes creates a pending structured trade intent.
3. The local brokerage service returns a confirmation code.
4. The user explicitly confirms with the exact code.
5. Deterministic Python code validates, persists, risk-checks, and submits the order.

The LLM is intentionally not allowed to submit trades directly from raw natural language.

## What v1 does

- Telegram + Hermes driven trading workflow
- Local FastAPI brokerage service
- SQLite persistence for intents and audit events
- IBKR TWS / IB Gateway adapter via `ib_insync`
- Paper trading first
- US stocks only
- Market and limit orders only
- Explicit text confirmation before submission

## Current v1 limitations

- Paper mode is the intended operating mode today
- Live mode is still gated and should be treated as not ready for normal use
- No options, futures, forex, crypto, or short selling
- No autonomous trading or strategy execution
- No Telegram inline button confirmations yet
- No quote-fetching dependency for previews; keep sizing limits conservative
- Real broker connectivity still requires a logged-in local TWS / IB Gateway session

## Required dependencies

The project includes a trading extra:

```bash
uv pip install -e ".[trading,dev]"
```

That extra currently installs:

- `fastapi>=0.104.0,<1`
- `uvicorn[standard]>=0.24.0,<1`
- `ib_insync>=0.9.86,<1`

If you are working inside the Hermes virtualenv and `python -m pip` is unavailable, use:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python 'fastapi>=0.104.0,<1' 'uvicorn[standard]>=0.24.0,<1' 'ib_insync>=0.9.86,<1'
```

## IBKR local setup

For phase 1, use IB Gateway Paper or TWS Paper running on the same machine as Hermes.

### Default IBKR ports

- TWS live: `7496`
- TWS paper: `7497`
- IB Gateway live: `4001`
- IB Gateway paper: `4002`

### Recommended phase-1 choice

Use IB Gateway Paper.

The current adapter defaults to IB Gateway ports:

- paper -> `4002`
- live -> `4001`

### TWS / IB Gateway API settings

In TWS or IB Gateway, enable the API/socket connection settings needed for local clients.

Typical local setup checklist:

- Log in to your paper account
- Enable API connections in TWS / IB Gateway settings
- Allow localhost connections
- Confirm the expected socket port
- Keep the app running while testing Hermes brokerage flows

Important IBKR constraint: this is not a fully headless cloud-native flow. A logged-in local IBKR app session is still required.

## Running the local brokerage service

Start the FastAPI service locally:

```bash
uvicorn brokerage.app:app --host 127.0.0.1 --port 8787
```

Or:

```bash
python -m brokerage.app
```

The default local service URL is:

```text
http://127.0.0.1:8787
```

## Hermes configuration

Hermes reads brokerage settings from the `brokerage` section in `~/.hermes/config.yaml`.

Current default config shape:

```yaml
brokerage:
  enabled: false
  service_url: http://127.0.0.1:8787
  default_account_mode: paper
  confirmation_ttl_seconds: 120
```

The deterministic backend also supports policy settings such as:

- `paper_max_shares`
- `paper_max_notional`
- `live_enabled`
- `live_max_shares`
- `live_max_notional`
- `allowed_symbols`
- `blocked_symbols`

A practical local config looks like this:

```yaml
brokerage:
  enabled: true
  service_url: http://127.0.0.1:8787
  default_account_mode: paper
  confirmation_ttl_seconds: 120
```

The shared bearer token is intended to come from environment-backed config:

```text
BROKERAGE_SERVICE_TOKEN=replace-me-with-a-long-random-secret
```

Example env file: `examples/brokerage.env.example`

## Enabling the Hermes brokerage tools

The brokerage wrappers live in a dedicated `brokerage` toolset and are not enabled globally by default.

Enable them in Hermes, then start a fresh session:

```bash
hermes tools enable brokerage
```

After changing toolsets, start a new session or use `/reset` so the updated toolset is available to the agent.

## HTTP API summary

The local service exposes:

- `GET /healthz`
- `POST /trade-intents`
- `POST /trade-intents/{intent_id}/confirm`
- `POST /trade-intents/{intent_id}/cancel`
- `GET /trade-intents/{intent_id}`

See also: `docs/brokerage/openapi.md`

## Example paper-trading flow

User request:

```text
Buy 1 share of AAPL at market in paper
```

Expected behavior:

1. Hermes calls `create_trade_intent`
2. The service stores a `pending_confirmation` intent
3. Hermes shows a preview and confirmation code such as `T-82K4`
4. User replies with exact text:

```text
CONFIRM T-82K4
```

5. Hermes calls `confirm_trade_intent`
6. The service validates policy, submits through the broker adapter, and persists final status
7. Hermes can call `get_trade_intent_status` to report the result

## Local development checklist

1. Install trading dependencies
2. Run IB Gateway Paper locally and log in
3. Start the brokerage FastAPI service on `127.0.0.1:8787`
4. Set `brokerage.enabled: true` in Hermes config
5. Set `BROKERAGE_SERVICE_TOKEN`
6. Enable the `brokerage` toolset in Hermes
7. Start a fresh Hermes session and run a paper trade request

## Security notes

This project is intentionally designed to separate natural-language interpretation from actual execution.

Key safeguards already in place:

- Trade requests become pending intents first
- Confirmation requires an explicit second message
- State transitions are persisted in SQLite
- Policy checks run before broker submission
- Paper and live are configured separately
- Live trading is disabled by default

## Live-trading warning

Do not treat this branch as production-ready for live money.

Before enabling live trading in any serious way, you should add at minimum:

- stronger operational auth and host hardening
- stricter risk limits and symbol allowlists
- order-status reconciliation and retry design
- better cancellation and fill handling
- stronger audit/reporting workflows
- manual operational runbooks and smoke tests

For now, use IBKR paper trading only.
