# Polytrader Polymarket CLOB v2 Setup

`polytrader` is a Python, DRY_RUN-first core for Polymarket short-interval crypto up/down markets. It is safe by default: no live orders are posted unless `DRY_RUN=false`, a private key is configured, and the live-mode checklist below is intentionally completed.

## Install

Use the repository virtual environment or `uv` workflow already used by Hermes:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.[dev]'
```

Install the Polymarket CLOB v2 SDK only when preparing live execution:

```bash
pip install -e '.[polymarket]'
```

The execution adapter imports from `py_clob_client_v2` and posts via `create_and_post_order` with CLOB v2 `tick_size` and `neg_risk` options.

## Environment

Copy `.env.example` to `.env` and fill in only what you need. Keep real secrets out of chat, docs, test fixtures, screenshots, and logs.

```bash
cp .env.example .env
```

Important defaults:

```env
DRY_RUN=true
CLOB_HOST=https://clob.polymarket.com
GAMMA_HOST=https://gamma-api.polymarket.com
CHAIN_ID=137
CRYPTO_SYMBOL=BTC
MARKET_SLUG=
```

## Safe/proxy wallet support

- `SAFE_ADDRESS` is treated as the Polymarket Safe/proxy wallet holding collateral.
- If `SAFE_ADDRESS` is set and `FUNDER_ADDRESS` is blank, `FUNDER_ADDRESS` defaults to `SAFE_ADDRESS`.
- If `SAFE_ADDRESS` is set and `SIGNATURE_TYPE` is blank, `SIGNATURE_TYPE` defaults to `2` for Gnosis Safe-style signing.
- `PRIVATE_KEY`, `CLOB_API_KEY`, `CLOB_API_SECRET`, and `CLOB_API_PASSPHRASE` are required only for authenticated/live workflows.

## Collateral-balance risk controls

Use collateral terminology, not legacy USDC-only names:

```env
MAX_COLLATERAL_PER_TRADE=10
MIN_COLLATERAL_BALANCE=25
MAX_OPEN_POSITIONS=1
MIN_EDGE=0.01
```

Risk gates block trades when collateral balance is below the floor, requested collateral exceeds the per-trade cap, a trade would breach the post-trade collateral floor, or open positions are already at the configured cap.

## Market selection

- Set `MARKET_SLUG` to pin one exact 5-minute up/down market.
- Otherwise set `CRYPTO_SYMBOL` (`BTC`, `ETH`, `SOL`, or `XRP`) and the selector filters active, non-closed Polymarket Gamma events to the chosen **5-minute** up/down market.
- Fifteen-minute markets are intentionally ignored by the selector.

## Fee-aware evaluation

Before a buy decision, the core rounds the ask price to the market tick and computes edge after the fee metadata supplied by CLOB v2:

```python
fee_base = min(price, 1.0 - price) ** fee_exponent
fee_per_share = fee_base * (fee_rate_bps / 10_000.0)
edge_after_fees = model_probability - price - fee_per_share
```

The trade is skipped unless `edge_after_fees >= MIN_EDGE`.

## Dry-run command

Run the core tests before any operational use:

```bash
scripts/run_tests.sh tests/test_polytrader_core.py
```

Dry-run execution returns an `ExecutionReceipt(status="dry_run")` and never calls the CLOB client's order-posting methods.

## Live-mode checklist

Do not set `DRY_RUN=false` until all items are true:

- [ ] Dry-run decisions have been logged and reviewed over enough market windows.
- [ ] `MAX_COLLATERAL_PER_TRADE`, `MIN_COLLATERAL_BALANCE`, and `MAX_OPEN_POSITIONS` are intentionally small.
- [ ] `MARKET_SLUG` or `CRYPTO_SYMBOL` points to the intended 5-minute up/down market class.
- [ ] CLOB v2 SDK is installed with `pip install -e '.[polymarket]'`.
- [ ] `PRIVATE_KEY` and CLOB API credentials are present only in `.env` or a secure secret store, never in chat/logs/docs.
- [ ] `SAFE_ADDRESS` / `FUNDER_ADDRESS` / `SIGNATURE_TYPE` match the wallet type you intend to use.
- [ ] A small live canary order size is configured and you explicitly accept the risk.

This software is not financial advice. Live trading can lose funds.
