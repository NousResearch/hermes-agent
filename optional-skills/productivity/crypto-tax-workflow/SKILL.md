---
name: crypto-tax-workflow
description: Fetch CEX trades (via CCXT) and on-chain wallet transactions (via Blockscout), then compute FIFO/LIFO capital gains / tax obligations using the installed `crypto_tax` Hermes plugin.
tags: [crypto, tax, binance, ccxt, fifo, capital-gains, plugin]
---

# Crypto Tax Workflow

Fetch spot trade history from a CEX supported by CCXT, or on-chain native ETH / ERC-20 transactions from an EVM wallet via Blockscout. Classify disposals with FIFO, split short-term vs long-term, and produce a per-event ledger + tax estimate.

**Install:** `hermes skills install JackTheGit/hermes-crypto-tax-plugin/skills/crypto-tax-workflow`
**Plugin:** Install from `~/.hermes/plugins/` or `git clone https://github.com/JackTheGit/hermes-crypto-tax-plugin`

## When to use

- User asks to fetch trades from Binance / another CEX and compute taxes or gains.
- User provides a wallet address (0x...) on Ethereum, Polygon, Arbitrum, Base, or Optimism and asks for tax obligations.
- User references the `crypto_tax` tool, `calculate_crypto_gains`, `fetch_wallet_transactions`, `fetch_cex_transactions`, or asks for "FIFO", "crypto taxes", "capital gains", "cost basis".
- User provides an `ANNUAL_INCOME_USD` or tax-year in env for bracket-accurate estimates.

The plugin supports BOTH CEX trades (via CCXT) and on-chain wallet transactions (via Blockscout API, no API key required).

## The crypto_tax plugin

Located at `.hermes/plugins/crypto_tax/`. Three tools exposed:

### `fetch_wallet_transactions(address, chain="ethereum", max_pages=20, min_value_eth=0.0, include_erc20=False)`

- `address`: EVM wallet address (0x...).
- `chain`: `"ethereum"` (default), `"polygon"`, `"arbitrum"`, `"base"`, `"optimism"`.
- `max_pages`: Pages to paginate (50 txs/page). Default 20 = 1000 txs. Raise for very active wallets.
- `min_value_eth`: Dust filter — skip native ETH transfers below this threshold. Use `0.0001` to filter noise.
- `include_erc20`: If True, also fetches ERC-20 token transfers for common tokens (USDT, USDC, DAI, WBTC, LINK, UNI, SHIB, AAVE).
- Uses Blockscout open API (no key required). Falls back to CoinGecko for historical ETH prices when available.
- **Caveat**: Only native ETH transfers + optional ERC-20. No internal tx tracing, no DEX swap detection, no NFT sales.
- **Caveat**: Outgoing ETH = "SELL" (taxable disposition). This is simplistic — wallet-to-wallet transfers and contract interactions are NOT real sales but are classified as such. Flag this to the user.
- Returns `{ "address", "chain", "total_fetched", "transactions": [...] }`.

### `fetch_cex_transactions(exchange_id, api_key=None, secret=None, symbols=None)`

- `exchange_id`: any CCXT id (`"binance"`, `"coinbase"`, `"kraken"`, …).
- Credentials fallback: pass `api_key`/`secret` explicitly, or set the env vars `EXCHANGE_API_KEY` + `EXCHANGE_SECRET` (the plugin reads them via the Hermes env system).
- **Pitfall**: if `symbols` is omitted, the plugin DEFAULTS to only `["BTC/USDT", "ETH/USDT"]`. That misses most of the user's history. ALWAYS pass a comprehensive `symbols` list. See `references/common-symbols.md` for a ready-to-use Binance list (~70 pairs).
- Returns `{ "exchange": str, "total_fetched": int, "transactions": [...] }`. Each transaction has `timestamp` (ISO 8601 Z-suffixed), `type` (BUY/SELL), `asset`, `amount`, `price_usd`, `fee_usd`.
- **Important**: `type` from CCXT is `side.upper()` only — STAKING, AIRDROP, MINING do NOT surface through this tool.

### `calculate_crypto_gains(transactions, method="FIFO", jurisdiction="GENERIC", annual_income_usd=0.0)`

- `method`: `"FIFO"` (default), `"LIFO"`, `"SPECIFIC_ID"` support varies.
- `jurisdiction`: `"US"` gives 2024-friendly bracket handling; `"GENERIC"` produces no bracket math.
- Returns dict with, at minimum:
  - `capital_gains_summary`: `{total_realized_gain_loss_usd, short_term_gain_loss_usd, long_term_gain_loss_usd}`
  - `ordinary_income_summary`: `{total_income_usd, staking_usd, airdrops_usd, mining_usd}`
  - `sales_detail`: list of disposals. **This IS the ledger** — each row is one lot-matched disposal event. Fields: `asset`, `sell_date`, `sold_amount`, `sell_price_usd`, `proceeds`, `cost_basis`, `gain_loss`, and usually `long_term` (bool).
  - `dispositions` may be present synonymously; trust `sales_detail` as the canonical key.

## Step-by-step workflow

### For CEX trades

1. **Load env before the plugin import.** The `.env` sits at `.hermes/.env` and holds `BINANCE_API_KEY` / `BINANCE_SECRET`. Use `python-dotenv` with `override=True` before calling into the plugin so `os.getenv` finds them.
2. **Call `fetch_cex_transactions`** with `exchange_id="binance"`, explicit credentials pulled from env, and a broad `symbols` list (references/common-symbols.md).
3. **Call `calculate_crypto_gains`** with the transactions, `method="FIFO"`, `jurisdiction="US"`, and `annual_income_usd` from env (default 80000 if unset).
4. **Persist the raw result as JSON** — e.g. `crypto_tax_raw.json`. Do NOT ask the plugin to write CSV; it doesn't. You do the CSV/analysis step yourself.

### For on-chain wallet addresses

1. **Call `fetch_wallet_transactions`** with `address=<the wallet>`, `chain="ethereum"`, `max_pages=30`, `min_value_eth=0.0001`. This paginates Blockscout and filters dust.
2. If the user mentions ERC-20 tokens, add `include_erc20=True`.
3. **Call `calculate_crypto_gains`** with the returned transactions, `method="FIFO"`, `jurisdiction="US"`.
4. **Persist the raw result as JSON** for downstream analysis.
5. **Analyze with stdlib only.** `execute_code` sandbox does NOT share user-space pip (no pandas, no numpy). Either:
   - (a) Write a post-analysis script in `/home/hermes/` and run it via `terminal` with system python (which has the user-installed pandas), OR
   - (b) Use `execute_code` with `csv` and `json` stdlib only.
6. **Report in a clean flat structure**: totals (proceeds, basis, net gain), STCG/LTCG split, per-asset breakdown, estimated tax (22% ST / 15% LT brackets for mid-income US single-filer; scale with `annual_income_usd`). Include the unknown-basis caveat.

## Pitfalls

- **`pip install` fails on PEP 668 hosts (Debian/Ubuntu 24.04+).** Escape hatch:
  ```bash
  pip3 install --user --break-system-packages --quiet python-binance pandas python-dotenv ccxt
  ```
  If that's not acceptable, use `pipx` or a container instead.
- **`execute_code` sandbox is isolated from user pip.** The script has Python stdlib + what's pre-installed in the sandbox — typically NOT pandas. If you need pandas, run the analysis via `terminal` where system python sees the user-site installs.
- **Unknown-basis disposals are common.** For any sell where the FIFO queue has no matching buy (e.g., asset bought on another exchange, transferred in, or received as income), the plugin records `cost_basis = 0` and reports gain = full proceeds. In the report:
  - Always print a separate line for `$N of unknown-basis gains` and the percentage.
  - Conservatively treat unknown-basis events as SHORT-TERM (no acquisition date means the user can't substantiate long-term holding for the IRS).
  - Advise the user to export records from other platforms / wallets to fill basis gaps.
- **Binance scans time out if you iterate every symbol.** The plugin's `fetch_cex_transactions` only queries the pairs you pass. The full-exchange iteration in `fetch_binance_and_tax.py` takes >10 minutes. Use the fixed common-symbols list (~70 pairs covers BTC/ETH/BNB/SOL/XRP/ADA/DOGE/AVAX/… and stable-quoted variants) for the plugin path.
- **`sales_detail` ≠ `transactions` input.** The input `transactions` list contains both BUYs and SELLs; `sales_detail` is the filtered, lot-matched disposal ledger. When post-processing for tax, operate on `sales_detail`, not the input.
- **The plugin does not deduct fees.** Fees are recorded per-transaction but do NOT reduce proceeds in `sales_detail`. Flag this to the user as a separate schedule-D line item.
- **Very active wallets (5,000+ transactions) will time out.** Always pass `max_pages=30` and `min_value_eth=0.0001` for wallet addresses. The Blockscout API returns 50 txs/page with a 0.3s politeness delay — a full scan of a busy wallet can take 10+ minutes and produce mostly noise (dust transfers, MEV spam). Caps and dust filters are NOT optional for on-chain addresses; apply them BEFORE calling `fetch_wallet_transactions`, not after.
- **Verify the ST/LT split after every calculation.** The `capital_gains_summary.short_term_gain_loss_usd + long_term_gain_loss_usd` MUST equal `total_realized_gain_loss_usd` (within $0.02). If they don't match, the plugin has a bug — unmatched disposals (no cost basis) are silently dropped from the split. See `references/st-lt-split-debugging.md` for the root cause and fix.

## Output

- `crypto_tax_raw.json` — full plugin return (referenceable JSON, re-analyzable).
- Stdout summary with: totals, STCG/LTCG split, per-asset breakdown (asset, events, proceeds, basis, gain), estimated federal tax.
- Save a human-readable CSV if asked (`sales_detail` + a `year` column); do not default to CSV — JSON is the canonical artifact.
