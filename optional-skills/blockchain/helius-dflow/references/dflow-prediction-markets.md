# DFlow Prediction Markets — Discovery, Trading & Redemption

## What This Covers

Prediction market discovery, trading, and redemption on Solana via DFlow APIs. Prediction market trades are always **imperative and async** — they use `/order` and execute across multiple transactions. Do not offer declarative trades for prediction markets.

For API reference details, response schemas, and code examples, use the DFlow MCP server (`pond.dflow.net/mcp`) or the DFlow Cookbook (`github.com/DFlowProtocol/cookbook`).

## Endpoints

* Trade API (dev): `https://dev-quote-api.dflow.net`
* Metadata API (dev): `https://dev-prediction-markets-api.dflow.net`

Dev endpoints work without an API key but are rate-limited. For production use, request an API key at: `https://pond.dflow.net/build/api-key`

## First Questions (Always Ask the User)

1. **Settlement mint?** USDC or CASH — these are the only two.
2. **Dev or production endpoints?** If production, remind them to apply for an API key at `pond.dflow.net/build/api-key`.
3. **Platform fees?** If yes, use `platformFeeScale` for dynamic fees.
4. **Client environment?** (web, mobile, backend, CLI)

## Core Concepts

* **Outcome tokens**: YES/NO tokens are **Token-2022** mints.
* **Market status** gates trading: only `active` markets accept trades. Always check `status` before submitting orders.
* **Redemption** is available only when `status` is `determined` or `finalized` **and** `redemptionStatus` is `open`.
* **Events vs Markets**:
  * **Event** = the real-world question (can contain one or more markets).
  * **Market** = a specific tradable YES/NO market under an event.
  * **Event ticker** identifies the event; **market ticker** identifies the market.
  * Use event endpoints for event data, and market endpoints for market data.
* **Settlement mints**: USDC (`EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`) and CASH (`CASHx9KJUStyftLFWGvEVf59SGeG9sh5FfcnZMVPCASH`). A market settles in whichever mint its outcome tokens belong to.
* **No fractional contracts**: users cannot buy a fractional contract.
* **Minimum order**: 0.01 USDC (10,000 atomic units), but some markets require more because the smallest purchasable unit is one contract and the price determines the minimum.
* **Atomic units**: Like all `/order` calls, the `amount` parameter must be in atomic units. For USDC (6 decimals): 1 USDC = `1_000_000`. For CASH: check the mint's decimal count.

## Market Lifecycle

**`initialized` -> `active` -> `inactive` -> `closed` -> `determined` -> `finalized`**

| Status | Trading | Redemption | Notes |
|---|---|---|---|
| `initialized` | No | No | Market exists but trading hasn't started |
| `active` | **Yes** | No | Only status that allows trades |
| `inactive` | No | No | Paused; can return to `active` or proceed to `closed` |
| `closed` | No | No | Trading ended; outcome not yet known |
| `determined` | No | Check `redemptionStatus` | Outcome decided; redemption may be available |
| `finalized` | No | Check `redemptionStatus` | Final state; redemption available for winners |

Key rules:

* `inactive` is a pause state. Markets can go back to `active` from `inactive`.
* Always check `redemptionStatus` before submitting redemption requests — `determined` or `finalized` alone is not sufficient.
* Filter markets by status: `GET /api/v1/markets?status=active`, `GET /api/v1/events?status=active`, or `GET /api/v1/series?status=active`.

## Maintenance Window

Kalshi's clearinghouse has a weekly maintenance window on **Thursdays from 3:00 AM to 5:00 AM ET**. Orders submitted during this window will not be cleared and will be reverted. Applications should prevent users from submitting orders during this window.

## Compliance (Geoblocking)

Prediction market access has jurisdictional restrictions. Builders are responsible for enforcing required geoblocking before enabling trading, even if KYC (Proof) is used. See: `https://pond.dflow.net/legal/prediction-market-compliance`

## Proof KYC Requirement

**Proof KYC is required only for buying and selling outcome tokens.** Not needed for browsing markets, fetching events/orderbooks/metadata, or viewing market details. Gate verification only at trade time. See `references/dflow-proof-kyc.md` for full integration details.

## Metadata API (Discovery + Lifecycle)

Common endpoints:

* `GET /api/v1/events?withNestedMarkets=true`
* `GET /api/v1/markets?status=active`
* `GET /api/v1/market/by-mint/{mint}`
* `POST /api/v1/filter_outcome_mints`
* `POST /api/v1/markets/batch`
* `GET /api/v1/orderbook/{market_ticker}`
* `GET /api/v1/orderbook/by-mint/{mint}`
* `GET /api/v1/tags_by_categories`
* `GET /api/v1/search?query={query}` — full-text search
* `GET /api/v1/filters_by_sports` — sports-specific filters
* `GET /api/v1/live_data` — REST-based live snapshots

## Categories and Tags (UI Filters)

1. Fetch categories from `GET /api/v1/tags_by_categories`.
2. Use the category name with `GET /api/v1/series?category={category}`.
3. Fetch events with `GET /api/v1/events?seriesTickers={comma-separated}` and `withNestedMarkets=true`.

Corner cases:

* **Too many series tickers**: chunk into smaller batches (5-10) and merge results.
* **Stale responses**: use a request ID or abort controller to ignore older responses.
* **Empty categories**: show a clear empty state instead of reusing prior results.
* **Defensive filtering**: post-filter by `event.seriesTicker` against requested tickers.

## Search API

`GET /api/v1/search?query={query}` for full-text search across events and markets.

Fields searched on **events**: `id` (event ticker), `series_ticker`, `title`, `sub_title`.
Fields searched on **markets**: `id` (market ticker), `event_ticker`, `title`, `yes_sub_title`, `no_sub_title`.

**Not searched**: tags, categories, rules, competition fields, images, settlement sources.

Matching rules:

* Query split on whitespace; **all tokens** must match.
* Ticker fields match upper and lower case.
* Text fields use full-text matching.
* Special characters are escaped before search.

## Candlesticks (Charts)

* **Market detail chart**: `GET /api/v1/market/{ticker}/candlesticks`
* **Event-level chart**: `GET /api/v1/event/{ticker}/candlesticks`

Confirm whether the ticker is a market ticker or event ticker, then use the corresponding endpoint. Use candlesticks (not forecast history) for charting and user-facing price history.

## Prediction Market Slippage

`/order` supports a separate `predictionMarketSlippageBps` parameter for the prediction market leg, distinct from the overall `slippageBps`.

* `slippageBps` controls slippage for the spot swap leg (e.g., SOL to USDC).
* `predictionMarketSlippageBps` controls slippage for the outcome token leg (USDC to YES/NO).

Both accept an integer (basis points) or `"auto"`. When trading directly from a settlement mint to an outcome token, only `predictionMarketSlippageBps` matters.

## Input Mint and Latency

Using the settlement mint (USDC or CASH) as input is the fastest path. Other tokens (e.g., SOL) add a swap leg with ~50ms of additional latency.

## Priority Fees

Prediction market trades use the same `/order` endpoint as spot swaps, so the same priority fee parameters apply:

* **Max Priority Fee** (recommended): set `priorityLevel` (`medium`, `high`, `veryHigh`) and `maxPriorityFeeLamports`. DFlow dynamically selects an optimal fee capped at your maximum.
* **Exact Priority Fee**: fixed fee in lamports, no adjustment.
* If no priority fee parameters are provided, DFlow defaults to automatic priority fees capped at 0.005 SOL.

For additional fee control, use Helius `getPriorityFeeEstimate` (see `references/helius-priority-fees.md`) to inform your `maxPriorityFeeLamports` value.

## Trading Flows

### Open / Increase Position (Buy YES/NO)

1. Discover a market and choose outcome mint (YES/NO).
2. Request `/order` from settlement mint (USDC/CASH) to outcome mint. Include `priorityLevel` for optimal fees.
3. Sign and submit transaction (use Helius Sender — see `references/helius-sender.md`).
4. Poll `/order-status` for fills (prediction market trades are always async).

### Decrease / Close Position

1. Choose outcome mint to sell.
2. Request `/order` from outcome mint to settlement mint.
3. Sign and submit transaction.
4. Poll `/order-status`.

### Redemption

1. Fetch market by mint and confirm `status` is `determined` or `finalized` and `redemptionStatus` is `open`.
2. Request `/order` from outcome mint to settlement mint.
3. Sign and submit transaction.

## Order Status Polling

All prediction market trades are async. Poll `GET /order-status?signature={signature}` with a 2-second interval.

Status values:

* `pending` — Transaction submitted, not confirmed yet
* `open` — Order live, waiting for fills
* `pendingClose` — Order closing, may have partial fills
* `closed` — Complete (check `fills` array for details)
* `expired` — Transaction expired before confirmation
* `failed` — Execution failed

**Keep polling** while status is `open` or `pendingClose`.

**Terminal states** — stop polling when you see:
* `closed` — Success. Read `fills` for execution details.
* `expired` — The transaction's blockhash expired. Rebuild and resubmit with a fresh blockhash.
* `failed` — Execution failed. Check the error, verify market is still `active`, and retry if appropriate.

Pass `lastValidBlockHeight` from the transaction to help detect expiry faster.

## Track User Positions

1. Fetch wallet token accounts using **Token-2022 program**.
2. Filter mints with `POST /api/v1/filter_outcome_mints`.
3. Batch markets via `POST /api/v1/markets/batch`.
4. Label YES/NO by comparing mints to `market.accounts`.

## Market Initialization

* `/order` automatically includes market tokenization when a market hasn't been tokenized yet.
* Initialization costs ~0.02 SOL, paid in SOL (not USDC).
* Any builder can pre-initialize using `GET /prediction-market-init?payer={payer}&outcomeMint={outcomeMint}`.
* DFlow pre-initializes some popular markets.
* If not pre-initialized, the first user's trade pays the initialization cost unless sponsored.

## Fees

### DFlow Base Trading Fees

Probability-weighted model: `fees = roundup(0.07 * c * p * (1 - p)) + (0.01 * c * p * (1 - p))` where `p` is fill price and `c` is number of contracts. Fees are higher when outcomes are uncertain and lower as markets approach resolution. Fee tiers based on rolling 30-day volume are available via the MCP server or at `pond.dflow.net/introduction`.

### Platform Fees (Prediction Markets)

Use `platformFeeScale` instead of `platformFeeBps` for outcome token trades: `k * p * (1 - p) * c` where `k` is `platformFeeScale` with 3 decimals of precision (e.g., `50` means `0.050`), `p` is the all-in price, `c` is the contract size.

No fee when redeeming a winning outcome (p = 1). Fee collected in settlement mint. `platformFeeMode` is ignored for outcome token trades. The `feeAccount` must be a settlement mint token account. Use `referralAccount` to auto-create it if it does not exist.

### Sponsorship

Three costs that can be sponsored:

1. **Transaction fees** — Solana transaction fees (paid by the fee payer)
2. **ATA creation** — Creating Associated Token Accounts for outcome tokens
3. **Market initialization** — One-time ~0.02 SOL to tokenize a market

Options:

* `sponsor` — Covers all three. Simplest for fully sponsored trades.
* `predictionMarketInitPayer` — Covers only market initialization. Users still pay their own transaction fees.

## Account Rent and Reclamation

**Winning positions**: When redeemed, the outcome token account is closed and rent is returned to `outcomeAccountRentRecipient`.

**Losing positions**: Burn remaining tokens and close the account to reclaim rent. Use `createBurnInstruction` and `createCloseAccountInstruction` from `@solana/spl-token`. This is a standard SPL Token operation — DFlow does not provide a dedicated endpoint.

## Market Images

Market-level images are not currently available. Event-level images exist. For market images, fetch from Kalshi directly: `https://docs.kalshi.com/api-reference/events/get-event-metadata`

## Common Mistakes

- Not checking market `status` before attempting a trade (only `active` markets accept trades)
- Not checking `redemptionStatus` before attempting redemption
- Confusing event tickers with market tickers
- Not implementing Proof KYC check before prediction market trades
- Using `platformFeeBps` instead of `platformFeeScale` for outcome token trades
- Submitting orders during the Thursday 3-5 AM ET maintenance window
- Using fractional contract amounts (not supported)
- Not enforcing geoblocking requirements

## Resources

* DFlow Docs: `pond.dflow.net/introduction`
* DFlow MCP Server: `pond.dflow.net/mcp`
* DFlow Cookbook: `github.com/DFlowProtocol/cookbook`
* Prediction Market Compliance: `pond.dflow.net/legal/prediction-market-compliance`
