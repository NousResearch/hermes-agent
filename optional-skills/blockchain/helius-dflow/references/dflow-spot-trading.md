# DFlow Spot Trading — Token Swaps on Solana

## What This Covers

DFlow is a DEX aggregator on Solana that sources liquidity across venues. This reference covers spot crypto token swaps using two trade types: **imperative** (recommended starting point) and **declarative**.

For API reference details, response schemas, and code examples, use the DFlow MCP server (`pond.dflow.net/mcp`) or the DFlow Cookbook (`github.com/DFlowProtocol/cookbook`).

## Endpoints

* Trade API (dev): `https://dev-quote-api.dflow.net`
* Metadata API (dev): `https://dev-prediction-markets-api.dflow.net`

Keep in mind:

* Dev endpoints are for end-to-end testing during development.
* Do not ship to production without coordinating with the DFlow team.
* Be prepared to lose test capital.
* Dev endpoints are rate-limited and not suitable for production workloads.

Dev endpoints work without an API key. For production use, request an API key at: `https://pond.dflow.net/build/api-key`

## CORS: Browser Requests Are Blocked

The Trading API does not set CORS headers. Browser `fetch` calls to `/order` or `/intent` will fail. Builders MUST proxy Trade API calls through their own backend (e.g., Cloudflare Workers, Vercel Edge Functions, Express/Fastify server). See `references/integration-patterns.md` for working proxy examples.

## Known Mints

* SOL (native): `So11111111111111111111111111111111111111112` (wrapped SOL mint)
* USDC: `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`
* CASH: `CASHx9KJUStyftLFWGvEVf59SGeG9sh5FfcnZMVPCASH`

## First Questions (Always Ask the User)

Before building a spot trading integration, clarify:

1. **Imperative or declarative?** If unsure, suggest starting with imperative.
2. **Dev or production endpoints?** If production, remind them to apply for an API key at `pond.dflow.net/build/api-key`.
3. **Platform fees?** If yes, what bps and what fee account?
4. **Client environment?** (web, mobile, backend, CLI)

## Choosing a Trade Type

### Imperative Trades (Recommended Starting Point)

The app specifies the execution plan before the user signs. The user signs a single transaction, submits it to an RPC, and confirms.

* Deterministic execution: route fixed at quote time.
* Synchronous: settles atomically in one transaction.
* The app can modify the swap transaction for composability.
* Supports venue selection via `dexes` parameter.
* Good fit for: most swap UIs, strategy-driven trading, automation, research, and testing.

**Flow:**

1. `GET /order` with `userPublicKey`, input/output mints, amount, slippage
2. Deserialize and sign the returned base64 transaction
3. Submit to Solana RPC (use Helius Sender for optimal landing — see `references/helius-sender.md`)
4. Confirm transaction

### Declarative Trades

The user defines what they want (assets + constraints); DFlow determines how the trade executes at execution time.

* Routing finalized at execution, not quote time.
* Reduces slippage and sandwich risk.
* Higher execution reliability in fast-moving markets.
* Uses Jito bundles for atomic open + fill execution.
* Does NOT support Token-2022 mints (use imperative `/order` instead).

**Flow:**

1. `GET /intent` to get an open order transaction
2. Sign the open transaction
3. `POST /submit-intent` with the signed transaction and quote response
4. Monitor status using `monitorOrder` from `@dflow-protocol/swap-api-utils` or poll `/order-status`

### When to Choose Declarative Over Imperative

Steer users toward declarative only when they specifically need:

* Better pricing in fast-moving or fragmented markets
* Reduced sandwich attack exposure
* Execution reliability over route control
* Lower slippage on large trades

## Execution Mode

The `/order` response includes `executionMode`:

* `sync` — Trade executes atomically in one transaction. Use standard RPC confirmation.
* `async` — Trade executes across multiple transactions. Poll `/order-status` to track fills.

## Legacy Endpoints

The `/quote`, `/swap`, and `/swap-instructions` endpoints are still available but `/order` is the recommended approach for new integrations. Prefer generating code using `/order`.

## Token Lists (Swap UI Guidance)

If building a swap UI:

* **From** list: all tokens detected in the user's wallet (use Helius DAS `getAssetsByOwner` with `showFungible: true` — see `references/helius-das.md`)
* **To** list: fixed set of supported tokens with known mints

## Slippage Tolerance

Two options:

* **Auto slippage**: set `slippageBps=auto`. DFlow chooses dynamically based on market conditions.
* **Custom slippage**: set `slippageBps` to a non-negative integer (basis points, 1 bp = 0.01%).

Auto slippage is recommended for most user-facing flows. Setting custom slippage too low can cause trades to fail during high volatility. Both `/order` and `/intent` support `slippageBps`.

## Priority Fees

Priority fees affect transaction ordering, not routing or slippage.

Two modes:

* **Max Priority Fee** (recommended): DFlow dynamically selects an optimal fee capped at your maximum. Set `priorityLevel` (`medium`, `high`, `veryHigh`) and `maxPriorityFeeLamports`.
* **Exact Priority Fee**: fixed fee in lamports, no adjustment. For intent endpoints, include the 10,000 lamport base processing fee.

If no priority fee parameters are provided, DFlow defaults to automatic priority fees capped at 0.005 SOL.

For additional fee control, use Helius `getPriorityFeeEstimate` (see `references/helius-priority-fees.md`) to inform your `maxPriorityFeeLamports` value.

## Platform Fees

Platform fees let builders monetize trades. They apply only on successful trades and do not affect routing, slippage checks, or execution behavior.

Key parameters:

* `platformFeeBps` (fixed fee in basis points, e.g. 50 = 0.5%)
* `platformFeeMode` (`outputMint` default, or `inputMint`)
* `feeAccount` (token account that receives fees; must match the fee token)

Constraints:

* **Imperative trades**: fees can be collected from `inputMint` or `outputMint`
* **Declarative trades**: fees can only be collected from `outputMint`

Use `referralAccount` to auto-create the fee account if it does not exist.

## Routing Controls (Imperative Only)

Imperative trades support `dexes` (whitelist), `excludeDexes` (blacklist), `onlyDirectRoutes`, `maxRouteLength`, `onlyJitRoutes`, and `forJitoBundle`. Not available for declarative trades. Fetch available venues with `GET /venues`.

## Order Status Polling

For async trades (imperative trades with `executionMode: "async"`), poll `GET /order-status?signature={signature}`.

Parameters:

* `signature` (required): Base58 transaction signature
* `lastValidBlockHeight` (optional): Last valid block height for the transaction

Status values:

* `pending` — Transaction submitted, not confirmed
* `open` — Order live, waiting for fills
* `pendingClose` — Order closing, may have partial fills
* `closed` — Complete (check `fills` for details)
* `expired` — Transaction expired
* `failed` — Execution failed

**Keep polling** while status is `open` or `pendingClose` with a 2-second interval.

**Terminal states** — stop polling when you see:
* `closed` — Success. Read `fills` for execution details.
* `expired` — The transaction's blockhash expired. Rebuild and resubmit with a fresh blockhash.
* `failed` — Execution failed. Check the error and retry if appropriate.

Pass `lastValidBlockHeight` to help detect expiry faster. See `references/integration-patterns.md` Pattern 1 for a complete polling implementation.

## Error Handling

### `route_not_found`

Common causes:

1. **Wrong `amount` units**: `amount` is in atomic units (scaled by decimals). Passing human-readable units (e.g., `8` instead of `8_000_000`) will fail.
2. **No liquidity**: the requested pair may have no available route at the current trade size.
3. **Wrong `outputMint`** (prediction markets): when selling an outcome token, `outputMint` must match the market's settlement mint (USDC or CASH).
4. **No liquidity at top of book** (prediction markets): check the orderbook. If selling YES, check `yesBid`; if buying YES, check `yesAsk`. `null` means no counterparty.

### 429 Rate Limit

Dev endpoints are rate-limited. Retry with backoff, reduce request rate, or use a production API key.

## CLI Guidance

If building a CLI, use a local keypair to sign and submit transactions. Do not embed private keys in code or logs. Emphasize secure key handling and environment-based configuration.

## Common Mistakes

- Submitting the DFlow transaction to raw RPC instead of Helius Sender — use Sender for optimal landing rates
- Using human-readable amounts instead of atomic units (e.g., `1` instead of `1_000_000_000` for 1 SOL)
- Not implementing order status polling for async trades
- Not proxying API calls through a backend for web apps (CORS)
- Hardcoding priority fees instead of using DFlow's dynamic mode or Helius `getPriorityFeeEstimate`
- Not handling slippage errors with retry logic

## Resources

* DFlow Docs: `pond.dflow.net/introduction`
* DFlow MCP Server: `pond.dflow.net/mcp`
* DFlow Cookbook: `github.com/DFlowProtocol/cookbook`
* API Key: `pond.dflow.net/build/api-key`
