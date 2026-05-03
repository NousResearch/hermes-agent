# Reference

## Endpoints

### With API Key (`ALCHEMY_API_KEY` is set)

Use standard Alchemy endpoints directly. No auth headers or payment needed.

| Product | Base URL | Example |
|---------|----------|---------|
| Node JSON-RPC | `https://{chainNetwork}.g.alchemy.com/v2/$ALCHEMY_API_KEY` | `https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` |
| NFT API | `https://{chainNetwork}.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/*` | `https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTsForOwner` |
| Portfolio API | `https://api.g.alchemy.com/data/v1/$ALCHEMY_API_KEY/*` | `https://api.g.alchemy.com/data/v1/$ALCHEMY_API_KEY/assets/tokens/by-address` |
| Prices API | `https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/*` | `https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/by-symbol` |

### Without API Key (MPP gateway)

All require SIWE auth and MPP payment.

| Route | Method | Description |
|-------|--------|-------------|
| `/:chainNetwork/v2` | POST | Node JSON-RPC — standard RPC methods plus Token API, Transfers API, and Simulation API (via JSON-RPC) |
| `/:chainNetwork/nft/v3/*` | GET/POST | [NFT API v3](https://www.alchemy.com/docs/reference/nft-api-quickstart) — REST endpoints for NFT data |
| `/data/v1/assets/*` | POST | [Portfolio API](https://www.alchemy.com/docs/reference/portfolio-apis) — multi-chain portfolio data (not chain-specific) |
| `/prices/v1/tokens/*` | GET/POST | [Prices API](https://www.alchemy.com/docs/reference/prices-api-quickstart) — token prices, historical prices (GET for current by symbol, POST for by-address and historical) |

**Base URL**: `https://mpp.alchemy.com`

Chain-specific routes use the chain slug in the URL (e.g. `https://mpp.alchemy.com/eth-mainnet/v2`). Non-chain-specific routes omit it (e.g. `https://mpp.alchemy.com/data/v1/assets/tokens/by-address`).

---

## API Method Details

The gateway exposes the same API methods, parameters, and response formats as the standard Alchemy APIs. All reference files below use gateway URLs (`mpp.alchemy.com`) and include an `Authorization` header (`SIWE`).

| Gateway route | What to look up | Reference file |
|---|---|---|
| `/:chainNetwork/v2` | `eth_*` methods | [references/node-json-rpc.md](../../references/node-json-rpc.md) |
| `/:chainNetwork/v2` | `alchemy_getTokenBalances`, `alchemy_getTokenMetadata`, `alchemy_getTokenAllowance` | [references/data-token-api.md](../../references/data-token-api.md) |
| `/:chainNetwork/v2` | `alchemy_getAssetTransfers` | [references/data-transfers-api.md](../../references/data-transfers-api.md) |
| `/:chainNetwork/v2` | `alchemy_simulateAssetChanges`, `alchemy_simulateExecution` | [references/data-simulation-api.md](../../references/data-simulation-api.md) |
| `/:chainNetwork/nft/v3/*` | `getNFTsForOwner`, `getNFTMetadata`, etc. | [references/data-nft-api.md](../../references/data-nft-api.md) |
| `/prices/v1/tokens/*` | `tokens/by-symbol`, `tokens/by-address`, `tokens/historical` | [references/data-prices-api.md](../../references/data-prices-api.md) |
| `/data/v1/assets/*` | `assets/tokens/by-address`, `assets/nfts/by-address`, etc. | [references/data-portfolio-apis.md](../../references/data-portfolio-apis.md) |

---

## Chain Network Slugs

Use these as the `:chainNetwork` path parameter for chain-specific routes (`/v2` and `/nft/v3`). Any chain can be queried with a SIWE auth token — the chain URL is independent of wallet type.

| Chain | Mainnet | Testnet |
|-------|---------|---------|
| Ethereum | `eth-mainnet` | `eth-sepolia` |
| Base | `base-mainnet` | `base-sepolia` |
| Polygon | `polygon-mainnet` | `polygon-amoy` |
| BNB | `bnb-mainnet` | `bnb-testnet` |
| Arbitrum | `arb-mainnet` | `arb-sepolia` |
| Optimism | `opt-mainnet` | `opt-sepolia` |
| World Chain | `worldchain-mainnet` | `worldchain-sepolia` |
| Tempo | `tempo-mainnet` | `tempo-moderato` |
| Hyperliquid | `hyperliquid-mainnet` | `hyperliquid-testnet` |
| MegaETH | `megaeth-mainnet` | `megaeth-testnet` |
| Monad | `monad-mainnet` | `monad-testnet` |
| Solana | `solana-mainnet` | `solana-devnet` |

## Payment Networks

Payments are made on these networks (independent of which chain you're querying):

### Tempo Payment Networks (EVM only)

Tempo uses on-chain USDC on EVM networks. Tempo requires an EVM wallet (SIWE auth).

| Network | CAIP-2 ID | USDC Address | EIP-712 Domain Name |
|---------|-----------|--------------|---------------------|
| Base Mainnet | `eip155:8453` | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` | `USD Coin` |

### Stripe Payments (credit card)

Stripe payments use a credit card via Stripe.js. No USDC funding is needed. The flow requires:
1. Collect card details via Stripe.js to obtain a payment method
2. Exchange the payment method for a SPT (Stripe Payment Token) via `mpp.alchemy.com/mpp/spt`
3. Use the SPT to create a payment credential in response to a 402

The 402 response `methods` array will include `"stripe"` when Stripe is available.

## Request Headers (Client → Gateway)

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Auth: `SIWE <base64(siwe_message)>.<signature>`. On payment: append `, Payment <credential>` |
| `x-token` | Alternative | Send the SIWE auth token here instead of `Authorization` when using the mppx SDK (see note below) |
| `Content-Type` | Yes | `application/json` |
| `Accept` | Recommended | `application/json` |

### Resolving the Authorization Header Conflict

MPP uses the `Authorization` header for both SIWE auth and payment credentials. This creates a conflict because the mppx SDK replaces the `Authorization` header with the payment credential on retry. Two approaches are supported:

1. **Multi-scheme Authorization (RFC 9110)** — Combine both in one header, comma-separated:
   ```
   Authorization: SIWE <token>, Payment <credential>
   ```
   The gateway parses comma-separated schemes and extracts each one independently. This works for manual flows (curl, custom fetch).

2. **`x-token` header (recommended for SDK usage)** — Send the SIWE auth via the `x-token` header instead, freeing the `Authorization` header for the mppx SDK to manage:
   ```
   x-token: SIWE <token>
   Authorization: Payment <credential>
   ```
   The gateway checks `x-token` first, then falls back to `Authorization` for auth extraction. **Use this approach when the mppx SDK handles the payment retry automatically**, since it will overwrite the `Authorization` header with the payment credential.

## Response Headers (Gateway → Client)

| Header | When | Description |
|--------|------|-------------|
| `X-Protocol-Version` | Always on success | `mpp/1.0` |
| `WWW-Authenticate` | 402 responses | Payment challenge(s) — one or more serialized MPP challenges |
| `Payment-Receipt` | After successful payment | Serialized payment receipt (transaction reference, network) |

## HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Request proxied successfully |
| 401 | SIWE authentication failed (see [authentication](authentication.md) for error codes) |
| 402 | Payment required — respond with a payment credential in the `Authorization` header |
| 404 | Invalid chain network slug or route |
| 500 | Internal gateway error |
