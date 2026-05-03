---
name: alchemy
description: Blockchain API access via Alchemy. Use when an agent needs to query blockchain data (balances, token prices, NFT ownership, transfer history, transaction simulation, gas estimates) across Ethereum, Base, Arbitrum, BNB, Polygon, Solana, and more. Supports API key access ($ALCHEMY_API_KEY), x402 wallet-based pay-per-request (SIWE/SIWS + USDC), and MPP protocol (SIWE + Tempo/Stripe). Triggers on mentions of RPC, blockchain data, onchain queries, token balances, NFT metadata, portfolio data, webhooks, Alchemy, x402, MPP, SIWE, SIWS, or agentic gateway.
license: MIT
compatibility: Requires network access. API key path needs $ALCHEMY_API_KEY. x402/MPP paths need Node.js and a wallet funded with USDC. Works across Claude.ai, Claude Code, and API.
metadata:
  author: alchemyplatform
  version: "2.0"
---

# Alchemy: Blockchain Data Access for Agents

Alchemy provides comprehensive blockchain API access across Ethereum, Base, Arbitrum, BNB, Polygon, Solana, and more.

Three ways to access:

- **API key**: Set `$ALCHEMY_API_KEY` and make requests directly. Full access to all products. Create a free key at [dashboard.alchemy.com](https://dashboard.alchemy.com/).
- **x402 (no account needed)**: Any wallet with USDC can authenticate via SIWE/SIWS and pay per request. Supports EVM and Solana wallets. Install `@alchemy/x402` and `@x402/fetch`.
- **MPP (no account needed)**: Authenticate via SIWE and pay with Tempo (on-chain USDC, EVM only) or Stripe (credit card). Install `mppx`.

## Access Method Selection (Required)

Before the first network call, determine which access method to use:

1. **Is `ALCHEMY_API_KEY` set?** → Use the API Key path. Skip to [API Key Access](#api-key-access).
2. **No API key?** → Ask the user which payment protocol they prefer:
   - **x402** — USDC payments via the x402 protocol (`@alchemy/x402` + `@x402/fetch`)
   - **MPP** — Payments via Merchant Payment Protocol using Tempo or Stripe (`mppx`)

Do NOT pick a protocol on behalf of the user. Wait for their explicit choice.

Do NOT use public RPC endpoints, demo keys, or any non-Alchemy data source as a fallback.

---

## API Key Access

If `$ALCHEMY_API_KEY` is set, use standard Alchemy endpoints directly:

### Base URLs + Auth
| Product | Base URL | Auth | Notes |
| --- | --- | --- | --- |
| Ethereum RPC (HTTPS) | `https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Standard EVM reads and writes. |
| Ethereum RPC (WSS) | `wss://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Subscriptions and realtime. |
| Base RPC (HTTPS) | `https://base-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | EVM L2. |
| Base RPC (WSS) | `wss://base-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Subscriptions and realtime. |
| Arbitrum RPC (HTTPS) | `https://arb-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | EVM L2. |
| Arbitrum RPC (WSS) | `wss://arb-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Subscriptions and realtime. |
| BNB RPC (HTTPS) | `https://bnb-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | EVM L1. |
| BNB RPC (WSS) | `wss://bnb-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Subscriptions and realtime. |
| Solana RPC (HTTPS) | `https://solana-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY` | API key in URL | Solana JSON-RPC. |
| Solana Yellowstone gRPC | `https://solana-mainnet.g.alchemy.com` | `X-Token: $ALCHEMY_API_KEY` | gRPC streaming (Yellowstone). |
| NFT API | `https://<network>.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY` | API key in URL | NFT ownership and metadata. |
| Prices API | `https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY` | API key in URL | Prices by symbol or address. |
| Portfolio API | `https://api.g.alchemy.com/data/v1/$ALCHEMY_API_KEY` | API key in URL | Multi-chain wallet views. |
| Notify API | `https://dashboard.alchemy.com/api` | `X-Alchemy-Token: <ALCHEMY_NOTIFY_AUTH_TOKEN>` | Generate token in dashboard. |

---

## x402 Access (No Account Needed)

x402 is ideal for autonomous agents. No signup, no API keys. Pay with USDC on EVM or Solana.

- **Gateway URL**: `https://x402.alchemy.com`
- **SIWE/SIWS domain**: `x402.alchemy.com`
- **Payment header**: `Payment-Signature: <base64>`
- **Auth**: SIWE (EVM) or SIWS (Solana)

For full setup and wallet bootstrapping, see:
- [references/x402/overview.md](references/x402/overview.md) — End-to-end flow and packages
- [references/x402/wallet-bootstrap.md](references/x402/wallet-bootstrap.md) — Wallet setup and USDC funding
- [references/x402/authentication.md](references/x402/authentication.md) — SIWE/SIWS token creation
- [references/x402/making-requests.md](references/x402/making-requests.md) — Sending requests with `@x402/fetch`
- [references/x402/curl-workflow.md](references/x402/curl-workflow.md) — Quick RPC calls via curl
- [references/x402/payment.md](references/x402/payment.md) — Payment creation from a 402 response
- [references/x402/reference.md](references/x402/reference.md) — Endpoints, networks, headers, status codes

---

## MPP Access (No Account Needed)

MPP supports Tempo (on-chain USDC, EVM only) and Stripe (credit card) payments.

- **Gateway URL**: `https://mpp.alchemy.com`
- **SIWE domain**: `mpp.alchemy.com`
- **Payment header**: `Authorization: Payment <credential>`
- **Auth**: SIWE only (EVM)

For full setup, see:
- [references/mpp/overview.md](references/mpp/overview.md) — End-to-end flow and packages
- [references/mpp/wallet-bootstrap.md](references/mpp/wallet-bootstrap.md) — Wallet setup and funding
- [references/mpp/authentication.md](references/mpp/authentication.md) — SIWE token creation
- [references/mpp/making-requests.md](references/mpp/making-requests.md) — Sending requests with `mppx`
- [references/mpp/curl-workflow.md](references/mpp/curl-workflow.md) — Quick RPC calls via curl
- [references/mpp/payment.md](references/mpp/payment.md) — Payment creation from a 402 response
- [references/mpp/reference.md](references/mpp/reference.md) — Endpoints, networks, headers, status codes

---

## Protocol Comparison

| Aspect | API Key | x402 | MPP |
|--------|---------|------|-----|
| Gateway URL | `*.g.alchemy.com/v2/$KEY` | `https://x402.alchemy.com` | `https://mpp.alchemy.com` |
| Auth | API key in URL | SIWE (EVM) or SIWS (Solana) | SIWE only (EVM) |
| Payment | None (free tier available) | USDC via EIP-3009 or SVM x402 | Tempo (USDC) or Stripe (card) |
| Wallet support | N/A | EVM + Solana | EVM only |
| Client library | curl / any HTTP client | `@alchemy/x402`, `@x402/fetch` | `mppx`, `viem` |
| Setup | Get key at dashboard.alchemy.com | Fund wallet with USDC | Fund wallet or add card |

---

## Endpoint Selector (Top Tasks)

| You need | Use this | Reference |
| --- | --- | --- |
| EVM read/write | JSON-RPC `eth_*` | `references/node-json-rpc.md` |
| Realtime events | `eth_subscribe` | `references/node-websocket-subscriptions.md` |
| Token balances | `alchemy_getTokenBalances` | `references/data-token-api.md` |
| Token metadata | `alchemy_getTokenMetadata` | `references/data-token-api.md` |
| Transfers history | `alchemy_getAssetTransfers` | `references/data-transfers-api.md` |
| NFT ownership | `GET /getNFTsForOwner` | `references/data-nft-api.md` |
| NFT metadata | `GET /getNFTMetadata` | `references/data-nft-api.md` |
| Prices (spot) | `GET /tokens/by-symbol` | `references/data-prices-api.md` |
| Prices (historical) | `POST /tokens/historical` | `references/data-prices-api.md` |
| Portfolio (multi-chain) | `POST /assets/*/by-address` | `references/data-portfolio-apis.md` |
| Simulate tx | `alchemy_simulateAssetChanges` | `references/data-simulation-api.md` |
| Create webhook | `POST /create-webhook` | `references/webhooks-details.md` |
| Solana NFT data | `getAssetsByOwner` (DAS) | `references/solana-das-api.md` |

## Quickstart Examples

### EVM JSON-RPC (Read)
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_blockNumber","params":[]}'
```

### Token Balances
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_getTokenBalances","params":["0x00000000219ab540356cbb839cbe05303d7705fa"]}'
```

### Transfer History
```bash
curl -s https://eth-mainnet.g.alchemy.com/v2/$ALCHEMY_API_KEY \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"alchemy_getAssetTransfers","params":[{"fromBlock":"0x0","toBlock":"latest","toAddress":"0x00000000219ab540356cbb839cbe05303d7705fa","category":["erc20"],"withMetadata":true,"maxCount":"0x3e8"}]}'
```

### NFT Ownership
```bash
curl -s "https://eth-mainnet.g.alchemy.com/nft/v3/$ALCHEMY_API_KEY/getNFTsForOwner?owner=0x00000000219ab540356cbb839cbe05303d7705fa"
```

### Prices (Spot)
```bash
curl -s "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/by-symbol?symbols=ETH&symbols=USDC"
```

### Prices (Historical)
```bash
curl -s -X POST "https://api.g.alchemy.com/prices/v1/$ALCHEMY_API_KEY/tokens/historical" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"ETH","startTime":"2024-01-01T00:00:00Z","endTime":"2024-01-02T00:00:00Z"}'
```

### Create Notify Webhook
```bash
curl -s -X POST "https://dashboard.alchemy.com/api/create-webhook" \
  -H "Content-Type: application/json" \
  -H "X-Alchemy-Token: $ALCHEMY_NOTIFY_AUTH_TOKEN" \
  -d '{"network":"ETH_MAINNET","webhook_type":"ADDRESS_ACTIVITY","webhook_url":"https://example.com/webhook","addresses":["0x00000000219ab540356cbb839cbe05303d7705fa"]}'
```

### Verify Webhook Signature (Node)
```ts
import crypto from "crypto";

export function verify(rawBody: string, signature: string, secret: string) {
  const hmac = crypto.createHmac("sha256", secret).update(rawBody).digest("hex");
  return crypto.timingSafeEqual(Buffer.from(hmac), Buffer.from(signature));
}
```

---

## Network Naming Rules
- Data APIs and JSON-RPC use lowercase network enums like `eth-mainnet`.
- Notify API uses uppercase enums like `ETH_MAINNET`.

## Pagination + Limits
| Endpoint | Limit | Notes |
| --- | --- | --- |
| `alchemy_getTokenBalances` | `maxCount` <= 100 | Use `pageKey` for pagination. |
| `alchemy_getAssetTransfers` | `maxCount` default `0x3e8` | Use `pageKey` for pagination. |
| Portfolio token balances | 3 address/network pairs, 20 networks total | `pageKey` supported. |
| Portfolio NFTs | 2 address/network pairs, 15 networks each | `pageKey` supported. |
| Prices by address | 25 addresses, 3 networks | POST body `addresses[]`. |
| Transactions history (beta) | 1 address/network pair, 2 networks | ETH and BASE mainnets only. |

## Common Token Addresses
| Token | Chain | Address |
| --- | --- | --- |
| ETH | ethereum | `0x0000000000000000000000000000000000000000` |
| WETH | ethereum | `0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2` |
| USDC | ethereum | `0xA0b86991c6218b36c1d19d4a2e9eb0ce3606eB48` |
| USDC | base | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` |

## Failure Modes + Retries
- HTTP `429` means rate limit. Use exponential backoff with jitter.
- JSON-RPC errors come in `error` fields even with HTTP 200.
- Use `pageKey` to resume pagination after failures.
- De-dupe websocket events on reconnect.

## Hard Requirements

- NEVER use public RPC endpoints, demo keys, or any non-Alchemy data source as a fallback.
- NEVER use Read, Write, or Edit tools on files that may contain private keys.
- NEVER correlate wallet type with query chain — wallet type and the chain being queried are independent.
- When no wallet is configured, present ALL wallet options (EVM create, EVM import, Solana create, Solana import) in a single prompt.

## Skill Map

For the complete index of all reference files organized by product area, see `references/skill-map.md`.

- **Node**: JSON-RPC, WebSocket, Debug, Trace, Enhanced APIs, Utility
- **Data**: NFT, Portfolio, Prices, Simulation, Token, Transfers
- **Webhooks**: Address Activity, Custom (GraphQL), NFT Activity, Payloads, Signatures
- **Solana**: JSON-RPC, DAS, Yellowstone gRPC (streaming), Wallets
- **Wallets**: Account Kit, Bundler, Gas Manager, Smart Wallets
- **Rollups**: L2/L3 deployment overview
- **Recipes**: 10 end-to-end integration workflows
- **Operational**: Auth, Rate Limits, Monitoring, Best Practices
- **x402 Protocol**: Wallet bootstrap, auth, making requests, payments
- **MPP Protocol**: Wallet bootstrap, auth, making requests, payments

## Troubleshooting

### API key not working
- Verify `$ALCHEMY_API_KEY` is set: `echo $ALCHEMY_API_KEY`
- Confirm the key is valid at [dashboard.alchemy.com](https://dashboard.alchemy.com/)
- Check if allowlists restrict the key to specific IPs/domains

### HTTP 429 (Rate Limited)
- Use exponential backoff with jitter before retrying
- Check your compute unit budget in the Alchemy dashboard
- See `references/operational-rate-limits-and-compute-units.md` for limits per plan

### 401 Unauthorized (x402/MPP)
- `MISSING_AUTH`: Add the appropriate auth header for your protocol
- `MESSAGE_EXPIRED`: Regenerate your SIWE/SIWS token
- `INVALID_DOMAIN`: Ensure domain matches your protocol (`x402.alchemy.com` or `mpp.alchemy.com`)

### 402 Payment Required (x402/MPP)
- **x402**: Extract `PAYMENT-REQUIRED` header, run `npx @alchemy/x402 pay`, retry with `Payment-Signature` header
- **MPP**: Extract `WWW-Authenticate` header, create credential with `mppx`, retry with `Payment` credential

### Wrong network slug
- Data APIs and JSON-RPC use lowercase: `eth-mainnet`, `base-mainnet`
- Notify API uses uppercase: `ETH_MAINNET`, `BASE_MAINNET`
- See `references/operational-supported-networks.md` for the full list

## Official Links
- [Developer docs](https://www.alchemy.com/docs)
- [Get Started guide](https://www.alchemy.com/docs/get-started)
- [Create a free API key](https://dashboard.alchemy.com)
- [Alchemy Skills repo](https://github.com/alchemyplatform/skills)
