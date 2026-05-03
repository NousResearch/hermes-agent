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

### Without API Key (x402 gateway)

All require SIWE or SIWS auth and x402 payment.

| Route | Method | Description |
|-------|--------|-------------|
| `/:chainNetwork/v2` | POST | Node JSON-RPC — standard RPC methods plus Token API, Transfers API, and Simulation API (via JSON-RPC) |
| `/:chainNetwork/nft/v3/*` | GET/POST | [NFT API v3](https://www.alchemy.com/docs/reference/nft-api-quickstart) — REST endpoints for NFT data |
| `/data/v1/assets/*` | POST | [Portfolio API](https://www.alchemy.com/docs/reference/portfolio-apis) — multi-chain portfolio data (not chain-specific) |
| `/prices/v1/tokens/*` | GET/POST | [Prices API](https://www.alchemy.com/docs/reference/prices-api-quickstart) — token prices, historical prices (GET for current by symbol, POST for by-address and historical) |

**Base URL**: `https://x402.alchemy.com`

Chain-specific routes use the chain slug in the URL (e.g. `https://x402.alchemy.com/eth-mainnet/v2`). Non-chain-specific routes omit it (e.g. `https://x402.alchemy.com/data/v1/assets/tokens/by-address`).

---

## API Method Details

The gateway exposes the same API methods, parameters, and response formats as the standard Alchemy APIs. All reference files below use gateway URLs (`x402.alchemy.com`) and include an `Authorization` header (`SIWE` or `SIWS` depending on wallet type).

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

Use these as the `:chainNetwork` path parameter for chain-specific routes (`/v2` and `/nft/v3`). Any chain can be queried with either a SIWE or SIWS auth token — the chain URL is independent of wallet type.

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

### EVM Payment Networks

| Network | CAIP-2 ID | USDC Address | EIP-712 Domain Name |
|---------|-----------|--------------|---------------------|
| Base Sepolia (testnet) | `eip155:84532` | `0x036CbD53842c5426634e7929541eC2318f3dCF7e` | `USDC` |
| Base Mainnet | `eip155:8453` | `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913` | `USD Coin` |

### SVM Payment Networks

Solana wallets pay with USDC on Solana. The payment network is determined by the 402 response `accepts` array; the CLI and `createSolanaPayment()` handle selection automatically.

**Gateway receiving wallet (EVM)**: `0x658dc531A7FE637F7BA31C3dDd4C9bf8A27c81e5`

## Request Headers (Client → Gateway)

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | EVM: `SIWE <base64(siwe_message)>.<signature>`; Solana: `SIWS <base64(siws_message)>.<base58_signature>` |
| `Content-Type` | Yes | `application/json` |
| `Accept` | Recommended | `application/json` |
| `Payment-Signature` | On payment | Base64-encoded x402 payment payload |

## Response Headers (Gateway → Client)

| Header | When | Description |
|--------|------|-------------|
| `X-Protocol-Version` | Always on success | `x402/2.0` |
| `PAYMENT-REQUIRED` | 402 responses | Encoded payment requirements |
| `PAYMENT-RESPONSE` | After successful payment | Encoded settlement result (transaction hash, network, payer) |

## HTTP Status Codes

| Status | Meaning |
|--------|---------|
| 200 | Request proxied successfully |
| 401 | SIWE/SIWS authentication failed (see [authentication](authentication.md) for error codes) |
| 402 | Payment required — respond with a `Payment-Signature` header |
| 404 | Invalid chain network slug or route |
| 500 | Internal gateway error |

## Testnet Funding

For development on Base Sepolia:

1. Get testnet ETH from a Base Sepolia faucet for gas
2. Get testnet USDC from the [Circle USDC faucet](https://faucet.circle.com/) — select "Base Sepolia"
3. USDC address on Base Sepolia: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`
