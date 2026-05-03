---

name: trails
description: Trails — Cross-chain swap, bridge, and DeFi orchestration via Sequence. Use when an agent wants to swap tokens across chains, bridge assets, fund a Bankr wallet from any chain, deposit into yield vaults (Aave, Morpho), get token prices, discover earn pools, or quote cross-chain routes. Integrates with Bankr submit() for on-chain execution. Also use when asked about Trails, Sequence swaps, cross-chain bridging, or DeFi yield deposits.

# Trails

Cross-chain swap, bridge, and DeFi orchestration powered by Sequence. Agents specify the action — Trails automatically determines the optimal multi-step path across chains.

**API Base:** `https://trails-api.sequence.app/rpc/Trails/<MethodName>`
**Auth Header:** `X-Access-Key: $TRAILS_API`
**Widget:** `https://demo.trails.build/`
**Bankr Integration:** `submit()` from `@bankr/cli` broadcasts on-chain transactions

## Environment Variables

```bash
export TRAILS_API=<your-sequence-project-access-key>   # from https://sequence.build
export BANKR_API_KEY=<your-bankr-key>                  # from bankr.bot/api
```

## Quick Start

### Get a swap quote (USDC on Polygon -> ETH on Base)

```bash
BANKR_WALLET=$(curl -s https://api.bankr.bot/agent/me \
  -H "X-API-Key: $BANKR_API_KEY" \
  | jq -r '.wallets[] | select(.chain == "evm") | .address')

curl -s https://trails-api.sequence.app/rpc/Trails/QuoteIntent \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d "{
    \"ownerAddress\": \"$BANKR_WALLET\",
    \"originChainId\": 137,
    \"originTokenAddress\": \"0x3c499c542cef5e3811e1192ce70d8cC03d5c3359\",
    \"originTokenAmount\": \"10000000\",
    \"destinationChainId\": 8453,
    \"destinationTokenAddress\": \"0x0000000000000000000000000000000000000000\",
    \"destinationTokenAmount\": \"0\",
    \"tradeType\": \"EXACT_INPUT\",
    \"options\": { \"slippageTolerance\": 0.005 }
  }" | jq '.intent.quote'
```

### Discover yield pools

```bash
curl -s https://trails-api.sequence.app/rpc/Trails/GetEarnPools \
  -H "Content-Type: application/json" \
  -H "X-Access-Key: $TRAILS_API" \
  -d '{"chainIds": [137]}' \
  | jq '[.pools[] | select(.isActive and .token.symbol == "USDC")] | sort_by(-.tvl) | .[0]'
```

## Task Guide

### When the user wants to fund a Bankr wallet

Use the Trails widget URL with the Bankr wallet address as `toAddress`. See `references/trails.md` Recipe 1.

### When the user wants to swap tokens (same-chain or cross-chain)

1. Get Bankr wallet address
2. QuoteIntent -> CommitIntent -> submit depositTransaction via Bankr -> ExecuteIntent -> WaitIntentReceipt

See `references/trails.md` Recipe 2.

### When the user wants to deposit into a yield vault

1. GetEarnPools to discover pool addresses and APYs
2. Approve the pool contract via Bankr submit()
3. Deposit via Bankr submit()

See `references/trails.md` Recipe 3.

### When the user asks about supported tokens or chains

Use `GetTokenList` (body: `{"chainIds": [137]}`) or `GetChains`.

### When the user asks about token prices

Use `GetTokenPrices`.

## API Methods


| Group                 | Method                             | Description                                      |
| --------------------- | ---------------------------------- | ------------------------------------------------ |
| **Intent lifecycle**  | `QuoteIntent`                      | Get quote + depositTransaction for a swap/bridge |
|                       | `CommitIntent`                     | Lock the intent, receive intentId                |
|                       | `ExecuteIntent`                    | Notify Trails the deposit tx is mined            |
|                       | `WaitIntentReceipt`                | Poll until intent is complete                    |
| **Intent management** | `GetIntent`                        | Look up intent by ID                             |
|                       | `GetIntentReceipt`                 | Get final receipt                                |
|                       | `SearchIntents`                    | List intents by owner/status                     |
|                       | `GetIntentHistory`                 | Paginated history                                |
|                       | `AbortIntent`                      | Cancel a pending intent                          |
| **Discovery**         | `GetEarnPools`                     | Active yield pools with APY, TVL, depositAddress |
|                       | `GetChains`                        | Supported chains                                 |
|                       | `GetTokenList`                     | Tokens per chain                                 |
|                       | `GetTokenPrices`                   | USD prices                                       |
|                       | `GetExactInputRoutes`              | Preview routes for exact-in                      |
|                       | `GetExactOutputRoutes`             | Preview routes for exact-out                     |
| **Reference**         | `GetExchangeRate`                  | Fiat conversion                                  |
|                       | `GetTrailsContracts`               | Contract addresses per chain                     |
| **Utility**           | `Ping` / `RuntimeStatus` / `Clock` | Health + server time                             |


## Key Notes

- `originTokenAmount` is in base units (e.g. `10000000` = 10 USDC with 6 decimals)
- Use `"0x000...000"` for native token addresses
- `submit` must be imported from `@bankr/cli/dist/lib/api.js` (not re-exported from main entry)
- Add `"destinationToAddress"` to QuoteIntent to send output to a different wallet than the payer

