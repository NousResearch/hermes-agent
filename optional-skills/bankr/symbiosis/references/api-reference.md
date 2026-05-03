# Symbiosis API Reference

Base URL: `https://api-v2.symbiosis.finance`

## POST /crosschain/v1/swap

Get a cross-chain swap quote with executable calldata.

### Request

```json
{
  "tokenAmountIn": {
    "chainId": 8453,
    "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "decimals": 6,
    "amount": "2000000"
  },
  "tokenOut": {
    "chainId": 137,
    "address": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "decimals": 6
  },
  "from": "0xUserAddress",
  "to": "0xUserAddress",
  "slippage": 200,
  "partnerId": "bankr"
}
```

**Fields:**
- `tokenAmountIn.amount` — amount in smallest units (e.g., "2000000" for 2 USDC with 6 decimals)
- `slippage` — in basis points (200 = 2%, recommended default)
- `partnerId` — always use "bankr"
- `from` / `to` — user's wallet addresses (can be different for cross-chain)

### Response

```json
{
  "tx": {
    "to": "0x691df9C4561d95a4a726313089c8536dd682b946",
    "data": "0x...",
    "value": "0",
    "chainId": 8453
  },
  "approveTo": "0x41Ae964d0F61Bb5F5e253141A462aD6F3b625B92",
  "tokenAmountOut": {
    "address": "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359",
    "amount": "1497245",
    "chainId": 137,
    "decimals": 6
  },
  "fee": {
    "amount": "250000",
    "decimals": 6,
    "priceUsd": 1.0
  },
  "estimatedTime": 32
}
```

**Key fields:**
- `tx` — the swap transaction to submit on the source chain
- `approveTo` — if present, you must approve this spender before the swap
- `tokenAmountOut.amount` — expected output in smallest units
- `estimatedTime` — seconds until destination chain receives funds

### Amount Conversion

| Human Amount | Decimals | Smallest Units |
|-------------|----------|---------------|
| 2 USDC | 6 | "2000000" |
| 0.1 ETH | 18 | "100000000000000000" |
| 100 USDT | 6 | "100000000" |
| 1.5 BNB | 18 | "1500000000000000000" |

Formula: `amount_smallest = human_amount * 10^decimals`

## POST /crosschain/v2/swap

Bitcoin-specific endpoint. Returns a deposit address instead of calldata.

### Response (Bitcoin)

```json
{
  "depositAddress": "bc1q...",
  "depositAddressExpiry": 1234567890,
  "tokenAmountOut": { ... }
}
```

The user sends BTC to `depositAddress` before `depositAddressExpiry`.

## GET /crosschain/v1/chains

Returns all supported chains.

## Cross-Chain Status Tracking

After submitting a swap transaction, track status at:
`https://explorer.symbiosis.finance/transactions/<sourceChainId>/<txHash>`

Typical completion: 15-60 seconds for most routes, up to 10 minutes for congested chains.
