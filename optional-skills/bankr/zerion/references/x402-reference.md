# Zerion x402 Reference

## Overview

x402 enables pay-per-request API access without signup or API keys. Agents pay $0.01 USDC on Base per request.

## How It Works

1. **Request**: Agent sends GET request to Zerion API
2. **402 Response**: Server returns payment requirements
3. **Payment**: Agent signs ERC-3009 USDC authorization
4. **Retry**: Agent retries with payment header
5. **Data**: Server validates payment, returns data

## Payment Flow

### Step 1: Initial Request

```http
GET /v1/wallets/0x.../portfolio HTTP/1.1
Host: api.zerion.io
```

### Step 2: 402 Response

```http
HTTP/1.1 402 Payment Required
Content-Type: application/json

{
  "x402": {
    "version": "1.0",
    "payment": {
      "chain": "base",
      "token": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
      "amount": "10000",
      "recipient": "0x...",
      "validAfter": 0,
      "validBefore": 1711036800,
      "nonce": "0x..."
    }
  }
}
```

### Step 3: Sign Payment (ERC-3009)

```typescript
import { signTypedData } from 'viem/accounts';

const signature = await signTypedData({
  domain: {
    name: 'USD Coin',
    version: '2',
    chainId: 8453, // Base
    verifyingContract: '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913'
  },
  types: {
    TransferWithAuthorization: [
      { name: 'from', type: 'address' },
      { name: 'to', type: 'address' },
      { name: 'value', type: 'uint256' },
      { name: 'validAfter', type: 'uint256' },
      { name: 'validBefore', type: 'uint256' },
      { name: 'nonce', type: 'bytes32' }
    ]
  },
  primaryType: 'TransferWithAuthorization',
  message: paymentDetails
});
```

### Step 4: Retry with Payment

```http
GET /v1/wallets/0x.../portfolio HTTP/1.1
Host: api.zerion.io
X-402-Payment: {"signature":"0x...","payment":{...}}
```

### Step 5: Success Response

```http
HTTP/1.1 200 OK
Content-Type: application/json
X-402-Receipt: {"txHash":"0x...","amount":"10000"}

{
  "data": {
    "type": "portfolio",
    ...
  }
}
```

## Using zerion-cli with x402

```bash
# Single request
zerion-cli wallet portfolio 0x... --x402

# Set as default
export ZERION_X402=true
zerion-cli wallet portfolio 0x...
```

## TypeScript SDK

```typescript
import { createZerionX402Client } from '@zerion/x402';

const client = await createZerionX402Client({
  network: 'base',
  privateKey: process.env.PRIVATE_KEY
});

const portfolio = await client.fetch('/v1/wallets/0x.../portfolio');
```

## Pricing

| Request Type | Cost |
|--------------|------|
| Any API call | $0.01 USDC |

Payment is per successful request. Failed requests (4xx, 5xx) are not charged.

## Supported Chains for Payment

Currently Base only. Payment happens on Base, but you can query data from all 41+ chains.

## Error Handling

```typescript
try {
  const data = await client.fetch('/v1/wallets/0x.../portfolio');
} catch (error) {
  if (error.code === 'INSUFFICIENT_BALANCE') {
    // Agent needs more USDC on Base
  }
  if (error.code === 'PAYMENT_FAILED') {
    // Transaction failed - retry
  }
}
```

## Requirements

- Wallet with USDC on Base
- Private key access for signing
- Base RPC endpoint (for payment submission)

## Resources

- x402 Protocol Spec: https://x402.org
- Zerion x402 Docs: https://developers.zerion.io/reference/x402
- ERC-3009 Standard: https://eips.ethereum.org/EIPS/eip-3009
