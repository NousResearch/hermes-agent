# Making Requests

The gateway supports JSON-RPC, NFT, Prices, and Portfolio APIs — all with the same auth and x402 payment flow. See [reference](reference.md) for the full list of supported endpoints, chain network slugs, and API methods.

Use `@alchemy/x402` with `@x402/fetch` or `@x402/axios` to make requests. Both wrappers automatically handle the 402 → sign → retry flow so you don't need to manage payments manually.

> **Wallet type vs query chain:** Your wallet type determines which auth scheme (SIWE/SIWS) and payment client to use. The chain URL in your request is independent — you can query any supported chain with either wallet type.

## Using an EVM Wallet

### Option A: `@x402/fetch`

```bash
npm install @alchemy/x402 @x402/fetch
```

```typescript
import { buildX402Client, signSiwe } from "@alchemy/x402";
import { wrapFetchWithPayment } from "@x402/fetch";

// Read private key from environment — never hardcode it
const privateKey = process.env.PRIVATE_KEY as `0x${string}`;

// Setup (do once)
const client = buildX402Client(privateKey);
const siweToken = await signSiwe({ privateKey });

// Wrap fetch with SIWE auth
const authedFetch: typeof fetch = async (input, init) => {
  const headers = new Headers(init?.headers);
  headers.set("Authorization", `SIWE ${siweToken}`);
  return fetch(input, { ...init, headers });
};

// Wrap with automatic x402 payment handling
const paidFetch = wrapFetchWithPayment(authedFetch, client);

// Make a request — swap the chain URL to query any supported chain
const response = await paidFetch("https://x402.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  body: JSON.stringify({
    id: 1,
    jsonrpc: "2.0",
    method: "eth_blockNumber",
  }),
});

const result = await response.json();
// { id: 1, jsonrpc: "2.0", result: "0x134e82c" }
```

### Option B: `@x402/axios`

```bash
npm install @alchemy/x402 @x402/axios axios
```

```typescript
import axios from "axios";
import { buildX402Client, signSiwe } from "@alchemy/x402";
import { wrapAxiosWithPayment } from "@x402/axios";

// Read private key from environment — never hardcode it
const privateKey = process.env.PRIVATE_KEY as `0x${string}`;

// Setup (do once)
const client = buildX402Client(privateKey);
const siweToken = await signSiwe({ privateKey });
const paidAxios = wrapAxiosWithPayment(axios.create(), client);

// Make a request — swap the chain URL to query any supported chain
const { data } = await paidAxios.post(
  "https://x402.alchemy.com/eth-mainnet/v2",
  {
    id: 1,
    jsonrpc: "2.0",
    method: "eth_blockNumber",
  },
  {
    headers: {
      Authorization: `SIWE ${siweToken}`,
    },
  },
);

// { id: 1, jsonrpc: "2.0", result: "0x134e82c" }
```

## Using a Solana Wallet

### Option A: `@x402/fetch`

```bash
npm install @alchemy/x402 @x402/fetch
```

```typescript
import { buildSolanaX402Client, signSiws } from "@alchemy/x402";
import { wrapFetchWithPayment } from "@x402/fetch";

// Read private key from environment — never hardcode it
const privateKey = process.env.PRIVATE_KEY as string; // base58-encoded

// Setup (do once)
const client = buildSolanaX402Client(privateKey);
const siwsToken = await signSiws({ privateKey });

// Wrap fetch with SIWS auth
const authedFetch: typeof fetch = async (input, init) => {
  const headers = new Headers(init?.headers);
  headers.set("Authorization", `SIWS ${siwsToken}`);
  return fetch(input, { ...init, headers });
};

// Wrap with automatic x402 payment handling
const paidFetch = wrapFetchWithPayment(authedFetch, client);

// Make a request — swap the chain URL to query any supported chain
const response = await paidFetch("https://x402.alchemy.com/solana-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  body: JSON.stringify({
    id: 1,
    jsonrpc: "2.0",
    method: "getSlot",
  }),
});

const result = await response.json();
// { id: 1, jsonrpc: "2.0", result: 123456789 }
```

### Option B: `@x402/axios`

```bash
npm install @alchemy/x402 @x402/axios axios
```

```typescript
import axios from "axios";
import { buildSolanaX402Client, signSiws } from "@alchemy/x402";
import { wrapAxiosWithPayment } from "@x402/axios";

// Read private key from environment — never hardcode it
const privateKey = process.env.PRIVATE_KEY as string; // base58-encoded

// Setup (do once)
const client = buildSolanaX402Client(privateKey);
const siwsToken = await signSiws({ privateKey });
const paidAxios = wrapAxiosWithPayment(axios.create(), client);

// Make a request — swap the chain URL to query any supported chain
const { data } = await paidAxios.post(
  "https://x402.alchemy.com/solana-mainnet/v2",
  {
    id: 1,
    jsonrpc: "2.0",
    method: "getSlot",
  },
  {
    headers: {
      Authorization: `SIWS ${siwsToken}`,
    },
  },
);

// { id: 1, jsonrpc: "2.0", result: 123456789 }
```

## How It Works

Both wrappers follow the same flow:

1. Send the request with the `Authorization` header.
2. If **200** — return the result immediately.
3. If **402** — read the `accepts` array, create a signed USDC payment using the x402 client, and **retry** with a `Payment-Signature` header.
4. Subsequent calls with the same auth token return 200 without payment.

## REST API Endpoints (Prices, Portfolio, NFT)

The `paidFetch`/`paidAxios` wrappers are designed for JSON-RPC endpoints
(`/:chainNetwork/v2`). For REST API POST endpoints like
`/prices/v1/tokens/historical`, use **plain `fetch`** with the
`Authorization` header (`SIWE <token>` or `SIWS <token>` depending on your wallet type) instead. The x402 wrapper can
corrupt POST request bodies on REST endpoints, causing 400 errors.

The auth token alone is sufficient for authentication on all endpoints
once payment has been established (e.g., via an initial GET request
through `paidFetch`).

## Selecting a Payment Network

If the 402 response offers multiple payment networks, `buildX402Client` is pre-configured for both Base Mainnet (`eip155:8453`) and Base Sepolia (`eip155:84532`). `buildSolanaX402Client` is pre-configured for Solana payment networks. The default behavior picks the first compatible option.

## Response Scenarios

### 200 — Success

```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "0x134e82c"
}
```

The response includes an `X-Protocol-Version: x402/2.0` header.

### 401 — Unauthorized

Authentication failed. See [authentication](authentication.md) for error codes.

```json
{
  "error": "Unauthorized",
  "message": "Invalid SIWE message format",
  "code": "INVALID_SIWE"
}
```

### 402 — Payment Required

The wallet has no account or credits. When using `@x402/fetch` or `@x402/axios`, this is handled automatically. The raw 402 body looks like:

```json
{
  "x402Version": 2,
  "error": "PAYMENT-SIGNATURE header is required",
  "resource": {
    "url": "https://x402.alchemy.com/eth-mainnet/v2",
    "description": "Payment required to access this resource",
    "mimeType": "application/json"
  },
  "accepts": [
    {
      "scheme": "exact",
      "network": "eip155:8453",
      "amount": "10000",
      "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
      "payTo": "0x658dc531A7FE637F7BA31C3dDd4C9bf8A27c81e5",
      "maxTimeoutSeconds": 300,
      "extra": {
        "name": "USD Coin",
        "version": "2"
      }
    }
  ]
}
```

## Supported Chains and Endpoints

See [reference](reference.md) for supported chain network slugs, all available routes, and detailed API method documentation.
