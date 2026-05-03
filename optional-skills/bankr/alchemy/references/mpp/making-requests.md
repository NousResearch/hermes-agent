# Making Requests

The gateway supports JSON-RPC, NFT, Prices, and Portfolio APIs — all with the same auth and MPP payment flow. See [reference](reference.md) for the full list of supported endpoints, chain network slugs, and API methods.

> **Wallet type vs query chain:** Your wallet type determines which auth scheme (SIWE) to use. The chain URL in your request is independent — you can query any supported chain.

## Recommended: `mppx/client` (Auto-Payment)

The easiest way to make requests is with `mppx/client`, which automatically handles the 402 payment flow.

```bash
npm install mppx viem
```

### Tempo (on-chain USDC)

```typescript
import { Mppx, tempo } from "mppx/client";
import { privateKeyToAccount } from "viem/accounts";
import { createWalletClient, http } from "viem";
import { base } from "viem/chains";
import { createSiweMessage, generateSiweNonce } from "viem/siwe";

const privateKey = process.env.PRIVATE_KEY as `0x${string}`;
const account = privateKeyToAccount(privateKey);

// Generate SIWE auth token
const message = createSiweMessage({
  address: account.address,
  chainId: base.id,
  domain: "mpp.alchemy.com",
  nonce: generateSiweNonce(),
  uri: "https://mpp.alchemy.com",
  version: "1",
  statement: "Sign in to Alchemy Gateway",
  expirationTime: new Date(Date.now() + 60 * 60 * 1000),
});

const walletClient = createWalletClient({ account, chain: base, transport: http() });
const signature = await walletClient.signMessage({ message });
const siweToken = `${btoa(message)}.${signature}`;

// Create mppx client — auto-handles 402 → challenge → credential → retry
const mppx = Mppx.create({
  methods: [tempo.charge({ account })],
  polyfill: false,
});

// Use x-token for SIWE auth because mppx manages the Authorization header
const res = await mppx.fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-token": `SIWE ${siweToken}`,
  },
  body: JSON.stringify({
    id: 1,
    jsonrpc: "2.0",
    method: "eth_blockNumber",
  }),
});

const result = await res.json();
// { id: 1, jsonrpc: "2.0", result: "0x134e82c" }
```

### Stripe (credit card)

Stripe requires a `createToken` callback that proxies through a server endpoint to create SPTs (Shared Payment Tokens). You need a server endpoint that calls the Stripe API with your secret key.

```typescript
import { Mppx, stripe } from "mppx/client";
import { loadStripe } from "@stripe/stripe-js";

const stripeJs = (await loadStripe("pk_test_..."))!;

const mppx = Mppx.create({
  methods: [
    stripe({
      client: stripeJs,
      createToken: async (params) => {
        // Proxy through your server to create an SPT (requires Stripe secret key)
        const res = await fetch("/api/create-spt", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        });
        if (!res.ok) throw new Error("Failed to create SPT");
        return (await res.json()).spt;
      },
      paymentMethod: "pm_card_visa", // or omit to collect via Stripe Elements
    }),
  ],
  polyfill: false,
});

// Use x-token for SIWE auth because mppx manages the Authorization header
const res = await mppx.fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-token": `SIWE ${siweToken}`,
  },
  body: JSON.stringify({
    id: 1,
    jsonrpc: "2.0",
    method: "eth_blockNumber",
  }),
});
```

> **Important: `x-token` header.** When using `mppx/client`, SIWE auth must go via the `x-token` header (not `Authorization`) because `mppx` manages the `Authorization` header for payment credentials.

## How It Works

The MPP payment flow with `mppx/client`:

1. Send the request with SIWE auth via `x-token` header.
2. If **200** — return the result immediately. Response includes `X-Protocol-Version: mpp/1.0` and optionally `Payment-Receipt` headers.
3. If **402** — `mppx/client` automatically reads the `WWW-Authenticate` header, parses the challenge, creates a payment credential, and retries with the credential in the `Authorization` header.
4. Subsequent calls with the same auth token return 200 without payment.

## Manual Flow (Advanced)

For full control over each step, use the low-level API with `Challenge.fromResponse`:

```typescript
import { Challenge } from "mppx";
import { tempo } from "mppx/client";
import { privateKeyToAccount } from "viem/accounts";

const account = privateKeyToAccount(process.env.PRIVATE_KEY as `0x${string}`);

const charge = tempo.charge({ account });

// 1. Get the 402 challenge
const response = await fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: `SIWE ${siweToken}`,
  },
  body: JSON.stringify({ id: 1, jsonrpc: "2.0", method: "eth_blockNumber" }),
});

const challenge = Challenge.fromResponse(response, { methods: [charge] });

// 2. Create credential
const credential = await charge.createCredential({ challenge });

// 3. Retry with credential
const paid = await fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: credential,
  },
  body: JSON.stringify({ id: 1, jsonrpc: "2.0", method: "eth_blockNumber" }),
});
```

## REST API Endpoints (Prices, Portfolio, NFT)

For REST API endpoints like `/prices/v1/tokens/historical`, use `mppx.fetch` (recommended) or plain `fetch` with the `Authorization` header (`SIWE <token>`). The `mppx/client` approach works with all endpoint types.

The auth token alone is sufficient for authentication on all endpoints once payment has been established.

## Response Scenarios

### 200 — Success

```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": "0x134e82c"
}
```

The response includes an `X-Protocol-Version: mpp/1.0` header and optionally a `Payment-Receipt` header.

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

The wallet has no account or credits. The response includes a `WWW-Authenticate` header with payment challenges:

```json
{
  "error": "Payment Required",
  "protocol": "mpp",
  "methods": ["tempo", "stripe"],
  "intent": "charge",
  "challenge": "Payment realm=\"MPP Payment\", method=\"tempo\", ...",
  "extensions": { "hint": "..." }
}
```

## Supported Chains and Endpoints

See [reference](reference.md) for supported chain network slugs, all available routes, and detailed API method documentation.
