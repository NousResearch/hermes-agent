# MPP Payment

The `mppx/client` library handles 402 Payment Required flows automatically — you don't need to manually parse challenges, create credentials, or retry requests. See [making-requests](making-requests.md) for full setup.

## Payment Methods

| Method | Description | Requirements |
|--------|-------------|-------------|
| **Tempo** | On-chain USDC payment (gasless, EVM only) | EVM wallet funded with USDC, SIWE auth |
| **Stripe** | Credit card payment via SPT (Shared Payment Token) | `createToken` callback proxied through a server, EVM wallet for auth |

The user chooses their payment method during setup (see [wallet-bootstrap](wallet-bootstrap.md)).

## How It Works

### Tempo (on-chain USDC)

`tempo.charge` takes a viem `account` and handles the full 402 flow: parsing the challenge, signing a TIP-20 transfer, and retrying with the credential.

```typescript
import { Mppx, tempo } from "mppx/client";
import { privateKeyToAccount } from "viem/accounts";

const account = privateKeyToAccount(process.env.PRIVATE_KEY as `0x${string}`);

const mppx = Mppx.create({
  methods: [tempo.charge({ account })],
  polyfill: false,
});

// mppx.fetch auto-handles the 402 → challenge → credential → retry flow
const res = await mppx.fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-token": `SIWE ${siweToken}`,
  },
  body: JSON.stringify({ id: 1, jsonrpc: "2.0", method: "eth_blockNumber" }),
});
```

### Stripe (credit card)

`stripe` (or `stripe.charge`) takes a `createToken` callback that proxies through a server endpoint to create SPTs. SPT creation requires a Stripe secret key, so it must happen server-side.

```typescript
import { Mppx, stripe } from "mppx/client";
import { loadStripe } from "@stripe/stripe-js";

const stripeJs = (await loadStripe("pk_test_..."))!;

const mppx = Mppx.create({
  methods: [
    stripe({
      client: stripeJs,
      createToken: async (params) => {
        // Proxy through your server (requires Stripe secret key)
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

// mppx.fetch auto-handles the 402 flow using the createToken callback
const res = await mppx.fetch("https://mpp.alchemy.com/eth-mainnet/v2", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "x-token": `SIWE ${siweToken}`,
  },
  body: JSON.stringify({ id: 1, jsonrpc: "2.0", method: "eth_blockNumber" }),
});
```

### SPT creation server endpoint

The `createToken` callback needs a server endpoint because SPT creation requires a Stripe secret key:

```typescript
// /api/create-spt
export async function POST(request: Request) {
  const { paymentMethod, amount, currency, expiresAt, networkId, metadata } =
    await request.json();

  const body = new URLSearchParams({
    payment_method: paymentMethod,
    "usage_limits[currency]": currency,
    "usage_limits[max_amount]": amount,
    "usage_limits[expires_at]": expiresAt.toString(),
  });

  const response = await fetch(
    "https://api.stripe.com/v1/test_helpers/shared_payment/granted_tokens",
    {
      method: "POST",
      headers: {
        Authorization: `Basic ${btoa(`${process.env.STRIPE_SECRET_KEY}:`)}`,
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body,
    },
  );

  if (!response.ok) {
    const error = await response.json();
    return Response.json({ error: error.error.message }, { status: 400 });
  }

  const { id: spt } = await response.json();
  return Response.json({ spt });
}
```

## Payment Details

- **Tempo**: On-chain TIP-20 token transfer on Tempo. Gasless when server uses `feePayer`. Settlement in ~500ms.
- **Stripe**: Card payment via SPT. Settlement through Stripe's payment rails.
- **Amount**: Determined by the gateway per request.
- **Settlement**: The gateway verifies the credential and settles the payment automatically.

## Payment Receipt

After a successful payment, the gateway includes a `Payment-Receipt` header in the response:

```typescript
import { Receipt } from "mppx";

const receipt = Receipt.fromResponse(response);
console.log("Transaction:", receipt?.reference);
```

## Payment Error Responses

If a payment fails, the gateway returns 402 with the original challenge:

```json
{
  "error": "Payment Required",
  "protocol": "mpp",
  "methods": ["tempo"],
  "intent": "charge",
  "challenge": "Payment realm=\"...\", method=\"...\", ..."
}
```

Common issues:
- **Insufficient USDC balance** (Tempo only) — fund the wallet with more USDC
- **Card declined** (Stripe only) — use a different card
- **Invalid SPT** (Stripe only) — the SPT may have expired; the `createToken` callback will be called again on retry
- **Unsupported payment method** — check the `methods` array; Stripe may not be enabled on this gateway
