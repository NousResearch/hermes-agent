# Gateway Overview

The Alchemy Agentic Gateway lets agents easily access Alchemy's developer platform.

**If `ALCHEMY_API_KEY` is set**, use standard Alchemy endpoints directly (e.g. `https://{chainNetwork}.g.alchemy.com/v2/$ALCHEMY_API_KEY`). No wallet, auth token, or payment setup is needed. Skip the rest of this document.

**If `ALCHEMY_API_KEY` is NOT set**, proceed directly with wallet-based authentication. Do NOT mention the API key, suggest obtaining one, or list it as an alternative — go straight to MPP wallet setup. The gateway authenticates with SIWE and handles per-request payments via the MPP protocol.

## Payment Methods

MPP supports two payment methods. The user must choose one during setup (see [wallet-bootstrap](wallet-bootstrap.md)):

| Method | How it works | Wallet type | Wallet funding needed? |
|--------|-------------|-------------|----------------------|
| **Tempo** | On-chain USDC payment (gasless) | EVM only (SIWE) | Yes — wallet must hold USDC |
| **Stripe** | Credit card via SPT (Shared Payment Token) | EVM | No — card is charged directly |

Both methods require a wallet for SIWE authentication. Tempo additionally requires the EVM wallet to be funded with USDC. Stripe uses a `createToken` callback that proxies through a server endpoint to create SPTs.

> **Wallet type vs query chain:** Your wallet type determines how you authenticate. It does NOT restrict which chains you can query — a SIWE token works with any supported chain URL. NEVER suggest a wallet type based on the chain being queried.

## Base URL

```
https://mpp.alchemy.com
```

The gateway exposes four API routes:

| Route | Example | Description |
|-------|---------|-------------|
| `/:chainNetwork/v2` | `/eth-mainnet/v2` | Node JSON-RPC + Token API + Transfers API |
| `/:chainNetwork/nft/v3/*` | `/eth-mainnet/nft/v3/getNFTsForOwner?owner=0x...` | NFT API v3 (REST) |
| `/data/v1/*` | `/data/v1/assets/tokens/by-address` | Portfolio API (not chain-specific) |
| `/prices/v1/*` | `/prices/v1/tokens/by-symbol?symbols=ETH` | Prices API (current + historical) |

See [reference](reference.md) for all endpoints, supported chains, and available methods.

## End-to-End Flow

### Tempo (on-chain USDC)

1. **Choose payment method** → Tempo. See [wallet-bootstrap](wallet-bootstrap.md).
2. **Set up an EVM wallet** — Create or import (Tempo requires EVM/SIWE).
3. **Fund the wallet** — Load USDC on an EVM network (e.g. Base Mainnet).
4. **Create an auth token** — Generate a SIWE token using `viem` (see [authentication](authentication.md))
5. **Send a request** — Use `mppx/client` (recommended): create `Mppx.create({ methods: [tempo.charge({ account })] })` and call `mppx.fetch()` with `x-token: SIWE <token>`. The 402 flow is handled automatically.
6. **Receive the result** — Response includes `X-Protocol-Version: mpp/1.0` and `Payment-Receipt`.

### Stripe (credit card)

1. **Choose payment method** → Stripe. See [wallet-bootstrap](wallet-bootstrap.md).
2. **Set up an EVM wallet** — Create or import (needed for auth only — no funding required).
3. **Create an auth token** — Generate a SIWE token using `viem` (see [authentication](authentication.md))
4. **Send a request** — Use `mppx/client` (recommended): create `Mppx.create({ methods: [stripe({ client: stripeJs, createToken, paymentMethod })] })` and call `mppx.fetch()` with `x-token: SIWE <token>`. The 402 flow and SPT creation are handled automatically via the `createToken` callback.
6. **Receive the result** — Response includes `X-Protocol-Version: mpp/1.0` and `Payment-Receipt`.

## Packages

### `mppx` — MPP Client Library (recommended)

```bash
npm install mppx
```

#### `mppx/client` — High-Level Client (recommended for most users)

Auto-handles the 402 payment flow. When using `mppx/client`, SIWE auth must go via the `x-token` header (not `Authorization`) because `mppx` manages the `Authorization` header for payment credentials.

| Export | Purpose |
|--------|---------|
| `Mppx` | Client with `fetch()` that auto-handles 402 challenges |
| `tempo` | Payment method factory for on-chain USDC (Tempo) |
| `stripe` | Payment method factory for credit card (Stripe) |

#### `mppx` — Low-Level Utilities (advanced use cases)

For manual control over the 402 flow:

| Export | Purpose |
|--------|---------|
| `Challenge` | Parse and inspect `WWW-Authenticate` challenges (`Challenge.fromResponseList(response)` or `Challenge.fromHeaders(headers)`) |
| `Credential` | Create and serialize payment credentials (`Credential.from(challenge, opts)`, `Credential.serialize(cred)`) |
| `Receipt` | Parse `Payment-Receipt` headers |

### `@stripe/stripe-js` — Stripe.js (for Stripe payments)

```bash
npm install @stripe/stripe-js
```

Required for collecting card details in the Stripe payment flow. Used to create a Stripe payment method, which is then exchanged for a SPT token.

### `viem` — Wallet & Auth Library

```bash
npm install viem
```

Used for wallet management and SIWE auth token generation:

| Function | Import | Purpose |
|----------|--------|---------|
| `generatePrivateKey()` | `viem/accounts` | Create a new EVM private key |
| `privateKeyToAccount(pk)` | `viem/accounts` | Import / derive address from a private key |
| `createSiweMessage(...)` | `viem/siwe` | Construct a SIWE message for auth |
| `generateSiweNonce()` | `viem/siwe` | Generate a random nonce for SIWE |
| `createWalletClient(...)` | `viem` | Create a client to sign messages |
