# x402 Reference

x402 enables pay-per-request RPC access via USDC micropayments. No API key or signup required — authenticate with SIWE (Ethereum) or SIWX (multi-chain including Solana), purchase credits with USDC, and access 140+ blockchain RPC endpoints.

## Overview

| Property | Value |
|----------|-------|
| **Protocol** | HTTP 402 Payment Required |
| **Payment Method** | USDC on Base, Polygon, or Solana |
| **Authentication** | SIWE / SIWX (EVM + Solana) |
| **Chains** | 140+ (same as Quicknode RPC network) |
| **Base URL** | `https://x402.quicknode.com` |
| **Use Cases** | Keyless RPC access, AI agents, pay-as-you-go, ephemeral wallets |

## How It Works

1. **Authenticate** — Sign a SIWE or SIWX message with your wallet to get a JWT session token
2. **Make RPC calls** — Send JSON-RPC requests to `POST /:network` with your JWT in the `Authorization: Bearer` header
3. **Pay when prompted** — When credits run out, the server returns HTTP 402. The `@quicknode/x402` package automatically signs a USDC payment and retries
4. **Repeat** — Credits are consumed (1 per successful JSON-RPC response). When exhausted, another 402 triggers a new payment automatically

## Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth` | POST | Authenticate via SIWE/SIWX, returns JWT session token |
| `/credits` | GET | Check credit balance |
| `/drip` | POST | Testnet faucet — free credits (Base Sepolia only) |
| `/:network` | POST | JSON-RPC request to a specific chain (e.g., `/ethereum-mainnet`) |
| `/:network/ws` | WebSocket | WebSocket RPC connection to a specific chain |
| `/discovery/resources` | GET | Bazaar-compatible catalog of all supported networks (public) |

## Credit Pricing

| Environment | CAIP-2 Chain ID | Credits | Cost |
|-------------|-----------------|---------|------|
| Base Sepolia (testnet) | eip155:84532 | 100 | $0.01 USDC |
| Base Mainnet | eip155:8453 | 1,000,000 | $10 USDC |
| Polygon Amoy (testnet) | eip155:80002 | 100 | $0.01 USDC |
| Polygon Mainnet | eip155:137 | 1,000,000 | $10 USDC |
| Solana Devnet | solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1 | 100 | $0.01 USDC |
| Solana Mainnet | solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp | 1,000,000 | $10 USDC |

1 credit per successful JSON-RPC response. Error responses are not metered.

## Recommended: @quicknode/x402 Package

The official `@quicknode/x402` package handles SIWX authentication, x402 USDC payments, JWT session management, and reconnection automatically.

```bash
npm install @quicknode/x402
```

```typescript
import { createQuicknodeX402Client } from '@quicknode/x402';

const client = await createQuicknodeX402Client({
  baseUrl: 'https://x402.quicknode.com',
  network: 'eip155:84532',        // pay on Base Sepolia (testnet)
  evmPrivateKey: '0xYOUR_KEY',
  preAuth: true,                   // handles auth, funding, and payments automatically
});

// client.fetch handles auth, SIWX, and payment automatically
const response = await client.fetch('https://x402.quicknode.com/base-sepolia', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'eth_blockNumber', params: [] }),
});
```

**`preAuth` option:** When `preAuth: true`, the client performs SIWX authentication and obtains a JWT upfront before any RPC call. This means when a 402 occurs, it can immediately submit payment (auth → pay). Without `preAuth`, the client sees payment requirements first, then authenticates, then pays — an extra round-trip (pay requirements → auth → pay).

## Alternative: @x402/fetch (Manual Setup)

For more control, use the lower-level `@x402/fetch` packages directly.

### EVM Setup

```bash
npm install @x402/fetch @x402/evm viem siwe
```

```typescript
import { createWalletClient, http } from 'viem';
import { privateKeyToAccount } from 'viem/accounts';
import { baseSepolia } from 'viem/chains';
import { SiweMessage, generateNonce } from 'siwe';
import { wrapFetchWithPayment, x402Client } from '@x402/fetch';
import { ExactEvmScheme, toClientEvmSigner } from '@x402/evm';

const BASE_URL = 'https://x402.quicknode.com';

// 1. Set up wallet
const walletClient = createWalletClient({
  account: privateKeyToAccount('0xYOUR_PRIVATE_KEY'),
  chain: baseSepolia,
  transport: http(),
});

// 2. Authenticate with SIWE
const siweMessage = new SiweMessage({
  domain: 'x402.quicknode.com',
  address: walletClient.account.address,
  statement: 'I accept the Quicknode Terms of Service: https://www.quicknode.com/terms',
  uri: BASE_URL,
  version: '1',
  chainId: 84532,
  nonce: generateNonce(),
  issuedAt: new Date().toISOString(),
});

const message = siweMessage.prepareMessage();
const signature = await walletClient.signMessage({ message });

const authResponse = await fetch(`${BASE_URL}/auth`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message, signature }),
});
const { token } = await authResponse.json();

// 3. Create x402-enabled fetch
const evmSigner = toClientEvmSigner({
  address: walletClient.account.address,
  signTypedData: (params) => walletClient.signTypedData(params),
});

const client = new x402Client()
  .register('eip155:84532', new ExactEvmScheme(evmSigner));

// IMPORTANT: @x402/fetch passes a Request object (not url+init) on payment
// retries. The inner fetch must handle both calling conventions.
const authedFetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  if (input instanceof Request) {
    const req = input.clone();
    req.headers.set('Authorization', `Bearer ${token}`);
    return fetch(req);
  }
  const headers = new Headers(init?.headers);
  headers.set('Authorization', `Bearer ${token}`);
  return fetch(input, { ...init, headers });
};

const x402Fetch = wrapFetchWithPayment(authedFetch, client);

// 4. Make RPC calls — payment is automatic on 402
const response = await x402Fetch(`${BASE_URL}/base-sepolia`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'eth_blockNumber', params: [] }),
});
```

### Solana Setup

```bash
npm install @x402/fetch @x402/svm @solana/kit tweetnacl bs58
```

```typescript
import { createKeyPairSignerFromBytes } from '@solana/kit';
import { wrapFetchWithPayment, x402Client } from '@x402/fetch';
import { ExactSvmScheme } from '@x402/svm';

// Create a Solana signer from your secret key (64 bytes)
const signer = await createKeyPairSignerFromBytes(secretKey);

// Register the signer for Solana Devnet
const client = new x402Client()
  .register('solana:EtWTRABZaYq6iMfeYKouRu166VU2xqa1', new ExactSvmScheme(signer));

const x402Fetch = wrapFetchWithPayment(authedFetch, client);

const response = await x402Fetch('https://x402.quicknode.com/solana-devnet', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'getBlockHeight', params: [] }),
});
```

## Authentication

All endpoints except `/auth` and `/discovery/resources` require a JWT Bearer token. Three auth paths are supported:

### Path 1: Legacy SIWE (EVM only)

```json
POST /auth
{ "message": "<SIWE string>", "signature": "0x<hex>" }
```

### Path 2: SIWX/EVM

```json
POST /auth
{ "message": "<SIWE string>", "signature": "0x<hex>", "type": "siwx" }
```

### Path 3: SIWX/Solana

```json
POST /auth
{ "message": "<SIWS string>", "signature": "<Base58>", "type": "siwx" }
```

SIWS message format (CAIP-122):
```
x402.quicknode.com wants you to sign in with your Solana account:
<Base58 address>

I accept the Quicknode Terms of Service: https://www.quicknode.com/terms

URI: https://x402.quicknode.com
Version: 1
Chain ID: EtWTRABZaYq6iMfeYKouRu166VU2xqa1
Nonce: <random 8+ chars>
Issued At: <ISO 8601 timestamp>
```

### Required message fields (all paths)

- `domain`: `x402.quicknode.com`
- `address`: your wallet address (0x... for EVM, Base58 for Solana)
- `statement`: `I accept the Quicknode Terms of Service: https://www.quicknode.com/terms`
- `uri`: `https://x402.quicknode.com`
- `version`: `1`
- `chainId`: See Credit Pricing table for supported chain IDs
- `nonce`: at least 8 random characters (single-use)
- `issuedAt`: current ISO 8601 timestamp (must be within 5 minutes)

### Auth response

```json
{ "token": "<JWT>", "expiresAt": "<ISO datetime>", "accountId": "<CAIP-10 ID>" }
```

JWT expires in 1 hour. The auth chain determines payment method, not which networks you can query.

## Extension-Based Authentication (SIWX Header)

For fully self-describing x402-native flows — no out-of-band knowledge of `/auth` needed:

1. Hit any endpoint with no auth — server returns 402 with `extensions`
2. Read the `sign-in-with-x` extension — contains SIWX challenge with `domain`, `uri`, `nonce`, `issuedAt`, `supportedChains`
3. Sign the challenge — construct SIWX message, sign, encode as `SIGN-IN-WITH-X` header (Base64 JSON)
4. Pay with USDC — include `PAYMENT-SIGNATURE` header. Settlement response contains JWT in `quicknode-session` extension
5. Extract JWT from `extensions['quicknode-session'].info.token`

## Bootstrapping (No Existing Wallet)

### EVM (Base Sepolia)

1. Generate wallet with `generatePrivateKey()` + `privateKeyToAccount()` from `viem/accounts`
2. Authenticate via SIWE → get JWT
3. Call `POST /drip` → receive free testnet USDC on Base Sepolia
4. Wait for USDC to arrive (poll via `eth_call` on a public RPC)
5. Make RPC calls — `@quicknode/x402` with `preAuth: true` handles auth and payment negotiation; you still need to fund the wallet via `/drip` or transfer USDC manually

### Solana

1. Generate Ed25519 keypair via `@solana/kit` or `tweetnacl`
2. Pre-fund with Solana Devnet USDC (mint: `4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU`) — no `/drip` faucet for Solana
3. Authenticate via SIWX/Solana → get JWT
4. Make RPC calls with `@x402/fetch` + `@x402/svm`

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/auth` | 10 requests / 10 seconds per IP |
| `/credits` | 50 requests / 10 seconds per account |
| `/drip` | 5 requests / 60 seconds per account |
| `/:network` | 1,000 requests / 10 seconds per network:account pair |

## Best Practices

1. **Use `@quicknode/x402` for simplicity** — handles auth, payments, and session management automatically
2. **Use testnet first** — Call `/drip` on Base Sepolia for free credits during development
3. **Reuse JWT tokens** — they last 1 hour, no need to re-authenticate per request
4. **Monitor credits** — check `/credits` periodically to anticipate top-ups
5. **Auth chain ≠ query chain** — authenticate on Base but query Solana, Ethereum, or any supported network
6. **WebSocket for subscriptions** — use `/:network/ws` for persistent connections

## npm Packages

| Package | Purpose |
|---------|---------|
| `@quicknode/x402` | Official all-in-one client (recommended) |
| `@x402/fetch` | Low-level fetch wrapper for 402 payment handling |
| `@x402/evm` | EVM payment scheme (EIP-712 signing) |
| `@x402/svm` | Solana payment scheme (SPL Token transfer) |
| `viem` | Ethereum wallet client, signing, chain utilities |
| `siwe` | Sign-In with Ethereum (EIP-4361) messages |
| `@solana/kit` | Solana SDK for keypair signers |

## Documentation

- **x402 Platform**: https://x402.quicknode.com
- **x402 Documentation (llms.txt)**: https://x402.quicknode.com/llms.txt
- **x402 Guide**: https://www.quicknode.com/guides/x402/access-quicknode-endpoints-with-x402-payments
- **Code Examples**: https://github.com/quiknode-labs/qn-x402-examples
- **x402 Protocol Spec**: https://www.x402.org
