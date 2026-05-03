# SIWA Bankr Signer

> Latest version: [https://siwa.id/skills/bankr/skill.md](https://siwa.id/skills/bankr/skill.md)

Sign SIWA messages using Bankr's Agent API wallets.

## Install

```bash
npm install @buildersgarden/siwa
```

No additional SDK is required — the Bankr signer communicates directly with the Bankr Agent API over HTTP.

## Create Signer

```typescript
import { createBankrSiwaSigner } from "@buildersgarden/siwa/signer";

const signer = await createBankrSiwaSigner({
  apiKey: process.env.BANKR_API_KEY!,
});
```

The wallet address is fetched automatically from Bankr's `/agent/me` endpoint.

## Register as ERC-8004 Agent

Bankr wallets are smart contract accounts (ERC-4337). To register on the ERC-8004 Identity Registry, submit the registration as an [arbitrary transaction](https://github.com/BankrBot/skills/blob/main/bankr/references/arbitrary-transaction.md) via Bankr's `/agent/submit` endpoint.

### Supported Networks

The Bankr `/agent/submit` endpoint supports these chains:

| Network    | Chain ID |
|------------|----------|
| Ethereum   | 1        |
| Polygon    | 137      |
| Base       | 8453     |
| Unichain   | 130      |

### Transaction Format

The `/agent/submit` endpoint requires a `transaction` object with these fields:

| Field     | Type   | Description                                             |
|-----------|--------|---------------------------------------------------------|
| `to`      | string | Contract address (`0x` + 40 hex characters)             |
| `data`    | string | ABI-encoded calldata (`0x`-prefixed, or `"0x"` if empty)|
| `value`   | string | Transaction value in wei (use `"0"` for registration)   |
| `chainId` | number | Target network ID (must be a supported network above)   |

### Example

```typescript
import { encodeRegisterAgent } from "@buildersgarden/siwa/registry";

// Prepare agent metadata
const metadata = {
  name: "My Agent",
  description: "A helpful AI assistant",
  capabilities: ["chat", "analysis"],
};
const agentURI = `data:application/json;base64,${Buffer.from(JSON.stringify(metadata)).toString("base64")}`;

// Encode the registration calldata (resolves the registry address automatically)
const { to, data } = encodeRegisterAgent({ agentURI, chainId: 8453 });

// Submit as arbitrary transaction via Bankr API
const submitRes = await fetch("https://api.bankr.bot/agent/submit", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": process.env.BANKR_API_KEY!,
  },
  body: JSON.stringify({
    transaction: { to, data, value: "0", chainId: 8453 },
    waitForConfirmation: true,
  }),
});

const result = await submitRes.json();
console.log("Transaction hash:", result.txHash);
```

> **Important:** Transactions submitted via `/agent/submit` are **irreversible**. Always verify calldata encoding and confirm the target contract address before submitting. Ensure the wallet has sufficient ETH/MATIC for gas. The endpoint handles UserOperation bundling internally for ERC-4337 smart accounts, and SIWA verification uses ERC-1271 automatically.

---

## SIWA Authentication Flow

The authentication flow consists of two steps:

> **Note:** The URLs below (`api.example.com`) are placeholders. Replace them with your own server that implements the SIWA verification endpoints. See the [Server-Side Verification](server-side.md) skill for implementation details.

1. **Get a nonce** from the server's `/siwa/mainnet/nonce` endpoint
2. **Sign and verify** by sending the signature to `/siwa/mainnet/verify`

### Step 1: Request Nonce

```typescript
const nonceRes = await fetch("https://api.example.com/siwa/mainnet/nonce", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    address: await signer.getAddress(),
    agentId: 42,
    agentRegistry: "eip155:8453:0x8004A169FB4a3325136EB29fA0ceB6D2e539a432",
  }),
});
const { nonce, nonceToken, issuedAt, expirationTime } = await nonceRes.json();
```

### Step 2: Sign and Verify

```typescript
import { signSIWAMessage } from "@buildersgarden/siwa";

const { message, signature, address } = await signSIWAMessage({
  domain: "api.example.com",
  uri: "https://api.example.com/siwa",
  agentId: 42,
  agentRegistry: "eip155:8453:0x8004A169FB4a3325136EB29fA0ceB6D2e539a432", //According to the chain
  chainId: 8453,
  nonce,
  issuedAt,
  expirationTime,
}, signer);

// Send to server for verification
const verifyRes = await fetch("https://api.example.com/siwa/mainnet/verify", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message, signature, nonceToken }),
});

const { receipt, agentId } = await verifyRes.json();
// Store the receipt for authenticated API calls
```

## Sign Authenticated Request (ERC-8128)

```typescript
import { signAuthenticatedRequest } from "@buildersgarden/siwa/erc8128";

const request = new Request("https://api.example.com/action", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ action: "execute" }),
});

const signedRequest = await signAuthenticatedRequest(
  request,
  receipt,  // from SIWA sign-in
  signer,
  8453,
);

const response = await fetch(signedRequest);
```

## Environment Variables

```bash
BANKR_API_KEY=your-bankr-api-key
```

## Already using Bankr?

If your agent already uses Bankr's [OpenClaw trading skill](https://github.com/BankrBot/openclaw-skills/tree/main/bankr) for swaps, bridges, or DeFi, this signer lets you add SIWA authentication on top — same API key, same wallet, no extra setup. Bankr's trading skill handles what your agent *does*; the SIWA signer handles how your agent *proves who it is*.
