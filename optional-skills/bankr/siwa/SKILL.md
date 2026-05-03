---
name: siwa
version: 0.2.0
description: >
  SIWA (Sign-In With Agent) authentication for ERC-8004 registered agents.
metadata:
  {
    "clawdbot":
      {
        "emoji": "🔐",
        "homepage": "https://siwa.id",
      },
  }
---

# SIWA SDK

Sign-In With Agent (SIWA) lets AI agents authenticate with services using their ERC-8004 onchain identity.

## Install

```bash
npm install @buildersgarden/siwa
```

## Skills

### Agent-Side (Signing)

Choose based on your wallet provider:

- [Bankr](references/bankr-signer.md) — Bankr Agent API wallets

### Server-Side (Verification)

- [Server-Side Verification](references/server-side.md) — Next.js, Express, Hono, Fastify

## SDK Modules

| Import | Description |
|--------|-------------|
| `@buildersgarden/siwa` | Core: signSIWAMessage, verifySIWA, createSIWANonce |
| `@buildersgarden/siwa/signer` | Signer factories |
| `@buildersgarden/siwa/erc8128` | ERC-8128 HTTP signing/verification |
| `@buildersgarden/siwa/receipt` | HMAC receipt helpers |
| `@buildersgarden/siwa/nonce-store` | Nonce stores (Memory, Redis, KV) |
| `@buildersgarden/siwa/next` | Next.js middleware |
| `@buildersgarden/siwa/express` | Express middleware |
| `@buildersgarden/siwa/hono` | Hono middleware |
| `@buildersgarden/siwa/fastify` | Fastify middleware |

## Links

- [Latest version of this skill](https://siwa.id/skill.md)
- [Documentation](https://siwa.id/docs)
- [ERC-8004](https://eips.ethereum.org/EIPS/eip-8004)
- [ERC-8128](https://eips.ethereum.org/EIPS/eip-8128)
