---
name: spraay-x402
description: On-chain batch payments and agent commerce via x402.
version: 1.0.0
author: LP (plagtech)
license: MIT
platforms: [macos, linux, windows]
required_environment_variables:
  - name: SPRAAY_GATEWAY_URL
    prompt: Spraay gateway URL
    help: Default is https://gateway.spraay.app — override only for self-hosted gateways
    required_for: full functionality
metadata:
  hermes:
    tags: [blockchain, payments, x402, batch, crypto, agent-commerce]
    requires_toolsets: [terminal]
---

# Spraay x402 Payments Skill

Send on-chain batch payments, escrow funds, bridge tokens, query gas prices, and access 150+ paid blockchain primitives through the Spraay x402 gateway — the payment rail built for AI agents.

Spraay is not a wallet or exchange. It is a gateway that returns unsigned transactions for agent signing, charges per-call via the HTTP 402 protocol, and supports 13+ chains (Base, Ethereum, Solana, Polygon, Arbitrum, Optimism, Stacks, Stellar, XRP, Bitcoin, Canton, and more).

## When to Use

- The user asks to **send crypto payments** to one or more recipients
- The user asks to **pay a batch** of addresses from a single transaction
- The user needs to **escrow funds** for a contract or milestone
- The user wants to **bridge tokens** between chains
- The user asks for **gas estimates**, **token prices**, or **on-chain data**
- The user wants to use the **x402 payment protocol** for agent-to-agent commerce
- The user mentions **Spraay**, **x402 gateway**, or **machine payments**

Do NOT use this skill for wallet creation, seed phrase management, or custodial key storage. Spraay is non-custodial — it returns unsigned transactions that the agent or user signs locally.

## Prerequisites

**Gateway URL (optional override):**

The public gateway is `https://gateway.spraay.app`. No API key is required — Spraay uses the x402 protocol where each call includes a micro-payment header. Set `SPRAAY_GATEWAY_URL` in `~/.hermes/.env` only if you run a self-hosted instance.

**MCP Server (alternative path):**

Spraay also ships an MCP server with 161 tools on Smithery. If MCP is preferred over HTTP:

```
hermes mcp add spraay --url https://smithery.ai/servers/Plagtech/Spraay-x402-mcp
```

The skill and MCP server expose the same capabilities — use whichever fits your setup.

**Wallet/Signer:**

The agent needs access to a signing mechanism (private key, hardware wallet, or agent wallet) to submit the unsigned transactions Spraay returns. Spraay itself never touches private keys.

## How to Run

All interactions go through the `terminal` tool using `curl` against the gateway REST API.

```bash
# Scan available primitives
curl -s https://gateway.spraay.app/scan | jq '.categories'

# Get a quote for a batch payment on Base
curl -s https://gateway.spraay.app/quote \
  -H "Content-Type: application/json" \
  -d '{
    "primitive": "batch_payment",
    "chain": "base",
    "recipients": ["0xRecipient1", "0xRecipient2"],
    "amounts": ["1000000", "2000000"],
    "token": "USDC"
  }' | jq .

# Execute (returns unsigned tx for agent signing)
curl -s https://gateway.spraay.app/execute \
  -H "Content-Type: application/json" \
  -H "X-402-Payment: <payment_header>" \
  -d '{
    "primitive": "batch_payment",
    "chain": "base",
    "recipients": ["0xRecipient1", "0xRecipient2"],
    "amounts": ["1000000", "2000000"],
    "token": "USDC",
    "sender": "0xYourAddress"
  }' | jq .
```

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/scan` | GET | List all 39 categories and 151+ paid primitives |
| `/quote` | POST | Get price quote and gas estimate for a primitive |
| `/execute` | POST | Execute primitive, returns unsigned tx |
| `/health` | GET | Gateway health check |

### Key Primitives by Category

| Category | Primitives | Price Range |
|----------|-----------|-------------|
| Batch Payments | `batch_payment`, `batch_payment_erc20` | $0.01-0.05 |
| Escrow | `create_escrow`, `release_escrow` | $0.05-0.25 |
| Token Swaps | `swap_exact_tokens` | $0.01-0.05 |
| Bridge | `bridge_tokens` | $0.05-0.25 |
| RPC/Data | `get_balance`, `get_gas_price`, `get_block` | $0.001-0.005 |
| AI Inference | `inference_request` (Bittensor SN64) | $0.03-0.05 |
| Oracle | `get_price_feed` | $0.005-0.01 |

### Supported Chains

Base (primary), Ethereum, Solana, Polygon, Arbitrum, Optimism, Avalanche, BNB Chain, Stacks, Stellar, XRP, Bitcoin, Canton.

## Procedure

1. **Scan** — Call `/scan` via `terminal` to list available primitives and confirm the desired operation exists.

2. **Quote** — Call `/quote` with the primitive name, chain, and parameters. The response includes the x402 payment amount (in USDC) and gas estimate.

3. **Review** — Present the quote to the user. Show the total cost (gateway fee + estimated gas). Ask for confirmation before proceeding.

4. **Execute** — Call `/execute` with the full parameters and the x402 payment header. The gateway returns an unsigned transaction object.

5. **Sign & Submit** — The unsigned transaction must be signed by the user's wallet or agent wallet. Spraay does not handle signing. If the agent has a configured signer, sign and broadcast. Otherwise, present the unsigned tx to the user for manual signing.

6. **Verify** — Check the transaction hash on the relevant block explorer. For Base: `https://basescan.org/tx/<hash>`.

## Pitfalls

- **Spraay is non-custodial.** It returns unsigned transactions. You need a signer. If no signer is configured, present the raw unsigned tx and guide the user to sign it manually.
- **x402 payment required.** Each paid primitive requires a micro-payment via the `X-402-Payment` header. The `/quote` endpoint tells you the exact amount. Free primitives (health, scan, some reads) do not require payment.
- **Batch contract address on Base is `0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC`.** Do not use any other address.
- **Token amounts are in smallest units.** USDC on Base uses 6 decimals, so `1000000` = 1 USDC. ETH uses 18 decimals.
- **Solana gateway** (`gateway-solana.spraay.app`) returns base64-encoded unsigned transactions, not EVM-style tx objects.
- **Rate limits.** The public gateway is not rate-limited for normal usage, but sustained high-volume callers should reach out for dedicated access.
- **Do not fabricate addresses.** Never generate or guess contract addresses. Use only the documented addresses from this skill or from the `/scan` response.

## Verification

```bash
# Confirm gateway is live
curl -s https://gateway.spraay.app/health | jq .

# Confirm primitives are available
curl -s https://gateway.spraay.app/scan | jq '.total_primitives'
# Expected: 151 or higher
```

## Resources

- Gateway: `https://gateway.spraay.app`
- Solana Gateway: `https://gateway-solana.spraay.app`
- Docs: `https://docs.spraay.app`
- MCP Server: `https://smithery.ai/servers/Plagtech/Spraay-x402-mcp`
- GitHub: `https://github.com/plagtech`
- Batch Contract (Base): `0x1646452F98E36A3c9Cfc3eDD8868221E207B5eEC`
- Pay Address: `0xAd62f03C7514bb8c51f1eA70C2b75C37404695c8`
