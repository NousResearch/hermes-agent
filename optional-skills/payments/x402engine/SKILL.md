---
name: x402engine
description: Discover and buy pay-per-call APIs with USDC.
version: 0.2.0
author: __agentc1__ (@agentc22), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Payments, x402, HTTP-402, USDC, Base, Solana, APIs]
    related_skills: [mpp-agent]
---

# x402engine Skill

Use the `x402engine` MCP when a user wants an agent to buy an API result per request with USDC. Its curated tools cover crypto and wallet data, code execution, image generation, transcription, travel, and IPFS. Free live discovery reports the broader gateway catalog, prices, and payment networks.

## When to Use

- The user asks for an x402 API, pay-per-call API, or software-paid API result.
- A task needs a one-off external API and the user prefers per-request USDC payment.
- The user wants to inspect x402engine services, pricing, or service health.

This skill is for Coinbase x402-style `PAYMENT-REQUIRED` challenges. For Machine Payments Protocol challenges using `WWW-Authenticate`, use the `mpp-agent` skill.

## Prerequisites

Requires Node.js 20.18 or later. Once this catalog entry is merged and present in the installed Hermes version:

```bash
hermes mcp install x402engine
```

Before catalog availability, use the manual MCP configuration in the [x402engine Hermes guide](https://x402engine.app/docs/hermes/). The installable community skill is also served at `https://x402engine.app/.well-known/skills/x402engine`.

Free discovery uses `X402_PAYMENT_NETWORK: "auto"` and works without credentials. Paid calls require one dedicated payer wallet:

- `X402_EVM_PRIVATE_KEY`: Base wallet funded with USDC
- `X402_SOLANA_PRIVATE_KEY`: Solana wallet funded with USDC and a small SOL balance

Keys belong in the active Hermes profile’s local `.env` file. Add a reference for the configured key under `mcp_servers.x402engine.env`, such as `X402_EVM_PRIVATE_KEY: "${X402_EVM_PRIVATE_KEY}"`, and set the network to `base` or `solana`. Never print, read back, or paste a wallet key into the conversation. Use a low-balance wallet created for agent purchases, never a primary wallet.

## How to Run

`X402_MAX_PAYMENT_USD` is a hard per-request cap and defaults to `0.10` in this catalog entry. The MCP rejects unknown assets, unsupported networks, and prices above this cap before signing.

Installing the server or configuring a wallet does not authorize spending. Use the user’s existing task and budget authorization; obtain it before paid use if absent. Track total task spend separately from the per-request cap.

Before a paid call:

1. Confirm the requested outcome.
2. Read the tool description or live discovery price.
3. Use the least expensive tool that satisfies the request.
4. Do not split or repeat calls to evade the configured cap.
5. Surface the expected price before unusually expensive or repeated operations.

Base is preferred when both wallets exist. Set `X402_PAYMENT_NETWORK` to `base` or `solana` in the MCP server configuration to force one rail.

## Quick Reference

| Task | MCP tool |
| --- | --- |
| Browse current catalog | `discover_services` |
| Check availability | `service_health` |
| Create an image | `generate_image` |
| Run a calculation | `execute_code` |
| Get crypto prices | `get_crypto_price` |

## Procedure

### 1. Discover the live catalog

Call `discover_services`. It is free and returns current routes, prices, and supported payment networks.

### 2. Check service health

Call `service_health`, optionally with a service ID. It is free. Avoid unhealthy services when an equivalent healthy option exists.

### 3. Enable the required paid tool

The catalog install enables only `discover_services` and `service_health` by default. Enable paid tools after reviewing their descriptions:

```bash
hermes mcp configure x402engine
```

Start a new Hermes session after changing the tool selection.

### 4. Make the call once

Call the selected MCP tool with the smallest sufficient request. The MCP requests the resource, validates the x402 challenge against its network, asset, and spending policy, then signs locally and sends the paid request. Check the result for API errors as well as successful data.

Do not retry a paid tool automatically after an ambiguous timeout. Check the returned error and service health first because settlement may already have occurred.

## Pitfalls

The complete gateway catalog is larger than this package’s tool set. Do not invent chat, video or speech tools merely because discovery lists those routes. Use the actual tools exposed in the session and their input schemas. For travel use future dates. If a direct API client returns a private job status URL, poll it without submitting a second paid job.


- `Payment required`: no payer key is configured. Configure one wallet key and its server env reference as described in Prerequisites, then start a new session.
- `exceeds X402_MAX_PAYMENT_USD`: the route costs more than the cap. Report the price and ask before raising the limit.
- `No supported ... USDC payment option`: the challenge offered a network or asset this MCP intentionally does not sign.
- Insufficient funds: fund the dedicated wallet with the correct chain's USDC; Solana also needs SOL for fees.
- Tool unavailable: run `hermes mcp configure x402engine`, enable it, and restart the session.

## Verification

```bash
hermes mcp test x402engine
```

Then call `discover_services` and `service_health`. A paid verification should use `get_crypto_price`, currently the lowest-cost curated tool, only when the user has authorized a real payment.
