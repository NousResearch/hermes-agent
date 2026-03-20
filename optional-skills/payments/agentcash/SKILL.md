---
name: agentcash
description: Pay-per-call access to premium APIs via x402 and MPP micropayments (USDC on Base, Solana, or Tempo). No API keys or subscriptions required — call enrichment, scraping, email, phone, image/video generation, and dozens more services with automatic wallet authentication and payment.
version: 0.1.0
author: fmhall
license: MIT
metadata:
  hermes:
    tags: [x402, MPP, Payments, APIs, Micropayments, USDC, Base, Solana, Tempo, Crypto, Web3]
    related_skills: []
---

# AgentCash — Paid API Access

Call any x402-protected API with automatic wallet authentication and payment. No API keys or subscriptions required.

---

## When to Use

- User needs to call a premium API (enrichment, scraping, email, phone, etc.) without managing API keys
- User wants to enrich contacts or companies (Apollo, LinkedIn, Whitepages)
- User wants to scrape web pages or search the web (Firecrawl, Exa)
- User wants to generate images or videos (GPT Image, Flux, Sora, Veo)
- User wants to send emails, make AI phone calls, or upload files
- User wants to search social media data (TikTok, Instagram, Reddit, LinkedIn)
- User asks about x402, agentcash, or pay-per-call APIs

---

## Prerequisites

- **Node.js** (v18+) — the CLI runs via `npx`
- **USDC balance** on Base or Solana — fund via `npx agentcash@latest fund` or redeem an invite code

No API keys, accounts, or subscriptions are needed for any service.

---

## Quick Reference

### Wallet Management

| Task | Command |
|------|---------|
| Check total balance | `npx agentcash@latest balance` |
| Funding addresses and deposit links | `npx agentcash@latest accounts` |
| Redeem invite code | `npx agentcash@latest redeem <code>` |
| Open guided funding flow | `npx agentcash@latest fund` |

Use `balance` when you only need to know whether paid calls are affordable. Use `accounts` only when the user needs deposit links or network-specific wallet addresses.

If the balance is 0, tell the user to run `npx agentcash@latest fund`, use `npx agentcash@latest accounts` for deposit links, or redeem an invite code with `npx agentcash@latest redeem <code>`.

### Using Services

#### 1. Discover endpoints on a service

```bash
npx agentcash@latest discover <origin>
```

Example: `npx agentcash@latest discover https://stableenrich.dev`

Read the output carefully. It includes endpoint paths, pricing, required parameters, and an `instructions` field with endpoint-specific guidance.

#### 2. Check a specific endpoint before calling it

```bash
npx agentcash@latest check <endpoint-url>
```

Returns the request and response schema plus pricing guidance. Use this before `fetch` to avoid 400 errors from wrong field names.

#### 3. Make the request

```bash
# POST
npx agentcash@latest fetch <url> -m POST -b '{"key": "value"}'

# GET
npx agentcash@latest fetch '<url>?param=value'
```

`fetch` handles both paid routes and SIWX routes. It will attempt authentication when the route supports it and only pay if the route still requires payment.

---

## Available Services

Any endpoint that is payment-protected can be accessed with AgentCash. Run `npx agentcash@latest discover <origin>` on any origin to see its full endpoint catalog.

| Origin | What it does | Price range |
|--------|-------------|-------------|
| `stableenrich.dev` | Apollo (people/org search), Exa (web search), Firecrawl (scraping), Google Maps, Clado (LinkedIn), Serper (news/shopping), WhitePages, Hunter (email verification) | $0.02–$0.44/call |
| `stablesocial.dev` | Social media data: TikTok, Instagram, Facebook, Reddit, LinkedIn (async two-step) | $0.06/call |
| `stablestudio.dev` | AI image/video generation: GPT Image, Flux, Grok, Nano Banana, Sora, Veo, Seedance, Wan | $0.04–$3.00/call |
| `stableupload.dev` | Pay-per-upload file hosting (6-month TTL) | $0.02–$2.00/upload |
| `stableemail.dev` | Send emails, forwarding inboxes, custom subdomains | $0.02–$5.00 |
| `stablephone.dev` | AI phone calls, phone numbers, top-ups | $0.54–$20.00 |
| `stablejobs.dev` | Job search via Coresignal | varies |
| `stabletravel.dev` | Travel search | varies |

There are many more services available beyond the ones listed here.

---

## Important Rules

- **Always discover before guessing.** Endpoint paths include provider prefixes (e.g., `/api/apollo/people-search`, not `/people-search`).
- **Read the instructions field.** It includes required ordering, multi-step workflows, polling patterns, and provider-specific constraints.
- **Payments settle on success only.** Failed requests (non-2xx) do not cost anything.
- **Check balance before expensive operations.** Video generation can cost $1–3 per call.

---

## Tips

- Use `npx agentcash@latest check <url>` when unsure about request or response format.
- Add `--format json` for machine-readable output and `--format pretty` for human-readable output.
- Base and Solana are both supported payment networks. Use the one called out by the endpoint or the one where the user has funds.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Insufficient balance" | Run `balance`, then `fund` or `accounts`, or redeem an invite code |
| "Payment failed" | Retry the request |
| "Invalid invite code" | The code is used or does not exist |
| Balance not updating | Wait for network confirmation and rerun `balance` |
