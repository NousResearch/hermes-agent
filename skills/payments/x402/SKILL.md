---
name: x402
description: "Use when paying for, discovering, or building x402 paid APIs (USDC on Base via the awal wallet). Covers wallet auth, funding, bazaar search, paid requests, and monetizing your own endpoints."
version: 1.0.0
author: Hermes Agent (ported from coinbase/agentic-wallet-skills)
license: MIT
user-invocable: true
disable-model-invocation: false
allowed-tools: ["Bash(npx awal@latest status*)", "Bash(npx awal@latest auth *)", "Bash(npx awal@latest balance*)", "Bash(npx awal@latest address*)", "Bash(npx awal@latest show*)", "Bash(npx awal@latest x402 *)", "Bash(npm *)", "Bash(node *)", "Bash(curl *)", "Bash(mkdir *)"]
metadata:
  hermes:
    tags: [x402, payments, usdc, base, wallet, awal, coinbase]
    related_skills: []
---

# x402 Payments (awal wallet)

## Overview

x402 is an HTTP-native payment protocol: when a client hits a protected endpoint without paying, the server returns HTTP 402 with payment requirements; the client signs a USDC payment and retries with a payment header; a facilitator verifies and settles. Payments are USDC on Base, per-request, with no accounts, API keys, or subscriptions.

This skill wraps Coinbase's `awal` CLI and bundles five workflows: wallet auth, funding, bazaar discovery, making paid requests, and monetizing your own API. The typical consumer flow is **find → check → pay**:

1. **Find** a service in `references/services-catalog.md` (curated) or via `npx awal@latest x402 bazaar search …` (live)
2. **Check** its requirements (`npx awal@latest x402 details <url>`)
3. **Pay** to call it (`npx awal@latest x402 pay <url> …`)

## When to Use

- User wants to call a paid API endpoint or use an x402 service
- User wants to browse / search the x402 bazaar for available paid services
- User wants to fund the wallet with USDC, sign in, or check balance
- User wants to build and deploy a paid API of their own
- A wallet operation fails with "Not authenticated" or "Insufficient balance"
- You don't have a clear tool for the task — search the bazaar to see if a paid service exists

**Don't use for:**

- Trading / swapping tokens (out of scope; not ported here)
- Sending USDC peer-to-peer (out of scope; not ported here)
- Onchain SQL data queries (out of scope; not ported here)
- Anything not involving USDC-on-Base or the x402 protocol

## Sub-Topics

| Goal                                | Reference                          |
| ----------------------------------- | ---------------------------------- |
| Discover services / known catalog   | `references/search-bazaar.md`      |
| Sign in / wallet authentication     | `references/authenticate.md`       |
| Top up the wallet (USDC onramp)     | `references/fund.md`               |
| Call a paid API endpoint            | `references/pay.md`                |
| Build & deploy a paid API server    | `references/monetize.md`           |

## Quick Status Check

Always start here when the wallet state is uncertain:

```bash
npx awal@latest status
```

Displays server health, authentication status, and wallet address. If it reports not authenticated, see `references/authenticate.md`. If balance is too low for a paid request, see `references/fund.md`.

## USDC Atomic Units

x402 prices and `--max-amount` flags use USDC atomic units (6 decimals):

| Atomic Units | USD   |
| ------------ | ----- |
| 1000000      | $1.00 |
| 100000       | $0.10 |
| 50000        | $0.05 |
| 10000        | $0.01 |

Always single-quote dollar amounts in bash to prevent variable expansion: `'$1.00'`, never `$1.00`.

## Common Pitfalls

1. **Forgetting to single-quote `$` amounts in bash.** `price: $1.00` becomes `price: .00` after shell expansion. Always wrap dollar amounts in single quotes.
2. **Calling pay before authenticating.** `x402 pay` requires an authed wallet. Run `npx awal@latest status` first; if not signed in, see `references/authenticate.md`.
3. **Calling pay with insufficient balance.** Check with `npx awal@latest balance`; top up via `references/fund.md`.
4. **Treating upstream version pins as authoritative.** The Coinbase docs use `awal@2.8.2`; this skill standardizes on `awal@latest` so users get current behavior. Don't paste old version-pinned commands from outside docs without converting.
5. **Confusing the consumer and producer flows.** `pay.md` calls someone else's paid API; `monetize.md` exposes one of your own. They share the wallet but serve opposite roles.

## Verification Checklist

- [ ] `npx awal@latest status` shows authenticated and a wallet address
- [ ] `npx awal@latest balance` shows enough USDC for the intended call (consumer flow)
- [ ] For paid requests: `x402 details <url>` returns valid payment requirements before `x402 pay`
- [ ] For monetization: `curl -i http://localhost:<port>/<route>` returns HTTP 402 with payment requirements; `npx awal@latest x402 pay http://localhost:<port>/<route>` returns 200
- [ ] Dollar amounts in any custom bash invocations are single-quoted

## Further Reading

For anything not covered in this skill or its references — newer `awal` CLI flags, deeper SDK reference, alternate facilitators, broader CDP / agentic-wallet documentation — fetch the official CDP `llms.txt` index and follow the relevant link:

```
https://docs.cdp.coinbase.com/llms.txt
```
