---
name: agentcard
description: Issues AgentCard single-use virtual cards, handles checkout 3DS challenges, manages the user's AgentCard agent email inbox, and uses the AgentCard USDC wallet for Base/x402 payments. Use when an agent needs to pay a merchant, create a capped virtual card, complete a checkout, fetch paid x402 content, send or read agent email, inspect spend limits, or troubleshoot AgentCard CLI payments.
version: 1.0.0
author: "Alchemy (https://alchemy.com)"
license: MIT
homepage: https://agentcard.ai
prerequisites:
  commands:
    - npm
metadata:
  hermes:
    tags:
      - payments
      - commerce
      - identity
      - virtual-cards
      - agent-commerce
      - email
      - wallet
      - x402
      - 3ds
    category: payments
    homepage: https://agentcard.ai
    requires_toolsets:
      - terminal
---

# AgentCard

AgentCard gives agents payment and identity tools with user-controlled limits:

- Single-use virtual cards capped to a requested amount and live for 30 minutes.
- A dedicated `@agentcard.email` inbox for signups, verification codes, and user-approved communication.
- A self-custodial USDC wallet on Base for x402 HTTP payments and USDC transfers.

Prefer exact, command-driven workflows. Do not use the removed `agentcard create` flow.

## Install In Hermes

This is an official optional skill:

```bash
hermes skills install official/payments/agentcard
```

After installation, Hermes will load this skill when payment, checkout, x402, AgentCard, or agent email tasks match its description.

## Install And Check The CLI

```bash
command -v agentcard >/dev/null || npm install -g agentcard
agentcard --version
agentcard --help
```

Use normal shell execution and parse stdout/stderr. Some commands intentionally block while polling for user action.

## Operating Rules

- Spend only for the amount and purpose the user requested. If the merchant, recipient, amount, or payment method is ambiguous, ask before paying.
- Never collect or type the user's real payment card details. During setup, show the printed Stripe setup URL to the user and let them complete it.
- Treat PAN, CVV, expiry, billing ZIP, wallet addresses, emails, 3DS codes, and verification codes as sensitive. Do not save card details or codes in durable notes unless the user explicitly asks.
- For cards, use the printed cardholder name and billing ZIP exactly. Mismatches can cause declines.
- Use `agentcard support` immediately for failed purchases, CAPTCHAs, declines, confusing errors, or merchant checkout blocks.

## Quick Reference

| Task | Command |
|------|---------|
| Sign up or log in | `agentcard signup --email <email>` |
| Set up profile, payment, email, wallet | `agentcard setup --first-name <first> --last-name <last> --phone <+E164>` or interactive `agentcard setup` |
| Reset setup | `agentcard setup --reset` |
| Show spend limit | `agentcard limit` |
| Update spend limit | `agentcard limit --amount <dollars>` |
| Issue card | `agentcard request new --amount <integer-dollars>` |
| List card requests | `agentcard request get` |
| Re-fetch live card | `agentcard request get <id>` |
| Get checkout 3DS code | `agentcard 3ds` |
| Show agent email | `agentcard mail info` |
| List email threads | `agentcard mail list [--include-sent]` |
| Read email thread | `agentcard mail get <thread-id>` |
| Send email | `agentcard mail send --to <email> --subject <subject> --body <body>` |
| Reply to thread | `agentcard mail reply <thread-id> --body <body>` |
| Show wallet | `agentcard wallet info` |
| Print USDC balance | `agentcard wallet balance` |
| Send USDC on Base | `agentcard wallet send --to <0x...> --amount <usdc>` |
| x402 fetch | `agentcard wallet fetch <url> [-H "Key: Value"] [-X METHOD] [-d BODY] [--max-cost <usdc>]` |
| Current account | `agentcard whoami` |
| Report issue | `agentcard support --message "details" [--card-id <id>] [--url <url>] [--error <error>]` |

## First-Time Setup

1. Authenticate:

   ```bash
   agentcard signup --email <user-email>
   ```

   The command sends a magic link and polls for up to five minutes. Tell the user to check their inbox and click the link. Let the process finish; do not kill and retry while it is polling.

2. Run setup every time after authentication and before payment work:

   ```bash
   agentcard setup --first-name <first> --last-name <last> --phone <+E164>
   ```

   Setup is idempotent. If the account is already configured, it prints the current profile and exits cleanly. The phone must include a country code, for example `+14155551234`. Use flags in headless or non-interactive terminals; plain `agentcard setup` may prompt only when the terminal is interactive. Setup verifies profile details, ensures a saved payment method exists for card holds, provisions the agent email inbox for signups and verification codes, and provisions the Base wallet for USDC/x402 payments. If setup prints a Stripe setup URL, relay the URL to the user and wait for completion, especially in headless or remote environments where the local browser may not be visible to them.

3. Verify:

   ```bash
   agentcard whoami
   agentcard limit
   agentcard mail info
   agentcard wallet info
   ```

Run setup again whenever a task may need cards, agent email, or wallet access. Use `agentcard setup --reset` only when the user explicitly wants to replace profile, payment, email, or wallet setup.

## Choose Payment Path First

Before issuing a card or initiating a wallet payment, determine the payment path from the merchant, recipient, or API surface:

1. Get the final total first, including shipping, taxes, and fees. Do not create a card before the final amount is known.
2. Understand what the user is buying well enough that the payment purpose is recognizable.
3. Choose the matching path:

| What you find | Use |
|---------------|-----|
| Merchant accepts a card at checkout | Capped card workflow |
| HTTP endpoint returns `402 Payment Required` with x402 requirements | Wallet and x402 workflow |
| User gives a Base USDC recipient address and amount | Direct USDC send, after confirming recipient and amount |
| Payment method, amount, merchant, or recipient is unclear | Ask the user before creating credentials or paying |

## Capped Card Workflow

1. Use the checkout total rounded up to the next whole dollar as the card cap, from `1` to `75`. For example, `$24.99` becomes `--amount 25`. Do not add extra padding beyond the rounded-up total.
2. If the rounded-up total is over `75`, do not issue a card. Ask the user for a lower-cost plan, a different payment path, or explicit approval for a merchant-supported split payment. Do not split a checkout across multiple cards unless the merchant supports separate charges and the user approves.
3. For subscriptions or trials, do not proceed until the user explicitly accepts that this card is live for 30 minutes and future recurring charges may fail unless the merchant only needs an initial authorization.
4. Check available spend:

   ```bash
   agentcard limit
   ```

   If the card cap exceeds the remaining spend limit, ask the user before raising it with `agentcard limit --amount <dollars>`.

5. Issue the card:

   ```bash
   agentcard request new --amount <dollars>
   ```

6. Use the printed request ID, PAN, CVV, expiry month/year, cardholder name, and billing ZIP. The card expires 30 minutes after issue.
7. If checkout asks for a 3DS or card verification code, then run:

   ```bash
   agentcard 3ds
   ```

   3DS is conditional, not a required step. Codes usually appear within a few seconds and are listed for five minutes. Choose the code whose amount matches the checkout charge amount, not the rounded card cap. If multiple codes match, use the newest. Enter it directly into checkout. If no amount matches, wait a few seconds and run `agentcard 3ds` again.

8. If details are needed again while the card is still live:

   ```bash
   agentcard request get <id>
   ```

AgentCard places a hold on the user's stored payment method for the requested amount. The user is charged only if the merchant uses the card; unused holds release within seven days.

## Wallet And x402 Workflow

Use the wallet for USDC on Base and HTTP endpoints that return `402 Payment Required` with x402 requirements.

```bash
agentcard wallet info
agentcard wallet balance
agentcard wallet fetch <url> --max-cost <usdc>
```

`wallet fetch` wraps native fetch, pays x402 requirements in USDC when needed, retries the request, and prints the final response body to stdout. Always pass `--max-cost` unless the user explicitly approved any valid charge from that endpoint. The value is decimal USDC, not atomic token units; for dollar-pegged USDC amounts, 15 USD cents is `--max-cost 0.15`.

Pass HTTP details through the CLI when the endpoint needs them:

```bash
agentcard wallet fetch <url> -H "Authorization: Bearer <token>" -X POST -d '<json-body>' --max-cost <usdc>
```

Use one `-H` flag per header. The final response body is printed to stdout; stderr contains HTTP/payment errors.

For direct USDC transfers:

```bash
agentcard wallet send --to <0x-recipient> --amount <usdc>
```

Before sending USDC, confirm the recipient address and amount are exactly what the user intended. Gas is sponsored, but the USDC spend counts toward the same AgentCard spend limit. Fund the wallet by sending USDC on Base to the address from `agentcard wallet info`.

## Agent Email Workflow

Use the agent email for account signup, verification codes, receipts, and user-approved communication. Read verification codes only for the active user-approved workflow, enter them directly into the target service, and do not paste them into chat or store them unless the user explicitly requests it.

```bash
agentcard mail info
agentcard mail list --include-sent
agentcard mail get <thread-id>
agentcard mail send --to <email> --subject <subject> --body <body>
agentcard mail reply <thread-id> --body <body>
```

Prefer `agentcard mail get <thread-id>` before replying so the response matches the current thread context.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `agentcard: command not found` | Run `npm install -g agentcard`, then `agentcard --version`. |
| Not logged in or session expired | Run `agentcard signup --email <email>` again. |
| Missing profile, payment method, email, or wallet | Run `agentcard setup` with flags or interactively. |
| Amount rejected | Cards require positive integer dollars and are capped at `75` during beta. Round checkout totals up to the next whole dollar; stop and ask the user if the rounded-up total is over `75`. |
| Spend limit exceeded | Run `agentcard limit`; ask the user before `agentcard limit --amount <dollars>`. |
| 3DS code missing | Run `agentcard 3ds` only after checkout requests a code. Wait a few seconds and retry; if no matching recent code appears, file support. |
| Card declined | Check the live 30-minute window, exact billing fields, 3DS challenge status, and spend limit. File support, then ask the user before issuing a replacement card. |
| Wallet not provisioned | Run `agentcard setup`. |
| Wallet balance too low | Show `agentcard wallet info` and ask the user to fund the Base USDC address. |
| x402 fetch fails | Retry with `--max-cost`, inspect stderr, and report persistent failures. |

## Report Issues

For any purchase failure, decline, CAPTCHA, anti-bot block, unexpected CLI error, confusing output, or merchant checkout problem, run:

```bash
agentcard support --message "what happened, what you tried, and what the user expected" --url "<checkout-or-api-url>" --error "<error-text>"
```

Add `--card-id <id>` when the issue involves a specific card request.

Redact secrets from support messages and errors: never include PAN, CVV, bearer tokens, 3DS codes, verification codes, or private keys.
