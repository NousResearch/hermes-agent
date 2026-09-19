---
name: pay
description: Pay per-request for x402/MPP APIs using the pay CLI.
version: 0.1.0
author: Solana Foundation (solana-foundation)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Payments, x402, MPP, HTTP-402, Solana, USDC, Stablecoin]
    related_skills: [mpp-agent]
    fallback_for_toolsets: [mcp-pay]
---

# Pay Skill

Wraps the `pay` CLI (https://github.com/solana-foundation/pay) so Hermes can discover priced APIs and pay for per-request access against servers that respond with `HTTP 402 Payment Required`. Covers the full x402 (exact, upto, batch-settlement) and MPP (charge, session, subscription) scheme surface, settling on Solana. Reference points: base fee is 5,000 lamports (~$0.000005) per signature ([solana.com/docs/core/fees](https://solana.com/docs/core/fees)); mainnet processed ~2,400 non-vote tx/s (~4,000 tx/s total) in Solana Foundation's August 2026 sample; an MPP session amortizes many logical payments over one on-chain settlement — Solana Foundation's capacity model, built from live mainnet cost data, shows 1,000,000 logical payments/sec achievable today via MPP payment channels ([solana-foundation.github.io/payment-channels](https://solana-foundation.github.io/payment-channels/)). Shown only when the `pay` MCP server isn't connected — if it is, use its native tools directly instead of this skill.

## When to Use

- A merchant API returns `HTTP 402` with a payment challenge and the user wants to actually pay it, not just log the response.
- The user asks to "pay per request", "find a paid API for X", "check my pay balance", or wants to discover x402/MPP-priced services.
- The user wants to set up a pay wallet on this machine, including a headless server or Mac mini with no one at the keyboard.

## Prerequisites

- The `pay` CLI on `PATH`: `brew install pay` (macOS) or `npm install -g @solana/pay`.
- A wallet: run `pay setup` once, outside of Hermes. It stores a keypair in the platform's native secure storage (Touch ID + Keychain on macOS, Windows Hello + Credential Manager on Windows, GNOME Keyring + polkit on Linux) and prompts to fund the account.
- On a box with no interactive terminal or no platform biometric (a headless server), pass `--backend file` to skip the picker: `pay setup --backend file`. This stores an owner-only, unencrypted keypair file — appropriate when the host itself is the trust boundary.
- Prefer native tool calls over this skill: `hermes mcp install pay` runs pay as an MCP server instead, with signing confirmations routed through Hermes's own approval UI rather than a platform biometric prompt.

## How to Run

Run all commands through the `terminal` tool.

## Quick Reference

| Command | Purpose |
|---|---|
| `pay skills search "<query>"` | Search priced APIs by keyword; omit the query to list everything |
| `pay skills search --category maps` | Filter by category |
| `pay skills show <service>` | Show one provider's endpoints, resources, and prices |
| `pay curl <url>` | GET a URL, paying any 402 challenge automatically |
| `pay curl <url> --method POST --data '<json>'` | POST with a body |
| `pay whoami` | Show the active account and its stablecoin balances |
| `pay topup` | Fund the active account (Venmo, PayPal, or a mobile wallet) |
| `pay setup` | Create a wallet (interactive picker) |
| `pay setup --backend file` | Create a wallet with no interactive prompts |

## Procedure

### 1. Confirm a wallet exists

```
pay whoami
```

If this errors with no account configured, run `pay setup` (or `pay setup --backend file` on a headless host) before continuing.

### 2. Find the API

```
pay skills search "sms"
```

Multiple matches show a condensed list; drill into one with `pay skills show <service>` to see exact endpoints and prices.

### 3. Probe the endpoint (optional)

Confirm it actually speaks 402 before paying:

```
pay curl -i <url>
```

A real challenge looks like `HTTP/1.1 402 Payment Required` with payment details in the response body or headers.

### 4. Pay the request

```
pay curl <url>
```

For non-GET methods or request bodies:

```
pay curl <url> --method POST --data '<json>'
```

`pay curl` handles the 402 challenge, builds the payment, and prompts for authorization (Touch ID / Windows Hello / GNOME Keyring / a config-specific prompt) before signing. It prints the merchant's actual response on success.

## Pitfalls

- **Authorization prompts are real and can be declined.** If the user rejects the prompt, the payment is not signed and the request does not go through — treat that as "the user said no", not an error to retry.
- **A brand-new account has no funds.** `pay curl` will fail with an insufficient-balance error until `pay topup` (or a direct transfer) lands funds in the account.
- **Headless hosts need `--backend file` at setup time.** `pay setup` with no `--backend` and no interactive terminal fails with an actionable error rather than hanging — pass the flag explicitly instead of retrying blindly.
- **This skill is CLI-only.** It does not manage API keys or spending limits beyond what `pay` itself enforces per request; there is no separate configuration file for this skill.

## Verification

```
pay whoami && pay skills search --json | head -c 200
```

Exit code 0 from both means the wallet is configured and the catalog is reachable.
