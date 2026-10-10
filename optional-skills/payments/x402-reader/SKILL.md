---
name: x402-reader
description: Pay x402 to read a public URL as markdown.
version: 0.1.1
author: twzrd (twzrd-sol), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Payments, HTTP-402, x402, Reader, Solana]
    related_skills: [mpp-agent]
---

# X402 Reader Skill

Turns a public URL into markdown by paying an x402 reader
(`https://reader.outbid.sh` by default). Quote is free. Live spend is
opt-in and capped before sign. This is the Solana/Base USDC x402 path;
MPP 402s belong to `mpp-agent`.

Gated `[linux, macos]` while the npm client matures on Windows.

## When to Use

- Need page text and the origin is fat HTML, JS-walled, or already 402s
- A fetch returns `HTTP 402` with x402 `accepts` (Solana or Base USDC)
- User says "scrape", "read this URL", "pay the 402", or "x402 reader"

Don't use for: MPP/`www-authenticate: tempo` (use `mpp-agent`), Stripe
checkout (use `stripe-link-cli`), or adding 402 to your own API.

## Prerequisites

- Node.js 20+ on `PATH` (`node --version` via `terminal`)
- For live pay only: a funded Solana or Base USDC wallet file, plus
  `X402_READER_PAYMENTS_ENABLED=1` in the MCP **server env**
- No wallet is required to quote

## How to Run

Install the official optional skill, then add the MCP:

```
hermes skills install official/payments/x402-reader
hermes mcp install x402-reader
```

`hermes mcp install` writes a stdio server (`npx -y x402-reader-mcp@0.1.2`)
that exposes `scrape` ($0.005) and `browse` ($0.05). Unarmed, both tools
return a free 422 miss or a visible 402 quote. Shell exports do **not**
reach the MCP — put payment env in `mcp_servers.x402-reader.env`.

## Quick Reference

| Tool | Route | Price |
|---|---|---:|
| `scrape` | `GET /scrape?url=` | $0.005 USDC |
| `browse` | `GET /browse?url=` | $0.05 USDC |

`scrape` never auto-opens `browse`. A `needs_browser` 422 is terminal.

## Procedure

### 1. Quote (no wallet)

Write headers and body to separate files so `quote.py` sees JSON only:

```
curl -sS -o /tmp/x402-body.json -D /tmp/x402-headers.txt \
  "https://reader.outbid.sh/scrape?url=https://example.com/"
python3 scripts/quote.py < /tmp/x402-body.json
```

Expect HTTP 402 and an `accepts` array. `scripts/quote.py` is display-only
(no network, no keys). Prefer the `solana` / `solana:` row. Amount `5000`
is $0.005 USDC. The paying MCP settles from the `PAYMENT-REQUIRED` header,
not from this parsed body.

### 2. Pay under a ceiling

Arm payments only when the user wants the page. Edit the installed MCP
block, then start a **new session** and call the MCP `scrape` tool
(not `npx` — that starts the stdio server and hangs):

```yaml
mcp_servers:
  x402-reader:
    env:
      X402_READER_PAYMENTS_ENABLED: "1"
      SVM_PRIVATE_KEY_FILE: "${SVM_PRIVATE_KEY_FILE}"
      X402_READER_MAX_USDC_PER_CALL: "0.05"
      X402_READER_MAX_USDC_TOTAL: "1"
```

Use `EVM_PRIVATE_KEY_FILE` instead of `SVM_PRIVATE_KEY_FILE` for Base.
Caps refuse above the limit **before** signing. Do not `read_file` the
key. Do not paste key material into the transcript.

### 3. Verify

A paid 200 is JSON `{ok, title, markdown, word_count}` plus a
`PAYMENT-RESPONSE` header with `payer` and `transaction`. A leftover 402
is a failed pay, not a quote.

## Pitfalls

- **MPP vs x402.** A `www-authenticate: tempo` header is `mpp-agent`, not
  this skill.
- **No silent browse upgrade.** `needs_browser` is free and terminal.
- **Do not curl a 402 and stop** when the user asked for the page — quote,
  then pay under the cap, or report the price and stop.
- **Wallet keys stay out of context.** Point at a file path; never dump it.
- **npx is the MCP launcher, not a pay command.**

## Verification

```
hermes skills install official/payments/x402-reader
hermes mcp install x402-reader
hermes mcp test x402-reader
```

`hermes mcp test` must list `scrape` and `browse`. A quote against
`https://example.com/` must show amount 5000.
