---
name: moltycash
description: >
  USDC payments from AI agents to humans via molty.cash. Use when the agent wants to
  tip someone, hire a person for a task, or create a pay-per-task gig. Payments settle
  on-chain via x402 on Base using Bankr wallet for signing.
  Do NOT use for token swaps, DeFi, or non-USDC payments.
metadata:
  {
    "clawdbot":
      {
        "emoji": "💸",
        "homepage": "https://molty.cash",
        "requires": { "bins": ["bankr"] },
      },
  }
---

# MoltyCash — Agent-to-Human Payments with USDC

[molty.cash](https://molty.cash) lets AI agents pay humans with USDC. Tip someone, hire them for a task, or post a gig for multiple people to earn from — all settled on-chain via [x402](https://x402.org) on Base.

This skill covers three actions: **tip**, **hire**, and **gig create**. All use the Bankr CLI (`bankr x402 call`) for x402 payment signing.

### `--max-payment`

`bankr x402 call` defaults to a $1 max payment. The total charged is amount + platform fee (see Fees below), so pass `--max-payment` when the total exceeds $1:

```bash
# Example: hire for $1.00 → fee is 3% ($0.03) → total $1.03 → needs --max-payment 1.03
bankr x402 call <url> --max-payment 1.03 ...
```

Max allowed value is `10`.

---

## Prerequisites

1. Bankr CLI installed and logged in (`bankr whoami` to verify)
2. Funded Bankr wallet (Base USDC)
3. `MOLTY_IDENTITY_TOKEN` — required for **tip**, **hire**, and **gig create**

### Getting an Identity Token

1. Login to [molty.cash](https://molty.cash) with your X account
2. Open the profile dropdown and click "Identity Token"
3. Generate your token and copy it
4. `export MOLTY_IDENTITY_TOKEN="your_token"`

---

## Tip

Send a USDC tip to any molty.cash user.

```bash
bankr x402 call https://api.molty.cash/0xmesuthere/a2a \
  --method POST \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tip",
    "params": {
      "amount": 0.10,
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'"
    }
  }'
```

### Tip any user

Replace `0xmesuthere` with any X handle:

```bash
bankr x402 call https://api.molty.cash/{username}/a2a \
  --method POST \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tip",
    "params": {
      "amount": 0.50,
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'"
    }
  }'
```

---

## Hire

Hire a specific person to complete a task. Payment is escrowed via x402. The person is auto-assigned and has 4 hours to submit proof.

```bash
bankr x402 call https://api.molty.cash/0xmesuthere/a2a \
  --method POST \
  --max-payment 1.03 \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "hire",
    "params": {
      "amount": 1.00,
      "description": "explain how bankr and MoltyCash integration works in a post",
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'"
    }
  }'
```

### Hire any user

Replace `0xmesuthere` with any X handle:

```bash
bankr x402 call https://api.molty.cash/{username}/a2a \
  --method POST \
  --max-payment 1.03 \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "hire",
    "params": {
      "amount": 1.00,
      "description": "Your task description here",
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'"
    }
  }'
```

### Hire Rules

| Rule | Detail |
|------|--------|
| Max amount | 10 USDC |
| Description | Max 500 characters |
| Assignment TTL | 4 hours to submit proof |
| Review deadline | 4h auto-approve if not reviewed |
| Hold period | 2h after approval before payment release |

---

## Gig Create

Create a gig that multiple people can earn from. You define the task, set a price per completion, and review submissions.

```bash
bankr x402 call https://api.molty.cash/a2a \
  --method POST \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "gig.create",
    "params": {
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'",
      "description": "Share a post about bankr and mention @moltycash on X",
      "price": 0.30,
      "quantity": 3
    }
  }'
```

### With eligibility criteria

```bash
bankr x402 call https://api.molty.cash/a2a \
  --method POST \
  --body '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "gig.create",
    "params": {
      "identity_token": "'$MOLTY_IDENTITY_TOKEN'",
      "description": "Share a post about bankr and mention @moltycash on X",
      "price": 0.30,
      "quantity": 3,
      "require_premium": true,
      "min_followers": 10000
    }
  }'
```

### Gig Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `description` | string | Yes | Task description, max 500 characters |
| `price` | number | Yes | USDC per completion |
| `quantity` | number | No | Number of slots (default 1) |
| `require_premium` | boolean | No | Require X Premium subscription |
| `min_followers` | number | No | Minimum follower count |
| `min_account_age_days` | number | No | Minimum account age in days |

### Gig Rules

| Rule | Detail |
|------|--------|
| Max total amount | 10 USDC |
| Max per-post price | 10 USDC |
| Gig deadline | 24 hours from creation |
| Assignment TTL | 4 hours to submit proof |
| Review deadline | 24h auto-approve if not reviewed |
| Hold period | 2h after approval; tweet re-checked before payment |

---

## Fees & Refunds

### Platform Fee

| Amount | Fee |
|--------|-----|
| < $1 | $0.01 flat |
| >= $1 | 3% |

The fee is added on top of the payment amount — the payer pays amount + fee.

### Refunds

- **Expired gigs**: Unclaimed slots are auto-refunded after the 24h gig deadline
- **Expired assignments**: Freed after 4h if no proof submitted — slot reopens for others
- **Unreviewed submissions**: Auto-approved after 4h if the payer doesn't review

---

## A2A Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST api.molty.cash/a2a` | Global — gig creation |
| `POST api.molty.cash/{username}/a2a` | Per-user — tip or hire a specific person |

## Links

- [molty.cash](https://molty.cash)
- [bankr.bot](https://bankr.bot)
- [x402.org](https://x402.org)
