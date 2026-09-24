---
name: credit-card-optimizer
description: Compare owned cards using explicit rewards assumptions.
version: 0.1.0
author: Mark (unsupportedpastels), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [credit-cards, rewards, benefits, calculations]
    category: bots
    related_skills: [grounded-citations, xlsx]
---

# Credit Card Optimizer Workflow Skill

Compare cards the user already holds using nickname-only inventory and transparent arithmetic. Synthetic terms are examples only; consequential recommendations require current official issuer verification.

## When to Use

- The user wants a card choice for a purchase or a benefit deadline tracker.
- The user supplies card nicknames, terms, caps, and preferences.
- Do not use to apply, pay, transfer, redeem, or access an issuer account.

## Prerequisites

The first useful task needs only nickname-level rows based on `templates/card-inventory.csv`. Never request or store full card numbers, security codes, passwords, account credentials, or verification codes. Use `terminal` with `scripts/compare_cards.py` for arithmetic. Current changing terms require `web_search`/`web_extract` against current official issuer sources, if configured.

## How to Run

1. Copy `templates/card-inventory.csv` into the owning profile workspace under a collision-safe filename.
2. Try `references/sample-input.md` with its clearly synthetic terms.
3. Run `terminal(command="python scripts/compare_cards.py --cards <inventory.csv> --amount <amount>")` from this skill directory.
4. Score the recommendation with `references/expected-output-rubric.md`.

## Quick Reference

The helper computes `eligible spend × reward rate + remaining applicable credit`, using decimal arithmetic and returning JSON sorted by estimated value. It does not validate issuer terms or modify the inventory.

A monthly benefit review may be recommended. Do not automatically create a scheduled job; explicit activation must approve the tracker path, cadence, reminders, and retention.

## Procedure

### 1. Minimize the inventory

Collect nickname, product name if needed, reward rate, relevant cap remaining, credit remaining, and user-stated protections or preferences. Reject secrets. **Done when** every card is identifiable without account credentials or full numbers.

### 2. Separate assumptions from verified terms

Label each term user-provided, synthetic, or current-official. For a consequential real choice, verify changing fees, credits, exclusions, and category rules from a current official issuer page and record the as-of date. **Done when** every decisive term has a provenance label.

### 3. Calculate candidates

Use `scripts/compare_cards.py` through `terminal`; do not estimate arithmetic in prose. Preserve the input CSV. **Done when** machine-calculated values exist for all eligible nicknames and errors are surfaced.

### 4. Add non-reward considerations

Compare purchase protection, warranty, foreign fees, utilization preference, and cap opportunity cost only from supplied or verified facts. **Done when** the recommendation distinguishes quantitative value from qualitative tradeoffs.

### 5. Recommend with verification status

Lead with the best card under stated assumptions, show alternatives where values depend on redemption, and apply `references/expected-output-rubric.md`. **Done when** the result names assumptions, calculation, issuer-verification status, and no account action.

## Pitfalls

- Synthetic sample rates and credits are not real issuer terms.
- Points are not cash unless a redemption value is explicitly supplied.
- Caps and credits may reset on different calendars.
- This is rewards guidance, not debt, tax, or investment advice.

## Verification

- [ ] Inventory contains nicknames only and no secrets.
- [ ] Arithmetic came from the helper and inputs are shown.
- [ ] Changing terms are current-official verified or visibly unverified.
- [ ] No application, payment, redemption, transfer, or account change occurred.
