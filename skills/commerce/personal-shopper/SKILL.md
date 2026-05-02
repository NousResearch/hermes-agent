---
name: personal-shopper
description: Search the web for a product the user described, compare offers, prepare a prefilled cart, and hand off the payment with the right method for the user's country. Universal (any merchant), agnostic on the LLM provider, runs entirely on free infrastructure (SearXNG search) by default. Supports Apple/Google Pay handoff, Privacy.com single-use card emission (US), and manual-paste from any banking app's disposable cards.
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [commerce, shopping, payment, cart, checkout, virtual-card, agentic]
    related_skills: []
prerequisites:
  commands: [python3, curl]
  python_packages: [httpx, selectolax, playwright]
---

# Personal Shopper

Lets the agent shop for the user end-to-end:

1. **Discover** -- the user describes what they want in natural language ("cafe en grain bio 1kg le moins cher", "USB-C 64GB drive under 15 EUR", "a cozy reading lamp under 40 EUR"). The skill searches the open web through a self-hosted SearXNG instance (or any SearXNG public mirror) and filters out comparators / aggregators.

2. **Compare** -- the agent feeds the search hits back into the conversation (or to a small extractor script) to produce a structured top-3 ranked by best unit price.

3. **Confirm** -- the user picks one offer with a single tap (inline keyboard via the Hermes gateway).

4. **Cart-build** -- a Playwright-driven inspector resolves the merchant's platform (Shopify, WooCommerce, or 'unknown') and produces a prefilled cart URL the user can finish in one or two taps.

5. **Pay** -- the chosen `payment provider` decides what message the agent sends back. Three providers ship with this skill:

   - `apple_pay_handoff` (default, universal) -- agent sends the cart URL, user pays with Apple Pay / Google Pay / saved card on the merchant's checkout page. Biometric satisfies SCA in the EU.
   - `privacy_com` (US) -- the agent calls the Privacy.com personal API (free tier, no business entity required) to emit a single-use virtual card capped to the exact purchase amount, then ships PAN+CVV+expiry to the user in a single Telegram/Discord/Slack message alongside the cart URL.
   - `manual_paste` (universal fallback) -- agent asks the user to generate a one-time card in their own banking app (Revolut, Monzo, Wise, Lydia, N26, Bunq...), parses the pasted PAN+expiry+CVV, and folds it back into a single ready-to-copy message at the merchant's checkout.

The skill is provider-agnostic on the LLM side (it doesn't make LLM calls itself -- the agent does, that's Hermes's job) and merchant-agnostic on the buy side (any French/EU/US e-commerce site works through the universal Shopify/WooCommerce/manual auto-detect).

## When to use

Use this skill when the user asks to find / compare / buy / order something. Phrases that trigger it:

- "trouve-moi le moins cher"
- "achete-moi X" / "commande-moi X"
- "find me the cheapest..."
- "compare prices for..."
- "I need a new ..."
- "prepare an order for..."

## When NOT to use

- The user is not actually shopping (just asking what something costs for context). Stick to plain `web_extract`.
- The user is on a B2B procurement workflow with a vendor portal -- this skill targets retail e-commerce.
- The merchant requires a non-public account (corporate intranet, etc.).

## Quick reference

```bash
# 1. Search + compare (returns ranked offers as JSON)
python3 SKILL_DIR/scripts/search_offers.py \
  --query "cafe en grain bio 1kg le moins cher" \
  --searxng-url http://127.0.0.1:8888 \
  --max-results 12 --json

# 2. Build cart for the chosen URL (returns cart URL + detected price)
python3 SKILL_DIR/scripts/build_cart.py \
  --url "https://www.terresdecafe.com/products/foo" \
  --quantity 1 --json

# 3a. Issue a Privacy.com single-use card (US users only)
python3 SKILL_DIR/scripts/issue_card_privacy.py \
  --amount-eur 19.90 \
  --merchant "Terres de Cafe" \
  --product "Foo grain 1kg" \
  --json

# 3b. Parse a card the user pasted from their own banking app
python3 SKILL_DIR/scripts/parse_pasted_card.py \
  "1234 5678 1234 5678 12/27 123" --json
```

Each script writes one JSON object to stdout. The agent parses it and renders the next message in the conversation.

## Configuration

The skill works with no configuration if you use:

- `apple_pay_handoff` (default) -- nothing to configure
- A reachable SearXNG instance (the user's own, the system-wide one on the same host, or a public mirror like `searx.be`)

For agent-issued cards via Privacy.com, set:

```bash
export PRIVACY_API_KEY="sk_live_..."   # https://privacy.com -> Account -> Developers
```

Privacy.com requires a US bank account to fund the issuing balance. Free tier: 12 cards/month.

For manual-paste, no setup needed -- the user generates the card in their banking app on demand.

## Spending policy

Hard-coded conservative defaults the agent should respect even if the user nudges them:

- Per-purchase cap : 50 EUR
- Per-day cap     : 100 EUR

Override via env (only do this if the user explicitly asks):

```bash
export SHOPPER_SPEND_CAP_EUR_PER_PURCHASE=80
export SHOPPER_SPEND_CAP_EUR_PER_DAY=150
```

The agent MUST re-check the cap after the live page price is scraped (it can differ from the search snippet). `build_cart.py` returns the live price; the agent enforces the cap before showing the user any cart link.

## End-to-end flow (recommended agent loop)

1. User: free-text request.
2. Agent: read this SKILL.md, then call `search_offers.py` with the user's query.
3. Agent: render the top-3 with the gateway's inline-keyboard (Telegram, Discord, Slack -- whichever Hermes is connected to).
4. User: taps a button (or replies "le 1er").
5. Agent: enforce cap; call `build_cart.py` on the chosen offer URL.
6. Agent: re-enforce cap with the scraped live price.
7. Agent: pick the right payment provider for the user (see `references/payment-providers.md`):
   - if user is US and has set `PRIVACY_API_KEY`, call `issue_card_privacy.py`
   - else if user wants manual-paste, ask them to paste a card from their banking app and call `parse_pasted_card.py`
   - else (default) just send the cart URL with Apple/Google Pay instructions.
8. Agent: deliver one final message to the user with the cart URL + (optionally) the card details. The user finishes the payment on the merchant's UI.

## References

- `references/payment-providers.md` -- comparison matrix + which to recommend per country.
- `references/privacy-com-setup.md` -- one-page guide for the Privacy.com API key flow.
- `references/manual-paste-flow.md` -- what the user does in their banking app, with screenshots-by-text.
- `references/spend-policy.md` -- rationale for the caps and audit-log design.

## Why this design

- **Discovery** -- restricting search to known good merchants is brittle. Filtering out comparator/aggregator domains gives 95% of the same UX with 5% of the maintenance.
- **Robust checkout** -- we never submit payment data, so no captcha race / fraud trip. The user's own session does the final tap.
- **Liability** -- the user does the final confirmation on the merchant's UI. The agent never moves money on its own. Per-purchase + per-day caps enforced before any cart link is generated.
- **Provider-agnostic** -- the LLM side is delegated to Hermes (this skill makes no LLM calls). The merchant side auto-detects platform. The payment side is pluggable via the three providers above.

## Limitations

- US-only `privacy_com` because of Privacy.com's funding requirement. The 'true 1-tap pay+order via agent' UX only exists in this provider, in this jurisdiction.
- The default `apple_pay_handoff` is 2 taps total in practice (one in the chat, one biometric on the merchant). This is the price for working everywhere without bank partnerships.
- `manual_paste` works for any user with disposable cards in their banking app, but adds a few taps in their banking app for each purchase.
