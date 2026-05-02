# Spending policy + audit log

The personal-shopper skill enforces two cumulative caps:

- **Per-purchase cap** (default 50 EUR) -- the maximum a single Buy
  tap can engage.
- **Per-day cap** (default 100 EUR) -- the maximum the user can engage
  across all purchases since 00:00 UTC today.

The defaults are conservative on purpose. Users who want higher caps
have to explicitly say so; the agent overrides via env vars:

```bash
export SHOPPER_SPEND_CAP_EUR_PER_PURCHASE=80
export SHOPPER_SPEND_CAP_EUR_PER_DAY=150
```

## When the agent runs the cap check

Every Buy goes through `cap_check.py` **twice**:

1. **Pre-cart-build** -- using the price extracted from the search
   snippet. Cheap, fast, blocks obvious over-cap selections.
2. **Post-cart-build** -- using the price scraped from the live
   product page (returned by `build_cart.py`). The agent calls
   `cap_check.py --commit` here so the ledger reflects only verified
   prices.

The double-check is intentional: search snippets can be stale or wrong
(strikethrough prices, multiple sizes shown together), so we never
trust them on their own.

## Ledger format

A tiny JSON file at `/tmp/personal-shopper-ledger.json` (or a path
the agent specifies via `--ledger`):

```json
{
  "entries": [
    {"price_eur": 14.20, "quantity": 1, "created_at": 1714663200},
    {"price_eur": 19.90, "quantity": 1, "created_at": 1714668000}
  ]
}
```

`spent_today()` sums entries since the day's 00:00 UTC. There's no
explicit cleanup -- the file grows by ~50 bytes per purchase, which
is fine for personal use. The agent can rotate the file weekly if
desired.

## Why this is belt-and-suspenders

The agent never moves money in V0 -- the user does the final tap on
the merchant's UI. Even so, the cap matters because:

- It prevents the agent from generating a `cart_url` that would lead
  to a surprise charge if the user taps reflexively.
- It draws a clear line in the audit log between "agent prepared a
  cart" and "agent prepared a cart over the daily limit (refused)".
  The latter is a useful security signal in case of a prompt-injection
  attempt.

## Audit log

Every script writes one structured event per call (via stderr or via
the agent's `delegate` tool). At minimum the agent should log:

- `query` events (`bot.query`) -- text + user_id
- `cart.built` -- url + price + strategy
- `policy.block` -- reason + offer
- `payment.dispatched` -- provider key + last4
- `manual_paste.received` -- last4 only
- `payment.error` -- exception class

Whatever audit sink Hermes provides (memory, dedicated log, MCP
endpoint) is fine; the events should be machine-readable so the
user can `/audit` them later.
