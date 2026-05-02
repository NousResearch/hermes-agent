# Payment providers

The personal-shopper skill ships with three payment providers. The
agent picks one based on the user's country and stated preferences.

## Comparison

| Provider | UX | Country | Setup |
|---|---|---|---|
| `apple_pay_handoff` | 2 taps total: in-chat selection + biometric on merchant page | Worldwide (any merchant accepting Apple/Google Pay) | None |
| `privacy_com` | Agent emits a single-use card via API; user pastes PAN/CVV at checkout (1 paste, no biometric) | US only (requires US bank account funding) | `PRIVACY_API_KEY` env var |
| `manual_paste` | User generates a one-time card in their banking app, pastes it back; agent folds the cart URL + card into one ready-to-copy message | Worldwide (any fintech with disposable cards: Revolut, Monzo, Wise, Lydia, N26, Bunq) | None |

## Recommendation by country

| Country | Recommended provider | Why |
|---|---|---|
| US | `privacy_com` | Best UX -- the only path where the agent itself emits the card. Free tier (12/month) covers personal use. |
| FR / EU / UK / CA / AU / SG / JP | `apple_pay_handoff` | Universal, biometric satisfies SCA, no setup. The 2-tap reality is the price for working without bank partnerships. |
| Anywhere with a fintech with disposable cards | `manual_paste` | Power-user fallback for users who want full control over which card is charged for what. |

## Why we don't have a 'global agent emits card' provider

The market for individual-targeted virtual card APIs is sparse:

- **Revolut, Monzo, Wise, Lydia, N26, Bunq, Boursorama** -- all have
  disposable cards in their app but no public API for individuals.
  PSD2 and licensing make this a non-starter for now.
- **Curve** -- aggregator, not an issuer.
- **Stripe Issuing, Marqeta, Lithic** -- B2B only, require KYB/business
  entity.
- **Privacy.com** -- the lone exception. Personal API, free tier, but
  US-only because of funding requirements.

The "global 1-tap pay+order via agent" promise is being prepared by
Stripe/Mastercard/Apple under the agentic-commerce umbrella, but is
not live as of 2026 for a particulier without business setup.

## Operational notes

- The provider is selected **per-user**, not per-purchase. A user in
  France using `apple_pay_handoff` always gets the same flow until
  they explicitly switch via /settings.
- The fallback chain on failure is `provider -> apple_pay_handoff`.
  If `issue_card_privacy.py` 5xx's or returns a non-200, the agent
  should send the user the cart URL with Apple Pay instructions and
  a one-line note explaining the fallback.
- The agent must NEVER persist a full PAN/CVV. The provider scripts
  emit them to stdout for one-shot delivery to the user; the agent
  forwards them in a single chat message and forgets them.
