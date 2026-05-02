# Manual-paste flow (universal fallback)

Use when the user wants the agent to handle the cart but prefers to
generate the card themselves in their own banking app. Works in every
country with a fintech that supports disposable / one-time cards.

## How the agent walks the user through it

1. Agent: "Achat pret a confirmer -- 14.20 EUR chez Foo. Genere une
   carte virtuelle a usage unique dans ton app bancaire avec un
   plafond de 15 EUR. Colle-la ici sous la forme `NUMERO MM/AA CVV`."

2. User opens their banking app and generates a card. Quick reference
   per provider:

   | App | Path |
   |---|---|
   | Revolut | Cards &rarr; + &rarr; Disposable virtual card |
   | Monzo | Account &rarr; Virtual card |
   | Wise | Cards &rarr; New virtual card |
   | Lydia | Comptes &rarr; Carte virtuelle one-shot |
   | N26 | Cards &rarr; Add card &rarr; Virtual MasterCard |
   | Bunq | Cards &rarr; Order Virtual card |
   | Boursorama | Comptes &rarr; e-Carte Bleue |
   | Credit Mutuel | Mes operations &rarr; e-Carte Bleue |

3. User pastes the card. Agent runs `parse_pasted_card.py`:

   ```bash
   python3 SKILL_DIR/scripts/parse_pasted_card.py \
     '4111 1111 1111 1111 12/27 123' --json
   ```

4. Agent sends back ONE message with the cart URL + the card details
   formatted for easy copy-paste at the merchant's checkout. After
   sending, the agent drops the PAN/CVV from working memory.

## Privacy / safety notes

- The agent never persists the PAN/CVV -- only `last4` is logged for
  audit.
- The user keeps control of which card is charged: they pick the
  funding source in their banking app, set the spend limit, and the
  card auto-revokes (or stays inactive) after the single charge.
- This flow is the safest in jurisdictions where data-residency or
  consent requirements would forbid an external API issuer.

## Edge cases

- **User pastes wrong format** -- `parse_pasted_card.py` exits 2 with
  `{"error": "could not parse..."}`. The agent re-prompts with the
  expected format.
- **Card declined at checkout** -- usually because the user's bank
  enforces 3DSecure on disposable cards. The user re-runs the bank's
  3DS challenge in their app and retries the merchant payment with
  the same card.
- **PAN starts with 6 (Discover) or 3 (Amex)** -- `parse_pasted_card.py`
  handles 12-19 digit PANs uniformly, no provider-specific logic.
