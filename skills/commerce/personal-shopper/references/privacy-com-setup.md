# Privacy.com setup (US users)

[Privacy.com](https://privacy.com) is the only consumer-facing virtual
card issuer with a public API for individuals. It's US-only because
the issuing balance must be funded from a US bank account (ACH).

## Onboarding

1. Sign up at <https://privacy.com> with a US bank account or US debit card.
2. Complete the standard KYC (name, address, SSN last 4, DOB).
3. Once your account is verified, fund the issuing balance ($1
   minimum on the free tier).

## API key

1. In the dashboard, go to **Account &rarr; Developers &rarr; API Keys**.
2. Click **Generate API Key**. Copy it -- it's shown only once.
3. Export it in the agent's environment:

   ```bash
   export PRIVACY_API_KEY="sk_live_..."
   ```

## Free tier

- 12 cards per month at no cost
- Each card auto-revokes after first use when emitted with `type=SINGLE_USE`
- No subscription required

## Limits

- Cards are USD only. The skill caps spend at `amount_eur * 1.10` USD
  to account for FX swings between issuance and the merchant charge.
- International merchants accept Visa/Mastercard from Privacy.com but
  some EU merchants may decline the foreign card; in that case the
  user falls back to `apple_pay_handoff` or `manual_paste`.
- Privacy.com cards do not support 3DSecure -- some EU merchants
  require it for first-time charges, which would block a Privacy
  card. This is rare for known/established e-commerce sites.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Bad/missing `PRIVACY_API_KEY` | Re-generate via Developers tab |
| `403 Forbidden` | Account not yet verified | Finish KYC + funding |
| `429 Too Many Requests` | Hit the 12/month free quota | Wait or upgrade |
| Card declined at checkout | Merchant blocks foreign cards | Fall back to `manual_paste` or `apple_pay_handoff` |
