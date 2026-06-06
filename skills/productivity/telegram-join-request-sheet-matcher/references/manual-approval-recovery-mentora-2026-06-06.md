# Manual approval recovery — Mentora / Le Mentorat Amz SC (2026-06-06)

## Context

The operator manually approved a Telegram join request before the automation could capture it. The goal was to recover the Telegram identity and write a durable access ledger entry so a future Stripe unpaid/canceled event can revoke the member once the Stripe customer/subscription reference is added.

Target group:

- Title: `Le Mentorat Amz SC`
- Chat ID: `-1002731462979`
- Userbot: `@Arnaud_amzsc`

## Verification sequence

1. Check userbot rights with the existing admin-check script:

```bash
cd /Users/arnaud/.hermes/telegram-userbot
HOME=/Users/arnaud /Users/arnaud/.local/bin/uv run --with telethon \
  python scripts/check_membership_admin.py --chat-id -1002731462979
```

Expected useful output:

```json
{
  "role": "ChannelParticipantAdmin",
  "rights": {
    "can_invite_users": true,
    "ban_users": true
  }
}
```

If Telethon still reports `ChannelParticipantSelf`, the account is not yet admin even if the UI seemed updated.

2. Search current group participants, not just pending requests. Once manually approved, the user is no longer in `requested=True` importers.

Search variants used successfully: `Mozart`, `Marie-Anne`, `Marie Anne`, `Marie`, `Anne`, `Marianne`.

3. Confirm the candidate with `channels.GetParticipantRequest` before writing the ledger.

## Recovered member

The member was recovered as:

```json
{
  "id": 8629965608,
  "first_name": "Mozar",
  "last_name": "Marie-Anne",
  "username": null,
  "display_name": "Mozar Marie-Anne"
}
```

Participant confirmation returned `is_member: true`.

## Ledger shape used

Path created:

```text
/Users/arnaud/.hermes/state/mentora_access_control/access_ledger.json
```

Initial record status:

```json
{
  "key": "telegram:8629965608",
  "status": "active_manual_link_pending_stripe",
  "source": "manual_admin_approval_recovered_after_entry",
  "chat_id": -1002731462979,
  "chat_title": "Le Mentorat Amz SC",
  "telegram_user_id": 8629965608,
  "telegram_username": null,
  "telegram_display_name": "Mozar Marie-Anne",
  "telegram_first_name": "Mozar",
  "telegram_last_name": "Marie-Anne",
  "stripe_customer_id": null,
  "stripe_subscription_id": null,
  "email": null,
  "payment_status": "unknown_to_link"
}
```

## Durable lesson

For paid-access Telegram automations, handle two capture paths:

- **Normal path:** pending join request matched to payment/customer record, then approved and written to ledger.
- **Recovery path:** already-approved member found via participant search, written to ledger as `active_manual_link_pending_stripe`, then later enriched with Stripe identifiers.

Do not claim payment-based revocation is reliable until the recovered ledger row has a Stripe customer/subscription reference linked to the Telegram user ID.
