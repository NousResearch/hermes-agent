# Mentora-style Stripe ↔ Telegram paid access control

Session pattern captured 2026-06-06.

## Scenario

A Telegram group requires access control based on Stripe subscription/payment status. The operator wants a Google Sheet they can inspect manually while an admin Telegram userbot performs approvals/removals only when the mapping is reliable.

Business logic:

- Paid/active customer requests Telegram access → approve join request or keep membership.
- Customer becomes `past_due`, `unpaid`, `canceled`, or `incomplete_expired` → remove them from the Telegram group after live mode is explicitly approved.
- Optionally send a short recovery message explaining that payment failed and access can be reactivated after payment is updated.

## Concrete architecture

Recommended local files:

- Worker: `~/.hermes/scripts/mentora_access_control.py`
- Wrapper: `~/.hermes/scripts/run_mentora_access_control.sh`
- Ledger/state: `~/.hermes/state/<workflow>/...`
- Telegram userbot/session: `~/.hermes/telegram-userbot/...`

Recommended Google Sheet tabs:

- `ACCESS_LEDGER`: one row per Stripe↔Telegram identity mapping.
- `ACTION_LOG`: append-only actions, dry-run previews, errors, removals, approvals.
- `CONFIG`: visible business rules/status policies.
- `PENDING_REVIEW`: duplicates, missing IDs, ambiguous customers, manually approved members without Stripe reference.

## Key implementation notes

1. Scan every relevant Stripe account/key. Do not assume one account contains all subscriptions.
2. Keep `stripe_account`/source-key name in the Sheet so future debugging knows where the subscription came from.
3. Use Stripe subscription/customer IDs as payment identifiers and Telegram numeric user IDs as Telegram identifiers.
4. Never remove access by display name or username alone.
5. For manually approved members, search current Telegram participants, confirm membership with Telethon, and add the Telegram ID to the ledger even if Stripe enrichment is still pending.
6. Dry-run should still sync the Sheet and produce decisions, but no Telegram approval/removal and no automated message.
7. Live `--apply` must be explicit. Paid revocation has higher downside than a join-request approval.
8. When removing access, perform the Telegram side effect first, then record `last_action_at`, `access_status`, `decision`, and action-log row.

## Example dry-run summary shape

```json
{
  "ok": true,
  "mode": "dry_run",
  "stripe_subscriptions_matched": 219,
  "stripe_active_subscriptions": 39,
  "stripe_errors": [],
  "actions": [],
  "reviews_count": 0,
  "sheet_synced": true
}
```

## Safe live-mode gate

Use a wrapper pattern like:

```bash
~/.hermes/scripts/run_mentora_access_control.sh          # dry-run/sheet sync
~/.hermes/scripts/run_mentora_access_control.sh --apply  # live side effects after explicit GO
```

Scheduler default should be dry-run/local delivery. Convert to live only after the operator has reviewed the Sheet and explicitly approved automatic approvals/removals.
