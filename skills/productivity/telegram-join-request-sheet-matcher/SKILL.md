---
name: telegram-join-request-sheet-matcher
description: Use when a Google Sheet receives landing/form leads and a Telegram userbot/admin account must match pending join requests, approve confident entrants, and write Telegram identity/audit columns back to the Sheet.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [telegram, google-sheets, lead-matching, launchagent, automation]
    related_skills: [google-workspace, scheduled-agent-operations]
---

# Telegram Join Request ↔ Google Sheet Lead Matcher

## Overview

This skill documents the architecture for approving Telegram community join requests only when they can be matched to a recent inbound lead in a Google Sheet. It is designed for flows such as:

1. a landing page, form, webhook, or Zapier/Make scenario appends a lead row to a Google Sheet;
2. the lead clicks a Telegram group/channel invite link that requires admin approval;
3. a local Hermes worker reads the Sheet and pending Telegram join requests;
4. if the request is confidently linked to the lead, the worker approves the Telegram request;
5. the worker writes the Telegram identity and match metadata back into the Sheet.

The goal is not perfect identity proof. The goal is fast, explainable matching with clear guardrails and an audit trail so operators can later connect “this lead row” to “this Telegram member”.

For paid communities, the same class of workflow can be extended into **subscription access control**: maintain a Google Sheet / ledger mapping Stripe customers or subscriptions to Telegram user IDs, approve paid join requests, and remove Telegram access when the payment/subscription becomes unpaid, canceled, or expired. Keep this extension dry-run until the operator explicitly approves live removals.

## When to Use

Use this skill when:

- A Google Sheet contains incoming leads from a landing page, ad campaign, webinar, or form.
- A Telegram group/channel has **pending join requests** instead of fully open membership.
- A local Telegram **user account** is connected through Telethon and has admin rights.
- The operator wants confident entrants auto-approved and recorded in the Sheet.
- Matching can be based on weak but useful signals such as first name plus close timing.
- The worker must run frequently, typically every 30-120 seconds on an always-on Mac.
- A paid community needs a visible admin Sheet that links `stripe_customer_id` / `stripe_subscription_id` / email to Telegram identity and access status.
- The business rule is: active paid subscription keeps or grants Telegram access; failed/unpaid/canceled subscription removes Telegram access and optionally sends a recovery message.

Do **not** use this skill when:

- You only have a Telegram bot token. Bots often cannot see/approve all join-request workflows the same way a user admin can.
- The Telegram account is not admin in the target group/channel.
- The Sheet does not contain a reliable timestamp or any name signal.
- The business consequence of false approval is high enough to require manual review only.
- The task is just sending Telegram messages; use Telegram/userbot-specific procedures instead.

## Prerequisites

### Google / Sheets

- Google Workspace OAuth is authenticated for the Hermes profile that will run the worker.
- The active Google account can read and update the target Sheet.
- The Sheet has a stable file ID.
- The first tab/range is known, or the worker can discover the first sheet title.
- Minimum recommended source columns:
  - `created_at`: ISO timestamp of the lead submission.
  - `prenom` or equivalent first-name field.
  - `email`: useful for audit/debugging.
  - `telephone`: useful for audit/debugging, even if Telegram phone is not visible.
  - optional UTM/source columns for later attribution.

Recommended command checks:

```bash
python ~/.hermes/skills/productivity/google-workspace/scripts/setup.py --check
python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py drive search "AMZLive" --max 10
python ~/.hermes/skills/productivity/google-workspace/scripts/google_api.py sheets get SHEET_ID "A1:Z5"
```

### Telegram / Telethon

- A local Telegram userbot directory exists, for example:

```text
~/.hermes/telegram-userbot/
  config/.env
  sessions/community_user.session
```

- `config/.env` contains at least:

```bash
TG_USER_API_ID=...
TG_USER_API_HASH=...
TG_USER_SESSION=community_user
```

- The session is authenticated and can list dialogs:

```bash
cd ~/.hermes/telegram-userbot
uv run --with telethon python scripts/list_chats.py
```

- The account is admin in the target Telegram group/channel.
- Required/admin-relevant rights normally include:
  - `invite_users`: required to approve join requests.
  - `ban_users` / moderation rights: often present for community admins.
  - `delete_messages`: not required for matching but often confirms admin level.

### Local scheduler

For near-real-time approval, prefer a macOS LaunchAgent on the always-on Mac. Hermes cron is acceptable for slower polling, but sub-minute/one-minute operational polling is usually cleaner through LaunchAgent.

Prerequisites:

- Mac remains awake/online.
- The worker script is idempotent.
- A local ledger prevents duplicate approvals/writes.
- Only one scheduler runs the same worker.

## Architecture

```text
Landing/Form/Webhook
        │
        ▼
Google Sheet lead row
(created_at, prenom, email, phone, source...)
        │
        │ every 30-120s
        ▼
Local Hermes worker on Mac
        │
        ├─ reads recent Sheet rows
        ├─ reads Telegram pending join requests
        ├─ computes confidence score
        ├─ approves Telegram request if confident
        ├─ writes Telegram identity columns to Sheet
        └─ records local ledger + last_run logs
```

## Recommended Sheet Output Columns

For lead/join-request matching, add these columns automatically if missing:

- `telegram_validated_name`: display name seen on Telegram.
- `telegram_validated_user_id`: stable Telegram user ID.
- `telegram_username`: username if present.
- `telegram_validated_at`: when the worker approved/wrote the match.
- `telegram_join_request_at`: Telegram join-request timestamp.
- `telegram_match_score`: numeric confidence score.
- `telegram_match_reason`: human-readable explanation, e.g. `prenom=Fabienne; delta=0.6min`.

Only write these columns after Telegram approval succeeds, unless doing an explicit dry-run.

For paid subscription access control, create a separate ledger-style tab such as `ACCESS_LEDGER` plus `ACTION_LOG`, `CONFIG`, and `PENDING_REVIEW`. Recommended access columns:

- `stripe_customer_id`, `stripe_subscription_id`, `stripe_subscription_status`, `stripe_current_period_end`, `stripe_account` or source key name.
- `email`, `customer_name`, optional product/price metadata.
- `telegram_user_id`, `telegram_username`, `telegram_display_name`, `telegram_is_member`.
- `access_status` (`active`, `revoked`, `pending_review`, etc.).
- `decision` (`keep_access`, `approve_join`, `remove_access`, `needs_manual_review`).
- `last_checked_at`, `last_action_at`, `notes`.

Keep `ACTION_LOG` append-only so the operator can audit each approval, removal, skipped action, and dry-run decision.

## Matching Rules

A safe baseline matching policy:

1. Normalize names:
   - lowercase;
   - remove accents;
   - remove punctuation and repeated spaces.
2. Extract the lead first name from the Sheet.
3. Extract Telegram display tokens from `first_name + last_name` and optionally username.
4. Require the lead first name to appear in Telegram display tokens.
5. Require Telegram request timestamp to be after the lead timestamp, with a small tolerance for clock skew.
6. Apply a maximum time window, for example 45 minutes.
7. Score higher when the delta is very short:
   - 0-15 min: strong;
   - 15-30 min: medium;
   - 30-45 min: weak but possible;
   - >45 min: no auto-approval by default.
8. If several Sheet rows could match one Telegram request, choose the highest score and closest timestamp.
9. Never match two Telegram requests to the same Sheet row in one run.
10. Never auto-approve below the configured score threshold.

Example scoring:

```text
Base first-name match: 75
Delta <= 15 min:       +20
Delta <= 30 min:       +10
Delta <= 45 min:        +5
Telegram starts with first name: +5
Approve threshold:      80
```

This intentionally approves cases like:

```text
07:00 lead: Fabienne
07:03 Telegram request: Fabienne Hankard
=> approve, score high, write reason delta=3.0min
```

But it blocks cases like:

```text
07:00 lead: Fabienne
12:45 Telegram request: Fabienne XYZ
=> too far outside the timing window; manual review
```

## Stripe Subscription Access-Control Extension

When extending this workflow from lead approval to paid community access, use a deterministic worker that reads Stripe, Telegram, and the Google Sheet on every run.

### Subscription status policy

Default safe policy:

- Grant/keep access for Stripe statuses such as `active` and, if the business accepts it, `trialing`.
- Remove access for `past_due`, `unpaid`, `canceled`, and `incomplete_expired` only after the operator has explicitly approved live removals.
- Treat ambiguous records as `PENDING_REVIEW` rather than guessing: duplicate emails, multiple active subscriptions, missing Telegram ID, unknown Stripe account, or manually approved members without Stripe mapping.

### Multi-account Stripe scanning

If the business uses several Stripe secret keys/accounts, scan all configured sources and write the source/account into the ledger. Do not assume the first Stripe key is exhaustive. Emit `stripe_errors` separately from successful matches so one broken account does not hide valid results from another.

### Safe execution mode

Use dry-run as the default for scheduled runs until there is an explicit GO for live actions:

```bash
~/.hermes/scripts/run_ACCESS_CONTROL.sh          # dry-run / sheet sync only
~/.hermes/scripts/run_ACCESS_CONTROL.sh --apply  # live approve/remove after GO
```

In dry-run, it is acceptable to sync the Sheet and append decision previews, but do not approve join requests, ban/kick members, or send recovery messages. In live mode, perform the external side effect first, then record it in the Sheet/action log.

### Telegram removal and recovery message

For unpaid/canceled users, the live flow is:

1. Confirm the ledger has a stable `telegram_user_id` mapped to the Stripe subscription/customer.
2. Confirm the user is currently a participant.
3. Remove access using the admin user session.
4. Record the action in `ACTION_LOG` and update `ACCESS_LEDGER`.
5. Optionally send a short recovery message with the payment-update link. Keep the message operational and non-judgmental, e.g. “Ton paiement a échoué; régularise ici pour réactiver automatiquement ton accès.”

Never remove by display name alone.



The worker script should:

1. Load Google credentials from the intended Hermes profile.
2. Load Telegram credentials/session from the userbot directory.
3. Use a non-blocking lock to avoid overlapping runs.
4. Read the target Sheet headers and recent rows.
5. Ensure output columns exist.
6. Fetch pending Telegram join requests:

```python
from telethon import functions, types

res = await client(functions.messages.GetChatInviteImportersRequest(
    peer=peer,
    requested=True,
    offset_date=None,
    offset_user=types.InputUserEmpty(),
    limit=100,
))
```

7. Match requests to rows using deterministic, explainable rules.
8. Approve only confident matches:

```python
await client(functions.messages.HideChatJoinRequestRequest(
    peer=peer,
    user_id=req.user_id,
    approved=True,
))
```

9. Batch-update the Sheet with Telegram metadata.
10. Update a durable local ledger, for example:

```text
~/.hermes/state/<workflow-name>/ledger.json
~/.hermes/state/<workflow-name>/last_run.json
```

11. Stay quiet when there is nothing to report, so scheduled runs do not spam the user.
12. Print JSON only for dry-run, errors, or actual approvals.

### Wrapper shell script

Use a wrapper to pin paths and dependencies:

```bash
#!/bin/bash
set -euo pipefail
export HERMES_HOME="/Users/arnaud/.hermes"
export GOOGLE_HERMES_HOME="/Users/arnaud/.hermes"
cd /Users/arnaud
exec /Users/arnaud/.local/bin/uv run \
  --with telethon \
  --with google-api-python-client \
  --with google-auth \
  /Users/arnaud/.hermes/scripts/YOUR_MATCHER.py
```

### macOS LaunchAgent

Use a LaunchAgent for frequent polling:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.hermes.telegram-join-sheet-matcher</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/arnaud/.hermes/scripts/run_YOUR_MATCHER.sh</string>
  </array>
  <key>StartInterval</key>
  <integer>60</integer>
  <key>RunAtLoad</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/Users/arnaud/.hermes/logs/YOUR_MATCHER.out.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/arnaud/.hermes/logs/YOUR_MATCHER.err.log</string>
  <key>WorkingDirectory</key>
  <string>/Users/arnaud</string>
</dict>
</plist>
```

Install and verify:

```bash
plutil -lint ~/Library/LaunchAgents/com.hermes.telegram-join-sheet-matcher.plist
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.hermes.telegram-join-sheet-matcher.plist 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.hermes.telegram-join-sheet-matcher.plist
launchctl kickstart -k gui/$(id -u)/com.hermes.telegram-join-sheet-matcher
launchctl print gui/$(id -u)/com.hermes.telegram-join-sheet-matcher | grep -E 'state =|runs =|last exit code|run interval'
```

## Dry-Run First

Before enabling approval:

```bash
uv run --with telethon --with google-api-python-client --with google-auth \
  ~/.hermes/scripts/YOUR_MATCHER.py --dry-run
```

A good dry-run output includes:

```json
{
  "status": "dry_run",
  "pending_join_requests": 2,
  "matches": 2,
  "approved": [
    {
      "row": 50,
      "lead_first_name": "Fabienne",
      "telegram_name": "Fabienne Hankard",
      "score": 100,
      "reason": "prenom=Fabienne; delta=0.6min"
    }
  ]
}
```

Only switch to live mode after the dry-run shows matches the operator would approve manually.

## Verification Checklist

After installation, verify all of the following:

- [ ] Google OAuth check returns authenticated.
- [ ] The target Sheet can be read.
- [ ] The target Sheet can be updated.
- [ ] Telegram session can list dialogs.
- [ ] The target Telegram group/channel resolves by ID or username.
- [ ] The Telegram account is admin in the target.
- [ ] Pending join requests can be fetched.
- [ ] Dry-run produces expected matches and explanations.
- [ ] Live run approves only expected users.
- [ ] Pending join request count decreases after live approval.
- [ ] Sheet rows contain Telegram metadata after live approval.
- [ ] LaunchAgent or cron reports exit code 0.
- [ ] No duplicate scheduler is running for the same workflow.
- [ ] The ledger and last_run files are written.

## Manual-Approval Recovery / Paid Access Ledger

When the operator manually approved a member before the worker captured the pending join request, recover the linkage before enabling payment-based revocation:

1. Re-check the Telegram userbot's admin status in the target group after the operator grants rights; do not trust the UI until Telethon reports `ChannelParticipantAdmin` with `invite_users` and `ban_users`/moderation rights.
2. Search current group participants by likely first name, last name, spelling variants, and partial tokens. Manual approvals are no longer pending join requests, so `GetChatInviteImportersRequest(requested=True)` may correctly return zero.
3. Confirm the recovered account is still a participant with `channels.GetParticipantRequest` before writing the ledger.
4. Create or update a durable access ledger entry with the Telegram identity even if the Stripe fields are not known yet. Mark it explicitly, e.g. `active_manual_link_pending_stripe`, so the missing payment reference is visible.
5. Later, enrich the same record with `stripe_customer_id`, `stripe_subscription_id`, and email. Only after that mapping exists can unpaid/canceled Stripe events revoke the correct Telegram member reliably.

Recommended ledger fields for this recovery path:

- `chat_id`, `chat_title`
- `telegram_user_id`, `telegram_username`, `telegram_display_name`, `telegram_first_name`, `telegram_last_name`
- `stripe_customer_id`, `stripe_subscription_id`, `email`
- `status`, `payment_status`, `source`, `joined_or_found_at`, `updated_at`, `notes`

See `references/manual-approval-recovery-mentora-2026-06-06.md` for a concrete recovered-member example.

See `references/stripe-subscription-access-control-mentora-2026-06-06.md` for a concrete Mentora-style paid access-control pattern using Stripe + Telegram + Google Sheets.

## Common Pitfalls

1. **Using a bot instead of a user account.** Telegram join-request approval can require a real user admin session. Prefer Telethon with a userbot session when this workflow needs admin-level visibility.

2. **Forgetting admin rights.** Being a member is not enough. Confirm the participant type is `ChannelParticipantAdmin` or creator and inspect admin rights.

3. **Approving on first name alone.** First name is useful only with timing and source context. Keep a time window and score threshold.

4. **Timezone drift.** Sheet timestamps and Telegram timestamps should both be normalized to UTC before computing deltas.

5. **Writing Sheet metadata before Telegram approval.** If Telegram approval fails but Sheet write succeeds, the audit trail lies. Approve first, then write.

6. **Running two schedulers.** Duplicate LaunchAgents/Hermes crons can approve/write twice or create confusing logs. Keep one scheduler per workflow.

7. **No durable ledger.** Even if the Sheet has metadata columns, keep a local ledger for approved Telegram user IDs and recent run state.

8. **Too much log noise.** For 60-second polling, normal no-op runs should be silent. Print only dry-run, approvals, or errors.

9. **Assuming phone numbers are visible.** Telegram user phone is usually not visible unless contact relationships allow it. Do not depend on phone matching.

10. **Overwriting manual Sheet columns.** Only write the Telegram-owned columns. Never broad-update rows without column mapping.

11. **Going live on removals too early.** Paid-access revocation is higher risk than approving a join request. Keep scheduled jobs dry-run/sheet-sync until the operator gives an explicit GO for `--apply`.

12. **Removing access without a Stripe↔Telegram mapping.** A failed payment event alone is not enough. Require a stable `stripe_customer_id` or `stripe_subscription_id` mapped to a stable `telegram_user_id`; names and emails are review signals, not removal identifiers.

13. **Ignoring manually approved members.** If the operator manually approved someone before automation ran, recover their Telegram ID from current participants and enter it into the access ledger before relying on future unpaid/canceled events.

## AMZLive Reference Implementation

The AMZSC implementation created on the always-on Mac used:

- Sheet: `AMZLive - Leads Immersion Terrain`
- Sheet ID: `1iMumebLrbNOKj0PWC9wmM103i5MIKKJY2QiDDbmlwAE`
- Telegram target: `Amz Seller Consulting team 🚀` / `@amazonsellerconsulting`
- Telegram chat ID: `-1001807006590`
- Worker: `~/.hermes/scripts/amzlive_telegram_join_matcher.py`
- Wrapper: `~/.hermes/scripts/run_amzlive_telegram_join_matcher.sh`
- LaunchAgent: `~/Library/LaunchAgents/com.hermes.amzlive-telegram-join-matcher.plist`
- Polling interval: 60 seconds
- Match window: 45 minutes
- Approval threshold: 80
- State dir: `~/.hermes/state/amzlive_telegram_matcher/`

Example verified matches:

- Lead `Fabienne` at `2026-06-06T05:07:52Z` matched Telegram `Fabienne Hankard` at `2026-06-06T05:08:29Z`, score `100`.
- Lead `Sandy` at `2026-06-06T05:24:39Z` matched Telegram `Sandy` at `2026-06-06T05:24:48Z`, score `100`.

Treat these specifics as an example, not universal defaults. For a new community, always rediscover Sheet ID, Telegram chat ID, columns, rights, and desired matching thresholds.
