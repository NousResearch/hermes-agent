---
name: email-delivery-routing
description: How to send email and route delivery targets in the Hermes gateway. Google Workspace OAuth, NOT SMTP.
---

# Email Delivery Routing

## Architecture
- The gateway's `email.py` adapter defaults to SMTP (requires `EMAIL_PASSWORD`) — this is **broken** for this setup.
- Actual email sending uses **Google Workspace OAuth via Gmail API**.
- `EMAIL_PASSWORD` in `.env` is intentionally commented out. SMTP is not the path.
- **Himalaya is UNINSTALLED** — do not reinstall or use for email.
- **NEVER** claim email is fixed without actually sending a test email and verifying delivery.

## Token Files — NAMED COPIES FOR CLARITY

| Name | Path | Account | Purpose |
|---|---|---|---|
| Jared token | `/root/.hermes/jared_google_token.json` | jared.zimmerman@gmail.com | Inbox scanning, calendar, contacts — READ operations on user's data |
| Jared token (live) | `/root/.hermes/google_token.json` | jared.zimmerman@gmail.com | Same as above — this is the active file used by `google_api.py` by default |
| Indigo token | `/root/.hermes-indigo/google_token.json` | mx.indigo.karasu@gmail.com | SENDING briefings, Indigo's own inbox |
| Indigo token (named) | `/root/.hermes-indigo/indigo_google_token.json` | mx.indigo.karasu@gmail.com | Named backup copy for clarity |

**CRITICAL**: Never overwrite `/root/.hermes/google_token.json` with Indigo's token. It MUST always be Jared's token. The previous "fix" of copying Indigo's token over Jared's broke inbox scanning for Jared's account.

## Identity Routing
| Context | Token (HERMES_HOME) | From | To |
|---|---|---|---|
| Briefings (Vesper) | `/root/.hermes-indigo` (Indigo) | mx.indigo.karasu@gmail.com | jared.zimmerman@gmail.com |
| Dispatch triage/labeling | `/root/.hermes` (Jared) | N/A (acting on inbox) | jared.zimmerman@gmail.com |

## Critical Rules
1. **Briefings ALWAYS come FROM Indigo TO user.** Use the agent's OAuth token (`HERMES_HOME=/root/.hermes-indigo`), not Jared's.
2. **Dispatch triage acts ON the user's inbox.** Use Jared's token (`HERMES_HOME=/root/.hermes` or default) to read/label/draft.
3. **Do NOT hard-map `dispatch = user email`.** Map by task context — some dispatch actions send as Indigo, others act on the user's inbox.
4. If `EMAIL_PASSWORD` is missing, fall back to Gmail API via `google_token.json`, not SMTP.
5. Never tell the user "it's fixed" until you've verified delivery — **by checking Jared's inbox**, not Indigo's SENT folder.
1. **Briefings ALWAYS come FROM Indigo TO user.** Use the agent's OAuth token, not the user's.
2. **Dispatch triage reads Indigo's inbox** (the only accessible account). If Jared's inbox needs scanning, a separate `jared.zimmerman@gmail.com` OAuth token must be re-established.
3. **Do NOT hard-map `dispatch = user email`.** Map by task context — some dispatch actions send as Indigo, others act on the user's inbox.
4. If `EMAIL_PASSWORD` is missing, fall back to Gmail API via `google_token.json`, not SMTP.
5. Never tell the user "it's fixed" until you've verified delivery — **by asking the user to confirm receipt**, not by checking SENT labels.
6. **`google_api.py gmail get` has a header display bug** — always verify headers via raw Gmail API if header values matter.

## Debugging
- "no delivery target resolved for deliver=email" = the cron job `deliver` field needs the `email:` prefix format, e.g. `deliver: "email:jared.zimmerman@gmail.com"`. Bare `deliver: "email"` will fail because there's no `EMAIL_HOME_CHANNEL` env var.
- "platform 'email' not configured/enabled" = the email platform needs `platforms.email.enabled: true` in config.yaml AND `platforms.email.address: mx.indigo.karasu@gmail.com`
- Cron job status "ok" only means the process started, NOT that the email was sent. Verify end-to-end by running `_send_email()` manually.
- If `google_api.py` fails with `invalid_grant`, the `/root/.hermes/google_token.json` has been overwritten with the user-only profile. Fix: `cp /root/.hermes-indigo/google_token.json /root/.hermes/google_token.json`
- **`google_api.py gmail get` HEADER BUG**: The `gmail get` command returns empty `to` and `subject` fields even when the raw Gmail API headers are correct. To verify actual headers, use the raw Gmail API directly via Python (see Verification section below).
- **Token identity check**: Both `/root/.hermes/google_token.json` and `/root/.hermes-indigo/google_token.json` currently authenticate as `mx.indigo.karasu@gmail.com`. There is no separate Jared token. This means you CANNOT programmatically verify delivery to Jared's inbox — you must ask the user to confirm receipt.
- **SENT ≠ delivered**: An email appearing in Indigo's SENT folder only proves it was emitted, not that it arrived in Jared's inbox. Never assume delivery based on SENT label alone.
- **`_send_email()` HERMES_HOME**: The Gmail API fallback in `send_message_tool.py` now sets `HERMES_HOME=/root/.hermes-indigo` when the agent token exists, ensuring the correct OAuth token is used for sending. If you see "Hermes Briefing" subjects instead of proper briefing titles, this fix is working (subject is extracted from the "Cronjob Response: <name>" header in the message content).
- **Vesper reporting invalid_grant**: The Vesper cron run itself may report `invalid_grant` for Jared's account during its briefing generation. This is separate from the delivery path — the briefing content gets generated, and then the cron delivery system sends it via `_send_email()`. An `invalid_grant` in the briefing body does NOT mean the email wasn't sent.

## Verification (checking if an email was actually delivered)
1. **Do NOT trust `google_api.py gmail get`** for header verification — it has a parsing bug that shows empty fields.
2. Use the raw Gmail API directly to check headers:
```python
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import json

token_path = '/root/.hermes-indigo/google_token.json'
with open(token_path) as f:
    t = json.load(f)
creds = Credentials(token=t.get('token'), refresh_token=t.get('refresh_token'),
    token_uri='https://oauth2.googleapis.com/token',
    client_id=t['client_id'], client_secret=t['client_secret'], scopes=t.get('scopes', []))
if not creds.valid and creds.refresh_token:
    creds.refresh(Request())
service = build('gmail', 'v1', credentials=creds)
msg = service.users().messages().get(userId='me', id='MESSAGE_ID',
    format='metadata', metadataHeaders=['To', 'Subject', 'From']).execute()
for h in msg.get('payload', {}).get('headers', []):
    print(f"{h['name']}: {h['value']}")
```
3. To confirm delivery reached Jared's inbox, **ask Jared to check** — there is no API access to his account currently.

## Critical Files
- `/root/.hermes/google_token.json` — currently a COPY of `/root/.hermes-indigo/google_token.json` (both authenticate as mx.indigo.karasu@gmail.com)
- `/root/.hermes/hermes-agent/tools/send_message_tool.py` — `_send_email()` function has Gmail API fallback (line ~697). **Fixed**: now sets `HERMES_HOME=/root/.hermes-indigo` for the agent token and extracts subject from "Cronjob Response:" header instead of hardcoding "Hermes Briefing".
- `/root/.hermes/hermes-agent/gateway/platforms/email.py` — gateway adapter, SMTP-only (not used for cron since EMAIL_PASSWORD is commented out)
- `/root/.hermes/hermes-agent/cron/scheduler.py` — `_deliver_result()` resolves delivery target, checks `_KNOWN_DELIVERY_PLATFORMS` and platform enabled status
- `/root/.hermes/skills/productivity/google-workspace/scripts/google_api.py` — CLI tool for Gmail API operations. **Known bug**: `gmail get` displays empty `to`/`subject` headers — use raw Gmail API for header verification.

## Architecture
- Cron delivery uses `send_message_tool._send_email()` which has a Gmail API fallback when `EMAIL_PASSWORD` is not set
- The Gmail API fallback now explicitly sets `HERMES_HOME=/root/.hermes-indigo` to use the agent token (with `gmail.send` scope)
- Subject line is extracted from the "Cronjob Response: <name>" pattern in the message content, falling back to the first line, then to "Hermes Briefing"
- The gateway adapter (`email.py`) is SMTP-only and requires `EMAIL_ADDRESS` + `EMAIL_PASSWORD` + `EMAIL_SMTP_HOST` — not used since EMAIL_PASSWORD is commented out
- The email platform MUST be enabled in config.yaml (`platforms.email.enabled: true`) or cron delivery returns "platform not configured/enabled"
- `EMAIL_PASSWORD` in `.env` is intentionally commented out — SMTP is not used
- **Both token files currently identify as the same account** (`mx.indigo.karasu@gmail.com`). A separate Jared token must be re-authorized if his inbox needs independent access.
- **NEVER** claim email is fixed without actually sending a test email and **getting user confirmation of receipt**.
- If `google_api.py` fails with `invalid_grant`, the `/root/.hermes/google_token.json` has been overwritten with the user-only profile. Copy the working agent token: `cp /root/.hermes-indigo/google_token.json /root/.hermes/google_token.json`