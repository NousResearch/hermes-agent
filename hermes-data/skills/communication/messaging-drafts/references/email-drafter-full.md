---
name: email-drafter
description: |
  Drafts new emails, threaded replies, and saved drafts via Gmail API (Python stdlib).
  Auth: per-user OAuth token via gws_auth.py (HERMES_SESSION_USER_ID).
  If token missing, generate auth URL and send to user. NEVER uses SA DWD for Gmail.
  Trigger: "email [name]", "reply to [name]'s email", "draft an email to [name]", "send [name] an email", "save an email draft", "follow-up email"
category: communication
version: 2.1.0
author: ndr@draas.com
---

# Email Drafter

## Critical notes

- `google_workspace_manager` does NOT exist — do not invoke it as a tool or CLI
- All Gmail operations: direct Gmail API via Python stdlib (`urllib.request` + `json`)
- `POST /gmail/v1/users/me/messages.send` returns HTTP 404 on this account — always use the draft workflow instead
- `POST /gmail/v1/users/me/drafts` **DOES accept `threadId`** — pass it INSIDE the message object: `{"message": {"raw": raw, "threadId": THREAD}}`. At the draft top level the field is ignored and Gmail mints a NEW thread even when `In-Reply-To`/`References` look correct. (An earlier note here claimed `threadId` caused HTTP 400 — that was wrong, re-verified 2026-10-14.) Note the split: `threadId` controls where the draft sits in the SENDER's Drafts list; the RECIPIENT's threading follows the MIME `In-Reply-To`/`References` headers, so set both.
- For forwarded emails where you need the inner body, use `format=raw` (not `format=full`)

---

## Execution environments — which tool to use

**Always use `terminal` with python3 heredoc for Gmail operations — not `execute_code`.**
`HERMES_SESSION_USER_ID` and other HERMES_ vars are available in terminal.

**Correct pattern — always use terminal:**
```bash
cd /opt/hermes && env -u PYTHONPATH /opt/hermes/.venv/bin/python3 - << 'PYEOF'
import sys; sys.path.insert(0, '/opt/hermes')   # REQUIRED
import os
from tools.gws_auth import load_credentials, get_auth_url, has_token
from google.auth.transport import requests as google_requests

# NOTE: the module root is /opt/hermes, NOT /data/hermes -- `sys.path.insert(0, '/opt/hermes')`
# NOTE: has_token() / load_credentials() take ONLY service_name. Identity comes from the
#       session, so do NOT pass a telegram id (that raises "Invalid or missing service name").
SERVICE_NAME = "google-draas"  # resolve via gws_resolve_account -- NEVER guess
if not has_token(service_name=SERVICE_NAME):
    print("GWS_AUTH_NEEDED")   # then call the send_oauth_url tool
    exit(1)

creds = load_credentials(SERVICE_NAME)
creds.refresh(google_requests.Request())
access_token = creds.token
# Now use access_token with urllib ...
PYEOF
```

**If you see `GWS_AUTH_NEEDED:`** — send the printed URL to the user via Telegram and ask them to click it to authorize. Once they complete auth, retry the operation.


---

## Gmail API — stdlib recipe

**Auth: per-user OAuth token via `gws_auth.py`.** Never use SA DWD for Gmail.

```python
import os
from tools.gws_auth import load_credentials, get_auth_url, has_token
from google.auth.transport import requests as google_requests

# NOTE: the module root is /opt/hermes, NOT /data/hermes -- `sys.path.insert(0, '/opt/hermes')`
# NOTE: has_token() / load_credentials() take ONLY service_name. Identity comes from the
#       session, so do NOT pass a telegram id (that raises "Invalid or missing service name").
SERVICE_NAME = "google-draas"  # resolve via gws_resolve_account -- NEVER guess
if not has_token(service_name=SERVICE_NAME):
    print("GWS_AUTH_NEEDED")   # then call the send_oauth_url tool
    exit(1)

creds = load_credentials(SERVICE_NAME)
creds.refresh(google_requests.Request())
access_token = creds.token
# Use access_token in Authorization: Bearer header with urllib — unchanged from before
```


---

## Common operations

**List/search messages:**
```python
query = urllib.parse.quote('to:@recipientdomain.com subject:keyword after:2026/04/01')
url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=10&q={query}'
req2 = urllib.request.Request(url, headers={'Authorization': f'Bearer {access_token}'})
with urllib.request.urlopen(req2, timeout=15) as resp:
    messages = json.loads(resp.read()).get('messages', [])
```

**Get full message:**
```python
url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}?format=full'
```

**Get full thread (all messages):**
```python
url = f'https://gmail.googleapis.com/gmail/v1/users/me/threads/{thread_id}?format=full'
```

**Trash emails (move to bin):**
```python
trash_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}/trash'
req = urllib.request.Request(trash_url, method='POST',
    headers={'Authorization': f'Bearer {access_token}'})
with urllib.request.urlopen(req, timeout=15) as resp:
    json.loads(resp.read())
```

**List today's emails (top N, with full header detail):**
```python
import re
from datetime import datetime, timezone, timedelta

ist = timezone(timedelta(hours=5, minutes=30))
today_start = datetime(2026, 5, 6, 0, 0, 0, tzinfo=ist).astimezone(timezone.utc)
today_str = today_start.strftime('%Y/%m/%d')

query = f'after:{today_str} in:inbox'
url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults=20&q={urllib.parse.quote(query)}'
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {access_token}'})
with urllib.request.urlopen(req, timeout=15) as resp:
    result = json.loads(resp.read())

for m in result.get('messages', []):
    mid = m['id']
    # Fetch full headers + snippet
    full_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{mid}?format=full'
    req2 = urllib.request.Request(full_url, headers={'Authorization': f'Bearer {access_token}'})
    with urllib.request.urlopen(req2, timeout=15) as resp:
        data = json.loads(resp.read())
    
    headers = {h['name']: h['value'] for h in data['payload']['headers']}
    labels = data.get('labelIds', [])
    snippet = data.get('snippet', '')
    
    # Classify
    if 'CATEGORY_PROMOTIONS' in labels:
        cat = '📢 Promotional'
    elif 'CATEGORY_PERSONAL' in labels:
        cat = '🏦 Transactional'
    elif 'CATEGORY_UPDATES' in labels:
        cat = '📋 Updates'
    elif 'IMPORTANT' in labels:
        cat = '💼 Work'
    else:
        cat = '📧 General'
    
    # Clean snippet HTML entities
    snippet = re.sub(r'&#39;', "'", snippet)
    snippet = re.sub(r'&amp;', '&', snippet)
    
    print(f"Category: {cat}")
    print(f"From: {headers.get('From', '')}")
    print(f"To: {headers.get('To', '')}")
    print(f"Cc: {headers.get('Cc', '')}")
    print(f"Subject: {headers.get('Subject', '')}")
    print(f"Date: {headers.get('Date', '')}")
    print(f"Summary: {snippet}")
```

**Classify email type (promotional vs transactional):**
```python
# Check labelIds for Gmail's native categories
labels = data.get('labelIds', [])
if 'CATEGORY_PROMOTIONS' in labels:
    category = 'promotional'
elif 'CATEGORY_UPDATES' in labels:
    category = 'transactional/updates'
elif 'CATEGORY_PERSONAL' in labels:
    category = 'personal'
# Also check raw email for list-unsubscribe (strong promo signal)
raw_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{msg_id}?format=raw'
# ...fetch raw, decode...
is_promo = 'list-unsubscribe' in raw_email.lower()
```

---

## 1. Trigger Conditions

Activate on: "email [name]", "reply to [name]'s email", "draft an email to [name]", "send [name] an email", "save an email draft", "follow-up email"

**Account:** session user's email from 'User Profile -> Email (Google Workspace)'. NEVER ndr@draas.com unless Nishant is requesting.

---

## 2. Stage 1 — Context Gathering

### For a NEW email
Always use `contact_resolver` — never guess an email address. Confirm the resolved contact with the user before drafting.

**🔴 HARD RULE (NDR, 2026-10-14): "any email must be looked up through contact."**
This applies to **every** address you put on an email, a calendar invite or a Drive
share — not just the To: line. `contact_resolver` alone is **not** sufficient: it reads
the DRAAS contacts **Sheet**, which carries stale and outright phantom addresses.
Cross-check the online Google Contacts (People API) on the relevant account and **name
the source** of each address in your reply.

Verified failure (2026-10-14): a family calendar invite went to the invented
`rivaanranka@gmail.com` instead of Roshini's real `rmurjani@gmail.com`. The Sheet row
was corrupted (three addresses concatenated with `:::` in one cell) and carried two more
addresses she has never had. Her true set is `rnr@draas.com`, `rnr@drahomes.in`,
`rmurjani@gmail.com` — exactly three.

Rules that follow:
- **Never build an address from a name pattern.** Near-anagram pairs are a live hazard —
  Rivaan's real address is `rankarivaan@gmail.com`; `rivaanranka@gmail.com` belongs to nobody.
- **A zero-hit `searchContacts` is not proof of absence.** Try spelling variants
  (Roshni/Roshini), the last name, and the email local-part before concluding.
- **If a wrong address was already used, fix it at the source and say what you reversed.**
  Calendar: `events().patch(body={"attendees": [...complete list...]}, sendUpdates="all")`.
  Then correct memory *and* the governing reference.

Full rule, worked case, lookup recipe and the corrupted-row anatomy →
`references/contact-email-resolution.md`.

**When `contact_resolver` misses, do not guess and do not stop.** Full ladder
(account_email requirement, honourific stripping, Gmail-search fallback, worked
Gauri→Gowri example) in the `contact-phone-lookup` skill →
`references/contact-resolver-fallbacks.md`. Short version:
- strip the honourific ("Gauri mam" → query `Gauri`), never trust a candidate
  that matched on a single initial;
- `gws_gmail_search(query="<name>", service_name="google-draas")` to get the
  canonical spelling + address, then `from:<address>` to learn their *role* from
  their subjects;
- state the resolution in the chat reply and let the user correct it.

**Context recovery before drafting.** A "draft an email to X about this" that
opens a fresh session has no "this" in it. Recover it with
`session_search(query="<name>")` sorted `newest`, then scroll the tail of the
matching session with `session_id` + `around_message_id`. The referent is almost
always the last thing the previous session offered to do — quote it back in your
reply so a wrong guess is caught before the draft is written.

### For a REPLY (existing thread)
1. Search by recipient domain + subject keyword to find the thread
2. Fetch full thread to get `threadId`, participants, and all prior messages
3. Present context before drafting: who, subject, last message date, participants

---

## 3. Stage 2 — Draft

### Work email tone
- No greeting, straight to the point
- Numbered tasks if there are asks
- No boilerplate ("Hope you're well", "Dear [name]") unless asked
- Subject: `[Project/Entity]: one-line description`

### Personal / casual tone (Roshni Ranka / "RO")
- Warmer, no subject prefix
- Plain text is fine

Always show the draft and wait for explicit "send" before doing anything further.

---

## 4. Stage 3 — Draft vs Send

### Creating a draft (new email)
```python
mime_msg = MIMEText(body, 'plain')
mime_msg['to'] = to_addr
mime_msg['cc'] = cc_addr  # optional
mime_msg['subject'] = subject
raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()

draft_payload = {
    "message": {
        "raw": raw,
        "payload": {
            "headers": [
                {"name": "Bcc", "value": bcc_addr}  # BCC goes in headers, not top-level
            ]
        }
    }
}  # NO threadId
url = 'https://gmail.googleapis.com/gmail/v1/users/me/drafts'
data = json.dumps(draft_payload).encode()
req = urllib.request.Request(url, data=data,
    headers={'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=15) as resp:
    result = json.loads(resp.read())
# result['id'] = draft ID
```

**BCC in drafts:** The Gmail API drafts endpoint **accepts `Bcc` in the message payload headers**. Set it as:
```python
"Bcc": "aamirkhan@me.com"  # or any BCC recipient
```
Do NOT pass `bcc` as a top-level field — it belongs in `payload.headers`.

After saving, tell the user the draft is ready at `https://mail.google.com/mail/u/0/#drafts`. They add any additional BCC recipients manually before sending.

### Creating a draft in an existing thread (reply)
```python
mime_msg = MIMEText(body, 'plain')
mime_msg['to'] = to_addr
mime_msg['cc'] = cc_addr
mime_msg['subject'] = subject
# Threading via MIME headers — NOT threadId in payload (causes HTTP 400)
mime_msg['In-Reply-To'] = original_message_id
mime_msg['References'] = original_message_id
raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()

draft_payload = {"message": {"raw": raw}}  # NO threadId
url = 'https://gmail.googleapis.com/gmail/v1/users/me/drafts'
# ...POST
```

### Sending — googleapiclient WORKS, urllib FAILS

**The urllib/stdlib approach** (`POST /gmail/v1/users/me/messages.send`) returns HTTP 404 on this account — confirmed reproducible. Do NOT use it for sending.

**The googleapiclient approach** (`gws_auth.build_service` → `gmail.users().messages().send()`) DOES work for sending. Confirmed working Jun 2026:

```python
from tools.gws_auth import build_service
gmail = build_service('gmail', 'v1')
sent = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
# Returns {'id': '...', 'threadId': '...'} — works!
```

**Decision:**
| Approach | Send Support | Draft Support | When to Use |
|----------|-------------|---------------|-------------|
| `gws_auth.build_service` + googleapiclient | ✅ Works | ✅ Works | Sending HTML emails, sending new messages, any send operation |
| urllib stdlib (`urllib.request` + `json`) | ❌ 404 | ✅ Works | Creating drafts (new or threaded), searching, listing, trashing |

**Rule of thumb:** Use `build_service` + googleapiclient for sending. Use the stdlib approach (which doesn't need the googleapiclient dependency from terminal) for drafts, search, and bulk operations.

### HTML Email with Styled Tables — Confirmed Working (Jun 2026)

Send richly formatted HTML emails with styled tables using `MIMEText(body, 'html')`:

```python
from tools.gws_auth import build_service
from email.mime.text import MIMEText
import base64

gmail = build_service('gmail', 'v1')

html_body = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 680px; margin: 0 auto; padding: 20px;">

<!-- Content with styled tables -->

<table style="width: 100%; border-collapse: collapse;">
  <thead>
    <tr style="background-color: #1a1a2e; color: white;">
      <th style="padding: 10px; border: 1px solid #ddd;">#</th>
      <th style="padding: 10px; border: 1px solid #ddd;">Item</th>
      <th style="padding: 10px; border: 1px solid #ddd;">Yes/No</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f9f9f9;">
      <td style="padding: 10px; border: 1px solid #ddd;">1</td>
      <td style="padding: 10px; border: 1px solid #ddd;">Requirement description</td>
      <td style="padding: 10px; border: 1px solid #ddd;">☐</td>
    </tr>
  </tbody>
</table>

</body>
</html>"""

mime_msg = MIMEText(html_body, 'html')
mime_msg['to'] = 'recipient@example.com'
mime_msg['subject'] = 'Subject Line'
mime_msg['from'] = 'sender@example.com'

raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
sent = gmail.users().messages().send(userId='me', body={'raw': raw}).execute()
```

**Verified patterns:**
- `MIMEText(body, 'html')` renders inline CSS and styled tables correctly in Gmail
- Tables with alternating row backgrounds improve readability for checklists
- Use `☐` (U+2610) for unchecked checkbox symbols in HTML tables
- Keep `max-width: 680px` for good mobile rendering
- Gmail strips `<style>` tags from `<head>` — use inline `style=` attributes on elements instead
- Full HTML doctype + `<html><body>` wrapper works fine in Gmail

### Sending confirmation

After a successful send, always confirm to the user:
> Sent! Subject: [subject] → [recipient]

---

## Supporting files
- `references/contact-email-resolution.md` — **read before addressing ANY email, calendar invite or share.** Why the contacts Sheet is not authoritative, how to cross-check online Google Contacts via People API, the anagram hazard, and the verified 2026-10-14 wrong-address incident (Roshini Ranka).
- `references/zodiac-hotels-liquidation.md` — ZHPL (Zodiac Hotels) entity context, participants, thread details, and decision summary. Populated from session on 2026-05-05.
- `references/bounce-notification-patterns.md` — Mailer-daemon bounce patterns: invalid domain vs address-not-found vs former employee addresses. Includes @findingform.design leaver list and cleanup query.
- `references/resignation-acceptance-workflow.md` — Full step-by-step workflow for drafting legally solid resignation acceptance emails: finding the resignation, locating employment documents, extracting legal terms, cross-checking notice periods, structuring the draft, and saving to Gmail. Includes TruBld case data.
- `references/bali-trip-2026.md` — Bali trip context
- `references/royal-sundaram-kanta-ranka-2025.md` — Insurance claim context
- `references/recipient-and-cc-resolution.md` — **read before any multi-recipient email.** Dictated person → address via the saved display name (the phone label and the Gmail display name are the same string); internal staff addresses come from Gmail headers, not the contacts sheet; the one-person-two-addresses rule; how to split To vs Cc when NDR says *"mark X as well"*; the `drafts().get(format="metadata")` Cc verification; and the *"X must confirm"* framing (a request for confirmation, never a record of agreement).

### Scripts
- `scripts/bulk-trash-bounces.py` — Reusable bulk trash script for mailer-daemon bounce notifications. Handles pagination automatically, resolves TELEGRAM_ID from `HERMES_SESSION_KEY` fallback, safe for daily cron.

- **`GWS_AUTH_NEEDED` output means auth URL was printed to stdout, not an error.** When the auth check runs, if the token is missing, the Python script prints `GWS_AUTH_NEEDED: https://auth.url...` and exits 1. This is NOT a Python traceback — it is an instruction. The correct response is to extract the URL and send it to the user via Telegram, then retry after they authorize. Never try to "debug" the Python script when this pattern appears.
- **Mailer-daemon bounce notifications accumulate fast.** When sending to non-existent recipients (e.g. @findingform.design addresses for people who left), Gmail generates multiple identical bounce emails per failed delivery attempt. These pile up in inbox. Proactively trash them — or set up a daily cron to handle them.
- **NEVER invent the anchor's `Message-ID` for `In-Reply-To`/`References` — always read it off the anchor message.** Verified 2026-10-14: a fabricated `<CAH3aL+...@mail.gmail.com>` looked exactly like a real Gmail id, the create call succeeded and returned a correct `threadId`, and nothing errored — yet recipient-side threading would have broken, because it follows `References`. Fetch it with `messages/<id>?format=metadata&metadataHeaders=Message-ID`. A made-up header is fabricated output; if you cannot read the real id, say so and stop. Full write-up: `references/gmail-draft-threading-traps.md` §3c.
- **Use `format=raw` for forwarded email inner bodies** — `format=full` strips the RFC822 wrapper.
- **`pdftotext` returns empty on scanned/image-based PDFs** — the text extraction will show 0 bytes. Always check file size first (`wc -c`), then try `pdfinfo` to confirm page count. For scanned PDFs, fall back to OCR with `tesseract` or use `pymupdf` (fitz) which handles image layers better. See `pdf-page-extraction` skill for the full workflow.
- **Multi-email drafting: confirm threading strategy per email.** When the user asks for two or more emails about the same topic (e.g. stern email to surveyor + cooperative email to insurance manager), do NOT assume they both go in the same thread or both as new emails. Confirm per-email:
  - "Email A (to surveyor): new separate email? Or reply in existing thread?"
  - "Email B (to manager): reply in existing thread? Or new email?"
  
  Example (Jun 2026): Nishant wanted the Venkatesh email as a separate new email, and the Azim email as a reply in the existing insurance claim thread. If you assume both are new or both are replies, you waste a round trip. Explicitly confirm threading per recipient before drafting.

---

## Cron integration

This skill's Gmail API operations (search, trash, list) are safe to embed in cron prompts for recurring email hygiene tasks:
- Daily bounce notification cleanup: `from:mailer-daemon@googlemail.com subject:"Delivery Status Notification"` → trash
- Weekly promotional digest: summarize top emails by category

---

## Dictated correction lists — read back, then default defensibly

When the user dictates a list of corrections to an email (voice, often 8–12 numbered points,
proper nouns garbled), the failure mode is silently mis-parsing a point and shipping the wrong
instruction to a colleague. Worked example 2026-10-14: a 10-point dictated creative-defect list
where "standee" arrived as "Stanley", "130" as "tier 150", and the first point came through as
*"our back panel has the clear as Uber request... this has to be there"* — readable as either
"correct it" or "keep it", i.e. **opposite** instructions.

Workflow:

1. **Play the list back as a numbered checklist with an explicit verdict per item** — `fix` /
   `drop` / `soften to a recommendation`. Do not start rewriting the draft until the verdicts
   are confirmed; one wrong verdict costs more than one extra round trip.
2. **Pull the genuinely ambiguous items out into their own block** and give BOTH readings with
   the consequence of each ("(a) correct the landmark list, or (b) leave it as is — which?").
   Never bury an ambiguity inside the numbered list, where it reads as already settled.
3. **Restate garbled numbers against a source of truth** ("you said 'tier 150 houses' — the
   RERA registration reads **130**; using 130 unless you meant otherwise").
4. **If the reply is non-committal** ("Okay", "yes", a thumbs-up) and does not actually answer
   the specific questions: implement the **defensible default** — the reading supported by the
   underlying facts — and **flag it prominently with a one-word veto path** ("I wrote this as a
   fix because the panel is factually wrong; say the word and I'll strip it in seconds"). Do
   not stall a deadline-bound deliverable waiting on a clarification the user has already waved
   through, and do not silently pick a reading either.
5. **Keep the dropped items visible in your reply** — say what you removed and why ("G+4
   dropped: you confirmed it's correct"). The user must see that a defect he raised is absent
   by decision, not by oversight.

## When the user says "that's already sorted" — verify the trail first

Before accepting that a precondition is closed, search the mail trail for the **written**
confirmation. 2026-10-14: the user said the Sub-Registrar's attestation route was settled, but
the trail showed NDR's written instruction to confirm it with two SROs (officer name,
designation, date) had never been answered — which turned the ask from "chase the landowner"
into "close the upstream gate first". A two-minute search reframed the entire recommendation.
State plainly what you found and what it implies; don't argue the point, just show the trail.

## Pitfalls

- **`gws_gmail_reply_draft` addresses the reply to the FROM of the target message — which is YOU when the user sent that message themselves.** Replying to an email in the user's own SENT folder produces a draft with `To: <the user>`. Observed Oct 2026: user said "let's reply to that thread" about a mail he had just sent to a colleague, and the draft came out addressed to himself. **Always read the created draft's headers back (`drafts/<id>?format=metadata`) and confirm `To` is the intended recipient before telling the user the draft is ready.**
- **`threadId` in the drafts payload DOES work** (contradicts the older note below). `POST /gmail/v1/users/me/drafts` with `{"message": {"raw": ..., "threadId": "<original thread>"}}` was verified Oct 2026 to return a draft whose `threadId` matches the original thread. Without `threadId`, Gmail silently assigns the draft a NEW thread even when `In-Reply-To`/`References` are correct — the draft then sits in its own conversation in the user's Drafts list. Always pass `threadId` when replying into an existing thread.
- **`DELETE /gmail/v1/users/me/drafts/<id>` returns 204 with an EMPTY body.** Calling `json.loads()` on it raises `JSONDecodeError`. Read the response first and only parse when non-empty, otherwise a successful delete looks like a crash (and a retry then 404s because the draft is already gone).
- **Verify the Cc with `drafts().get(format="metadata")` — `gws_gmail_get` does NOT return a `Cc` key at all** (re-verified 2026-10-30). An earlier note here said "re-read the sent message's headers and check for a `Cc` key"; that check is useless, because the `gws_gmail_get` header dict simply omits Cc whether or not it is set. A draft created via `gws_gmail_draft_create(cc="A <a@x.com>, B <b@x.com>")` was read back with `build_service("gmail","v1", service_name="google-draas").users().drafts().get(userId="me", id=DRAFT_ID, format="metadata")` and the full Cc list was present, correct and complete. So: the tool DOES carry Cc into the draft; what fails is the *verification*, not the Cc. Use the metadata recipe in `references/recipient-and-cc-resolution.md` §5 and print the Cc line, then read it name-by-name against what NDR dictated. Still tell NDR to eyeball the composer before sending (a human may edit the list), but do not tell him the Cc "did not carry over" without the metadata read.
- **`gws_gmail_reply_draft` and Cc:** same rule — the reply draft's Cc must be confirmed from metadata, not from `gws_gmail_get`.
- **BCC in draft creation:** The Gmail API drafts endpoint does NOT accept `bcc` as a top-level field. To add BCC recipients when creating a draft, include `Bcc` in `payload.headers`:
  ```python
  "payload": {"headers": [{"name": "Bcc", "value": "bcc@example.com"}]}
  ```
  Do NOT use `"bcc": "bcc@example.com"` at the top level of the message object — it will be silently ignored. After creating the draft, direct the user to open `https://mail.google.com/mail/u/0/#drafts` and they can add additional BCC recipients manually before sending.
- **Gmail search with multiple conditions may return 0 results** even when matching messages exist. If one query returns 0, split into single-condition queries and examine each independently. E.g., `from:jeet.kumar@housing.com subject:microsite` → 0; but `from:jeet.kumar@housing.com` alone → finds the thread.
- **`build_service()` returns a Resource object, not a service with `.credentials`** — always use `load_credentials()` then `discovery.build(..., credentials=creds)` separately.
- **Bulk operations require pagination** — Gmail API caps at 500 messages per search call regardless of `maxResults`. When trashing or listing large batches, always loop on `nextPageToken` until exhausted. `resultSizeEstimate` is approximate; do not rely on it for exact counts.
- **Cron sessions: `HERMES_SESSION_USER_ID` may be empty** — cron jobs set `HERMES_SESSION_KEY=agent:main:telegram:dm:<telegram_id>` but not `HERMES_SESSION_USER_ID`. If `has_token(TELEGRAM_ID)` fails with an empty string, extract the Telegram ID from `HERMES_SESSION_KEY` (format: `agent:main:telegram:dm:<numeric_id>`) and use it directly as the fallback `TELEGRAM_ID`.
- **`ModuleNotFoundError: No module named 'tools'` from the heredoc is the WRONG INTERPRETER, not a missing dependency** (verified 2026-10-14). Running the recipe as `cd /data/hermes && python3` fails, because the module root is `/opt/hermes` and `PYTHONPATH=$HOME/.local` shadows the venv's packages. Use `cd /opt/hermes && env -u PYTHONPATH /opt/hermes/.venv/bin/python3 - << 'PYEOF'` with `import sys; sys.path.insert(0, '/opt/hermes')` as the first line. Do not go hunting for an install fix.
- **Before drafting an email that asserts the STATE of artifacts, read every artifact it names.** 2026-10-14: NDR asked for links to "this one and two other sheets" plus an ask that a colleague validate which is correct, and the natural draft said "the merger has been done on the Deepseek version" — implying the other two were untouched. Reading all three showed **every copy already had the change applied by its own LLM**, which changed the whole ask from "validate one against two" to "compare all three". Same rule for access claims: `drive_permissions_list` each file rather than assuming the recipient can open it, and state in the draft who already had access. One extra read before drafting beats a wrong premise shipped to a colleague.

## Drafting checklist

Before presenting the draft, confirm:
- [ ] Thread identified and threadId extracted (for replies)
- [ ] Correct recipient(s) and CC
- [ ] Tone matches the recipient (work vs personal)
- [ ] Draft created in Gmail Drafts — NOT sent automatically
- [ ] Draft link shared with user for manual send
