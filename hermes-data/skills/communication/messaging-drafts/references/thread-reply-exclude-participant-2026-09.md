# Threaded reply while EXCLUDING a participant (NDR pattern, 2026-09)

Worked example: NDR dictated "send a WhatsApp message to Sinchana AND an email —
make it part of the same thread related to the Palya design, but do NOT include
Arvind architect — this is to Sinchana only... if I've already sent her an earlier
assignment email for this land, use it; otherwise create a fresh email."

Context: after delivering Rainbow Mayfair RERA research (site plan / approval /
section+elevation / brochure), NDR wanted Sinchana to study those row-villa blocks
and adapt them to the Palya land (Sy No 8/1, 2A 6G, Ashokji land, ~2A 6G).

## 1. Find ALL candidate threads, then pick by PARTICIPANT SET

`gws_skill_bridge.call('gmail_search', service_name='google-draas', query='<land name>', max=15)`
returned TWO related threads with the same subject topic:

- "Palya Row Villas (Sy No 8/1, 2A 6G) - Premium Concept Proposal Brief"
  (thread `1a038b7c2f50401d`) — participants INCLUDE Arvind Jain
  (arch_arvind2000@yahoo.co.in) → **EXCLUDE**
- "Palya Land (Sy No 8/1, 2A 6G) — Row Villa Master Plan & FAR Block Layout —
  Priority" (thread `1a00ea1690cf71d0`) — participants NDR ↔ Sinchana only → **USE**

Rule: when the user says "same thread" AND "don't include X", the thread to use is
the one whose participant set matches the requested recipients exactly. Never
guess from the subject alone — verify with
`gws_skill_bridge.call('gmail_thread_get', ..., thread_id=...)` and read the
`from`/`to` of every message before drafting. The subject can look identical
across two threads.

## 2. Threading a reply with draft_reply_create

`draft_reply_create` takes **`message_id`** (NOT thread_id) — the id of the
message you are replying to. Pass the LAST message in the chosen thread FROM the
intended recipient (here Sinchana's reply `1a00ef57aaedb280`). The bridge:

- reads that message's Message-ID header,
- sets In-Reply-To / References (correct threading),
- prefixes `Re:` to the subject if missing,
- defaults `To:` to the original message's From (so no cc needed),
- attaches the draft to `original.threadId` (stays in the same thread).

Call shape (all args are plain kwargs to `tools.gws_skill_bridge.call`):

```python
out = bridge.call('draft_reply_create', service_name='google-draas',
                  message_id='1a00ef57aaedb280',
                  to='Sinchana Gowda <sgowda@draas.com>',   # explicit is safer than relying on From
                  body=body)
```

Returns JSON with `draft_id`, `message_id`, `threadId`. Report the draft to NDR
with the drafts link: `https://mail.google.com/mail/u/0/#drafts?compose=<draft_id>`.
NEVER send — draft only (hard rule).

### HTML + missing-Message-ID nuances (verified 2026-09-09, Roshini JD-forward reply)

- **HTML bodies:** pass `html=True` plus the full HTML string as `body` — the
  bridge builds `MIMEText(body, "html")`. Write the HTML to a temp file
  (`write_file /tmp/<name>.html`) and read it in the script instead of inlining
  in a heredoc — the `&` backgrounding guard rejects heredocs containing HTML
  entities (`&mdash;`, `&nbsp;`).
- **Recipient's last message has NO Message-ID** (Gmail iPhone forwards often
  lack one — checked via `metadataHeaders=["Message-ID"]` returning empty): the
  bridge simply skips In-Reply-To/References, but still attaches the draft to the
  thread because `drafts().create` carries `threadId`. Gmail threads it
  correctly and the subject gets a clean `Re:` prefix. Acceptable — do NOT
  hand-roll raw MIME just to force threading headers.
- **Verify after create:** `draft_get` (or raw `drafts().get`) must show exactly
  the intended recipient in `To`, **no `Cc`** (participant exclusion worked),
  `mimeType == 'text/html'`, and the original `threadId` unchanged.
- **Cron/background sessions:** the bridge terminal may resolve to the wrong
  user's mailbox (stale `HERMES_SESSION_USER_ID`). Profile-check with
  `email-drafter` → `scripts/check_gmail_mailbox.py` / run with
  `HERMES_SESSION_USER_ID=<owner id>` prefix — see
  `email-drafter` → `references/cron-session-gws-identity.md`.

## 3. Grant Drive folder access BEFORE sharing the link

The recipient cannot open a Drive folder link without permission. Grant a viewer
permission on the folder id first (inherits to all children — plans + brochure):

```python
from tools.gws_auth import build_service
drive = build_service('drive', 'v3', service_name='google-draas')
perm = drive.permissions().create(
    fileId=FOLDER_ID,
    body={'type': 'user', 'role': 'reader', 'emailAddress': 'sgowda@draas.com'},
    sendNotificationEmail=False).execute()
```

Pitfall (2026-09): the skill-bridge `drive_share` helper can 400 with "The
permission type field is required" — it omits the required `type`. The direct
`permissions().create` with an explicit `type` field is the reliable path. Use
`role='reader'` unless NDR says the recipient must edit.

## 4. WhatsApp leg — same message, exact 12-digit phone

Generate the WhatsApp deep link with `tools.whatsapp_link_tool`
(terminal fallback when the tool is not in the live toolset — see
messaging-links SKILL.md). Before delivering, verify `phone=` digits are EXACTLY
12 for Indian numbers (91 + 10 digits): Sinchana `+919008720170` →
`919008720170` (12 ✓). No raw `&` in the message text. Both legs carry the same
Drive folder link.

## 5. Voice corrections observed in this dictation

- "Paliya" → **Palya** (land, Sy No 8/1 — multiple existing email threads use Palya)
- "Shinchana" → **Sinchana** (Sinchana L Gowda TruBld, sgowda@draas.com, +919008720170)
- "sale thread" → **same thread** — NDR dictates "the same thread of email"
- "flow plans / flow plan" → floor plans (already in voice-corrections-real-estate.md)

Apply silently; never echo the voice error back.