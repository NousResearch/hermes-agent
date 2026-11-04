# Gmail Threaded Reply Draft WITH Attachments (verified 2026-10-07, PS session)

The native `gws_gmail_reply_draft` tool creates a threaded draft but does NOT accept
attachments. When the user wants a threaded reply draft with files attached (typical for
channel-partner / outreach follow-ups), build it via the Gmail API directly with
`tools.gws_auth.build_service`.

## Recipe

Run in the /opt/hermes venv as the SESSION user (never assume ndr):

    cd /opt/hermes && HERMES_SESSION_USER_ID=<session-slug> /opt/hermes/.venv/bin/python3 script.py

```python
import base64
from email.message import EmailMessage
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')
drive = build_service('drive', 'v3', service_name='google-draas')

# 0) Confirm identity before reporting where files land
about = drive.about().get(fields='user(emailAddress)').execute()

# 1) Fetch the LAST message's headers for correct threading
meta = gmail.users().messages().get(
    userId='me', id=LAST_MSG_ID,
    format='metadata', metadataHeaders=['Message-ID', 'References', 'Subject', 'From']).execute()
headers = {h['name'].lower(): h['value'] for h in meta['payload']['headers']}
msg_id = headers['message-id']
new_refs = (headers.get('references', '') + ' ' + msg_id).strip()

# 2) Attachments from Drive
# PDF file:                        drive.files().get_media(fileId=...).execute()
# Google Sheet -> PDF attachment:  drive.files().export(fileId=..., mimeType='application/pdf').execute()
# (native gws_drive tools cannot export a Sheet to PDF; use build_service)

# 3) Build MIME message
msg = EmailMessage()
msg['From'] = 'Prakash Singh <psingh@draas.com>'
msg['To'] = TO
msg['Subject'] = SUBJECT          # keep the thread subject exactly (Re: ...)
msg['In-Reply-To'] = msg_id
msg['References'] = new_refs
msg.set_content(body)
for fname, data in payloads:
    msg.add_attachment(data, maintype='application', subtype='pdf', filename=fname)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode() \
    .replace('+', '-').replace('/', '_').replace('=', '')

# 4) Draft in-thread — NEVER send
draft = gmail.users().drafts().create(
    userId='me', body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()

# 5) Verify with the native gws_gmail_draft_list tool — the new draft id must appear.
```

## Pitfalls

- Use `msg.as_bytes()` — NOT `email.generator.Generator(...).flatten(msg)` (TypeError on str).
- Chain `References` from the previous message's header; if empty, In-Reply-To alone is fine.
- Match the thread's exact subject with `Re:` prefix so Gmail keeps it in-thread even if
  a client ignores threadId.
- Draft-only: never call `messages().send()`.
- The 3-attachment pattern (investment guide PDF 6.6 MB + empanelment form PDF + Sheet
  exported to PDF) fits fine in a draft; no size issues seen.