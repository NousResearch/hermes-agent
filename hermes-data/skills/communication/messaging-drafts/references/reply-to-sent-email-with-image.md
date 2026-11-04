# Reply to a sent email (your own outbound) with an embedded image

When Prakash or the user says *"reply to the emails I just sent to channel partners, using this render as the hero image"* — they mean **reply to each sent email** such that the reply goes **back to the original recipients** (not back to the user themselves). The image is embedded in the HTML body via a public Google Drive URL (not CID, not an attachment).

## When this applies

- The original email was **sent by the user** (`from: psingh@draas.com` → `to: partner@email.com`).
- The user now wants a **follow-up reply** in each thread, with the same content re-styled plus a hero image.
- The image lives on Drive and is set to `anyoneWithLink` reader permission.

## Why the native `gws_gmail_reply_draft` tool won't work here

`gws_gmail_reply_draft` replies to the **original sender** of a message. For a *sent* email, the "original sender" is the user themselves — so using it creates a draft addressed back to `psingh@draas.com`, not the channel partner. The correct recipient set comes from the **original `To` header**, not `From`.

## Recipe (Gmail API, hand-built MIME, worked 2026-10-03 for 10 drafts)

```python
import sys, base64
sys.path.insert(0, '/opt/hermes/tools')
from gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')
USER = 'me'

# 1. Get the sent message's headers
orig = gmail.users().messages().get(userId=USER, id='MESSAGE_ID', format='metadata').execute()
headers = orig['payload']['headers']

def hdr(name):
    for h in headers:
        if h['name'] == name: return h['value']
    return ''

thread_id   = orig['threadId']
msg_id_hdr  = hdr('Message-ID')
refs_hdr    = hdr('References')
orig_to     = hdr('To')        # ← THIS is the channel partner's address
orig_cc     = hdr('Cc')
subject     = hdr('Subject')
subject     = subject if subject.startswith('Re:') else 'Re: ' + subject

# Build References
new_refs = f"{refs_hdr}\r\n\t{msg_id_hdr}" if refs_hdr else msg_id_hdr

# 2. Build the raw MIME — To is orig_To, not the sender!
#    Image is embedded via public Drive URL in an <img> tag.
IMG_URL = "https://drive.google.com/uc?id=FILE_ID"   # must be anyoneWithLink

raw_parts = [
    'From: "Prakash Singh" <psingh@draas.com>',
    f'To: {orig_to}',
]
if orig_cc:
    raw_parts.append(f'Cc: {orig_cc}')
raw_parts += [
    f'Subject: {subject}',
    f'In-Reply-To: {msg_id_hdr}',
    f'References: {new_refs}',
    'MIME-Version: 1.0',
    'Content-Type: text/html; charset="utf-8"',
    'Content-Transfer-Encoding: quoted-printable',
    '',
    f'<img src="{IMG_URL}" width="100%"> ... body ...',
]

raw_email = '\r\n'.join(raw_parts)
encoded_raw = base64.urlsafe_b64encode(raw_email.encode('utf-8')).decode('ascii')

# 3. Create draft with threadId to keep it in the conversation
draft_body = {'message': {'raw': encoded_raw, 'threadId': thread_id}}
result = gmail.users().drafts().create(userId=USER, body=draft_body).execute()
```

## Verification

```python
d = gmail.users().drafts().get(userId='me', id=DRAFT_ID, format='metadata').execute()
m = d['message']
hh = {h['name']: h['value'] for h in m['payload']['headers']}
# Check:
#   hh['To']         → contains the channel partner, NOT psingh@draas.com
#   m['threadId']    → equals the original thread ID
#   hh['In-Reply-To']→ equals the original Message-ID
#   hh['Subject']    → starts with 'Re: ' + original subject
```

## Making an image publicly accessible on Drive

The Gmail client fetches the image from Google's servers when the recipient opens the email. The image must have `anyoneWithLink` reader permission:

```python
drive = build_service('drive', 'v3', service_name='google-draas')
perm = {'type': 'anyone', 'role': 'reader'}
drive.permissions().create(fileId='FILE_ID', body=perm).execute()
```

Then use `https://drive.google.com/uc?id=FILE_ID` as the image URL in the `<img>` tag.

## Pitfalls

- **`To: orig_From` is the most common failure.** When replying to a SENT email, setting `To: {hdr('From')}` addresses the draft back to the user themselves. The correct logic: `To: {hdr('To')}` — the original recipients.
- **Inline `text/html` is simpler than CID for this pattern.** Since we only need one image (not a multi-image gallery with PDF attachments), a single `<img src="https://...">` inside `Content-Type: text/html` avoids the MIMEMultipart/related assembly entirely. The image is fetched by the mail client at open time. Drawback: the image doesn't appear offline and may be blocked by clients that don't auto-load remote content (Gmail does for Drive URLs from known senders).
- **Threading must be explicit.** `threadId` in the draft body and `In-Reply-To`/`References` in the MIME headers. Either without the other will create a new conversation or mis-thread.
- **If `format='full'` returns no To header**, switch to `format='raw'`, base64-decode, and read the raw MIME lines. Gmail's API sometimes omits headers in structured output.
- **Drive URL pattern.** Use `uc?id=FILE_ID` (not `open?id=...` or the web-ui sharing link). The `uc` parameter forces direct content delivery suitable for `<img>` embedding.
- **Permission check.** Adding `anyone` reader permission may already exist from previous sharing. The Drive API returns the existing permission ID if it's already set — no error.