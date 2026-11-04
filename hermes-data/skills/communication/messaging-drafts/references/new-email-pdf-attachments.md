# New email (non-threaded) draft with PDF attachment(s)

Use when the user asks to draft a brand-new email to someone with PDFs attached —
not a reply (threaded variant is in threaded-reply-pdf-attachments.md). Proven
2026-10-07 for the K-RERA signing loop (3 drafts in one run, ndr@draas.com).

## Recipe (MIME via build_service, DRAFT ONLY — never send)

```python
import sys, base64, json
sys.path.insert(0, '/opt/hermes')
from email import policy, message_from_bytes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate, make_msgid
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')
USER = 'me'

def attach(outer, path, disp_name):
    with open(path, 'rb') as f:
        part = MIMEApplication(f.read(), _subtype='pdf')
    part.add_header('Content-Disposition', 'attachment', filename=disp_name)
    outer.attach(part)

outer = MIMEMultipart()
outer['To'] = to
outer['Cc'] = cc
outer['Subject'] = subject
outer['Date'] = formatdate(localtime=True)
outer['Message-ID'] = make_msgid(domain='draas.com')
outer.attach(MIMEText(body, 'plain', 'utf-8'))
attach(outer, pdf_path, disp_name)
raw = base64.urlsafe_b64encode(outer.as_bytes()).decode()
res = gmail.users().drafts().create(userId=USER, body={'message': {'raw': raw}}).execute()
draft_id = res['id']
```

- No In-Reply-To/References — a new draft, so no thread headers.
- Batch pattern: loop over multiple recipients in one script, print all draft_ids,
  then verify each.
- Requires HERMES_SESSION_USER_ID=<session-slug> in the env (e.g.
  HERMES_SESSION_USER_ID=ndr-7449813913) and the session user's own OAuth token
  (google-draas for ndr@draas.com).

## Pitfalls

- **Subject headers: plain ASCII only.** If the user's dictated subject contains
  dashes/emdash, write `-` (or plain words), NOT `&mdash;` or HTML entities — Gmail
  stores them literally and the subject displays as `&mdash;`. If you already created
  a draft with entities, rebuild it and delete the superseded one (Gmail has no
  in-place draft edit).
- **Gmail has no in-place draft edit** for fixes: rebuild with same headers + new
  content, delete old draft, verify only one remains in Drafts.
- **Always read back each draft** (format='raw' required): print To/Cc/Subject,
  attachment names + sizes (len of decoded payload), body head — before telling the
  user it's ready.

## Verification pattern

```python
d = gmail.users().drafts().get(userId='me', id=draft_id, format='raw').execute()
msg = message_from_bytes(base64.urlsafe_b64decode(d['message']['raw']), policy=policy.default)
print(msg['To'], msg['Cc'], msg['Subject'])
for p in msg.walk():
    if p.get_filename():
        print(p.get_filename(), len(p.get_payload(decode=True) or b''))
```

Check: To = intended primary recipient, Cc contains exactly who user named, subject
clean, attachment(s) present with correct names and non-zero sizes.