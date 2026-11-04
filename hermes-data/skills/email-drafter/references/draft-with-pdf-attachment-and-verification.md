# Draft with a PDF attachment, then verify it (verified 2026-10-24)

When the mail must carry the source document (utility demand, sanction, statement, scan), the
native `gws_gmail_draft_create` tool has no attachment parameter — build the MIME yourself and
post the raw message as a draft. This is the sanctioned path: `tools.gws_auth.build_service`,
never hand-rolled credentials.

```python
import sys, base64
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
from email.message import EmailMessage

svc = build_service('gmail', 'v1', service_name='google-draas')   # resolve first, never guess

msg = EmailMessage()
msg['To'] = 'ndr@draas.com'
msg['Subject'] = '...'
msg.set_content(body)                      # plain text body, UTF-8

with open(att_path, 'rb') as f:
    data = f.read()
msg.add_attachment(data, maintype='application', subtype='pdf',
                   filename='Meaningful_Name.pdf')     # rename scanner timestamps

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
```

## Pitfalls

- **Never put a placeholder in Cc/Bcc.** `msg['Cc'] = '[Accounts - name/email]'` returns
  `HttpError 400 … "Invalid Cc header"` and the whole draft fails. Placeholders belong in the
  **body** only; carry the intended CC in the chat reply instead ("say the word and I'll add
  accounts").
- **Verify the attachment by walking nested parts.** A `format='metadata'` fetch of the
  top-level `payload.parts` returns `[]` for attachments, because the structure is
  `multipart/mixed → [text/plain, application/pdf]`. Use
  `svc.users().drafts().get(userId='me', id=draft_id, format='full')` and recurse:

```python
def walk(p, depth=0):
    print(' ' * depth, p.get('mimeType'), p.get('filename') or '')
    for c in p.get('parts', []) or []:
        walk(c, depth + 2)
```

- **`From` will show the mailbox's primary sendAs**, not the vault label. For Anbu the vault
  service resolves as `pm2.blr@draas.com` while outgoing mail shows
  `Anbarasan M <anbarasan@draas.com>`. That is correct — do not "fix" it.
- Confirm which mailbox you actually wrote into with
  `svc.users().getProfile(userId='me').execute()['emailAddress']` before telling the user
  where the draft is.
- Drafts only. There is no send path here and none should be added.
