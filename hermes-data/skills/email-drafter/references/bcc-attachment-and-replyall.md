# Raw-MIME drafts with Bcc + image attachment + reply-all (verified 2026-09-29)

Three consecutive email jobs (reply-all on the Ranka Udaya GFC thread, threaded chase
on the gift-deed thread, and a NEW email to two directors with a photo attached and
Roshini Bcc'd) all used the same raw build. Everything below was verified end-to-end:
drafts().get returned labels ['DRAFT'], correct threadId, headers, attachment list, and
the draft showed up in drafts().list.

## Core build (new email, with Bcc + image attachment)

```python
import sys, base64, os, email, email.message
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
svc = build_service('gmail', 'v1', service_name='google-draas')
assert svc.users().getProfile(userId='me').execute()['emailAddress'] == 'ndr@draas.com'

img_bytes = open(img_path, 'rb').read()            # photo the user pasted
msg = email.message.EmailMessage()
msg['From'] = "Nishant Ranka <ndr@draas.com>"
msg['To'] = "Dharmesh Ranka <ddr@draas.com>, Manish Ranka <mdr@draas.com>"
msg['Bcc'] = "Roshini Ranka <rnr@draas.com>"       # Bcc is a normal header on EmailMessage
msg['Subject'] = "..."
msg.set_content(body)
msg.add_attachment(img_bytes, maintype='image', subtype='jpeg',
                   filename='Unit_2401_Broken_Tiles.jpg')   # BEFORE encoding
raw_out = base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii')
draft = svc.users().drafts().create(userId='me', body={'message': {'raw': raw_out}}).execute()
```

## Threaded reply / reply-all

1. `threads().get(id=THREAD, format='metadata')` → last message id.
2. `messages().get(id=anchor, format='raw')` → `email.message_from_bytes(...)` the MIME.
3. `In-Reply-To` = anchor Message-ID; `References` = all `<...>` ids from anchor's
   References + anchor's own ID (dedupe).
4. Reply-ALL recipients: take To + Cc VERBATIM from the counterparty's latest message
   headers, minus NDR himself — never reconstruct from memory.
5. `drafts().create(body={'message': {'raw': raw, 'threadId': THREAD}})` pins it to the chain.

## Verification before handing over (always)

- `drafts().get(id=..., format='full')` → labelIds contains 'DRAFT'; threadId matches;
  From/To/Cc/Bcc/Subject correct; walk payload parts and print every attachment
  filename + mimeType + size.
- `drafts().list(...)` and confirm the id is in the list.
- Report it as a DRAFT only — never "sent" (sending is the user's action).

## Pitfall

- Do not hand-type recipient emails; resolve colleagues via contact_resolver first
  (e.g. Sarthak Sharma → admin3.blr@draas.com, Manish → mdr@draas.com, Dharmesh →
  ddr@draas.com * Work — but NDR's own recent threads sometimes use the personal
  gmail; flag the choice when the two differ).