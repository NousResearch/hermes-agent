# Threaded draft to EXPLICIT recipients, with an attachment

Use when NDR says "send it in the same email chain as X" but the reply must be
addressed to someone other than the last sender — e.g. "in the chain where we
update Dr. Sameer, make sure it goes to Roshini, Charitra, Vigila and Annie".

## Why the native reply path is not enough

`gws_gmail_reply_draft` / `drafts().create` without a raw body derives `To:`
from the last message's `From:`. If the last message in the chain was NDR's own
sent mail, that yields `To: ndr@draas.com`; if it was a colleague, the To/CC
split is wrong. You need the thread *and* hand-set recipients.

There is no native tool that sets both. Build the raw message yourself.

## Recipe (runs inside `execute_code`, not a subprocess)

```python
import sys, base64
sys.path.insert(0, "/opt/hermes")          # required before tools.gws_auth
from tools.gws_auth import build_service
from email.message import EmailMessage

svc   = build_service("gmail", "v1", service_name="google-draas")
THREAD = "1a0c8098a9eba35b"                # from threadId on any message in the chain

t    = svc.users().threads().get(userId="me", id=THREAD, format="metadata").execute()
last = t["messages"][-1]
h    = {x["name"].lower(): x["value"] for x in last["payload"]["headers"]}

msg = EmailMessage()
msg["To"]      = "SAMEER RASTOGI <samdoc_mamc@yahoo.com>"
msg["Cc"]      = "Roshini Ranka <rnr@draas.com>, annie baa <anniekbaa@gmail.com>"
msg["From"]    = "Nishant Ranka <ndr@draas.com>"
msg["Subject"] = h["subject"] if h["subject"].lower().startswith("re:") else "Re: " + h["subject"]
msg["In-Reply-To"] = h["message-id"]
msg["References"]  = h["message-id"]
msg.set_content(body)

msg.add_attachment(open(path, "rb").read(), maintype="image", subtype="jpeg", filename=name)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(
        userId="me", body={"message": {"raw": raw, "threadId": THREAD}}).execute()
```

Key points:
- `format="metadata"` on `threads().get` is enough for headers and is cheap.
- Notes on prior messages in the chain often contain `&gt;`-escaped HTML; the
  `message-id` and `subject` headers are clean.
- `threadId` in the request body is what puts the draft in the chain; setting
  `In-Reply-To`/`References` alone does not always do it.
- Never `.send()` — draft only, always.

## Verify by reading the draft back (mandatory)

```python
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="full").execute()
m = d["message"]
print(m["threadId"], m.get("labelIds"))
h = {x["name"]: x["value"] for x in m["payload"]["headers"]}
for k in ["To", "Cc", "Subject", "In-Reply-To"]: print(k, h.get(k))
# walk payload parts and print mimeType + filename to confirm the attachment landed
```
Confirm: `labelIds == ['DRAFT']`, threadId matches the intended chain, every
intended recipient is present, and the attachment part is there with its size.

## Pre-send checklist for third-party correspondence (medical, legal, official)

A draft that goes to a professional on someone else's behalf carries two extra
risks that must be surfaced to NDR in the completion report, not silently assumed:

1. **Attachment attribution.** A photo forwarded by a family member may be of
   the sender, not the patient. If the caption is first-person ("the state of
   *my* lips for 2 weeks now") while NDR describes it as the patient's, file and
   attach it as instructed but state the ambiguity explicitly in the report so
   he confirms before sending. Getting this wrong sends a stranger's body part
   to a patient's oncologist.
2. **Assertions NDR supplied verbally that are not in the record.** e.g. "her
   doctor told her to stop the drug". Write it as he instructed, then say in the
   report that no written record of the instruction was found and that it reads
   as fact to the recipient.

Neither of these is a reason to hold the draft — it is a reason to flag it.
