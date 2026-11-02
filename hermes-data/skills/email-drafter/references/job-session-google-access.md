# Job/api-server sessions: getting a Google identity, and drafting with attachments

Two things that repeatedly block a background/job run of this workflow. Both verified 27-Oct-2026
while forwarding a succession-certificate document set.

## 1. There is no session user — set it in the SHELL, not in the script

A job / api_server session has no gateway session context, so `gws_resolve_account`,
`contact_resolver`, and any in-process `build_service(...)` fail closed with
*"No session user context (HERMES_SESSION_USER_ID not set)"*.

What works — write the script to a file and run it through `terminal` with the user prefixed as an
environment variable:

```bash
HERMES_SESSION_USER_ID=ndr python3 /tmp/work/s1.py   # ndr for ndr@draas.com
```

```python
import sys; sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")
```

What does NOT work: calling `gateway.session_context.set_session_vars(user_id="ndr")` at the top of an
`execute_code` script and then `build_service(...)` — `gws_auth` resolves the identity through
`get_gws_identity_env()`, which did not see the in-process ContextVar and raised the same error. Keep
`execute_code` for pure processing; run the Google calls from a file under `terminal` with the env
prefix.

Only the accounts NDR owns are reachable this way (google-draas / google-ahfl / google-gmail).
`contact_resolver` and `entity_resolver` are tool-layer only and stay unavailable in a job — if you
need the contacts sheet, read it directly with `build_service("sheets","v4", ...)`:
spreadsheet `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab `NDR DRAAS Google contacts.csv`
(78 columns — Phones start at col AB, Addresses at col AN; read `A1:BZ1` for the header before
writing by index). `Gmail`-side truth (headers, attachments, thread) is always a better address
source than the sheet anyway.

## 2. Forwarding with attachments — native tools take none

`gws_gmail_draft_create` / `gws_gmail_reply_draft` have no attachment parameter. Build the MIME
yourself and create the draft raw (never send):

```python
from email.message import EmailMessage
import base64
msg = EmailMessage()
msg["To"] = "A <a@x.com>, B <b@y.com>"
msg["Cc"] = "C <c@z.com>"
msg["From"] = "Nishant Ranka <ndr@draas.com>"
msg["Subject"] = "Fwd: <original subject>"
msg.set_content(body)                      # plain text body
for fn, maintype, subtype in files:
    with open(path, "rb") as f:
        msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=fn)
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

MIME types that matter: `.docx` →
`application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `.pdf` → `application/pdf`.

**Verify by re-reading the draft, not by trusting the create response:**

```python
full = svc.users().drafts().get(userId="me", id=d["id"], format="full").execute()
# walk payload.parts recursively, print (filename, body.size) for every part with a filename,
# and print the To / Cc / Subject headers back
```

Report the draft_id, the message_id and the attachment list with byte sizes. "Draft created" without
that read-back is not proof the files are on it.

## 3. Report shape for a job that ends in drafts

Both legs of a two-part instruction (a WhatsApp leg and an email leg) finish as drafts. The job
summary must say, per leg: the exact draft id(s), recipients/subject, what is attached, the expiry
window where relevant, and one line that **nothing has been sent**. Put file paths of the source
documents in the summary so a lapsed draft can be re-staged in one call.
