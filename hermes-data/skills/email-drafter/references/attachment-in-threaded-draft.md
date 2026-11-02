# Threaded reply draft WITH an attachment (image / PDF) — raw MIME, no native tool

Verified 2026-10-28 (Dr Vasisth Das → Ankit Jain, "Dad's reports, CD contents",
thread `1a15b34c6d7aea3d`).

## The gap: neither native draft tool can attach anything

`gws_gmail_draft_create` and `gws_gmail_reply_draft` take only
`to / cc / bcc / subject / body / html / service_name`. **There is no
`attachments` parameter**, and the bridge's `draft_reply_create` is text-only
too. So the moment the user says *"send him the snapshot / the report / the
PDF"*, the native tools cannot do the job and the answer — until it is written
down here — is raw MIME via the sanctioned client:

```python
import base64, sys
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service

svc = build_service("gmail", "v1", service_name="google-draas")
m = svc.users().messages().get(userId="me", id=MSG_ID, format="metadata").execute()
h  = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}

new = EmailMessage()
new["From"]    = "Nishant Ranka <ndr@draas.com>"     # must be set explicitly
new["To"]      = "Ankit Jain <drankitjn@gmail.com>"  # From the thread HEADERS, never the sheet
new["Subject"] = h["subject"]                        # borrow the parent's exact subject
new["Date"]    = formatdate(localtime=True)
new["Message-ID"] = make_msgid(domain="draas.com")
new["In-Reply-To"] = h["message-id"]
new["References"]  = h["message-id"]                 # parent's References + parent id if deeper
new.set_content(PLAIN)
new.add_alternative(HTML, subtype="html")
new.add_attachment(open(PNG, "rb").read(), maintype="image", subtype="png",
                   filename="Dr Vasisth Das - WhatsApp reply 28 Oct 2026.png")

raw = base64.urlsafe_b64encode(new.as_bytes()).decode()
d = svc.users().drafts().create(userId="me",
      body={"message": {"raw": raw, "threadId": THREAD_ID}}).execute()
print("DRAFT_ID:", d["id"])
```

Run it with `env -u PYTHONPATH /opt/hermes/.venv/bin/python3 <script>` (the
venv's interpreter, not `uv run`, and the PYTHONPATH strip keeps
`tools.gws_auth` importable).

Notes that cost time when rediscovered:

- `threadId` **must** be passed on `drafts().create`, otherwise the draft lands
  as a detached new mail with the same subject.
- Set `From` yourself — `EmailMessage` does not inherit it, and a missing
  `From` produces a draft the API will not create.
- Attaching bytes (`add_attachment(data, ...)`) beats `MIMEApplication`:
  `maintype`/`subtype` plus `filename` is all Gmail needs and it avoids a
  double base64 hop.
- Name the file for the human who receives it (`Dr Vasisth Das - WhatsApp
  reply 28 Oct 2026.png`), not `image1.png`.

## Always read the draft back — the four assertions

```python
dv  = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
cur = message_from_bytes(base64.urlsafe_b64decode(dv["message"]["raw"]))
print(cur["To"], cur["Subject"], cur["In-Reply-To"], dv["message"]["threadId"])
print(svc.users().messages().get(userId="me", id=dv["message"]["id"],
        format="metadata").execute()["labelIds"])   # expect ['DRAFT']
print([p.get_filename() for p in cur.walk()
       if (p.get("Content-Disposition") or "").startswith("attachment")])
```

1. `To` names the counterparty (not NDR — see the reply-anchor trap in
   `threaded-reply-and-draft-verification.md`).
2. `threadId` equals the parent thread.
3. `labelIds == ['DRAFT']` — proves it is a draft and that nothing was sent.
4. the attachment filename is present.

Then confirm there is exactly ONE new draft (`gws_gmail_draft_list`, compare
against the pre-existing ids) — a script accidentally run twice leaves a
duplicate the user may send by mistake.

## What goes in the body when the attachment is a chat screenshot

The recipient cannot verify a rendering, so the mail must carry the meaning in
text, with the source quoted verbatim, and the image as corroboration:

> This is the doctor's advice: **sooner rather than later**. …
> His words: "Sooner is better….how soon ? this decision needs to be taken in
> conjunction with the treating cardiologist and CT surgeon- but don't delay
> indefinitely."
> So exactly how soon only the treating cardiologist and CT surgeon can say —
> but his view is sooner rather than later.
> I've attached a snapshot of his reply on WhatsApp so you have it in his own
> words.

Disclose the rendering caveat **to NDR in the chat**, not in the outgoing mail:
the image is a faithful rendering from the bridge's message record, not a
native device screenshot, and offer to swap it for a real phone capture. Quote
the attachment filename and the draft id back to him so he can find it.

## Do not answer questions the source did not answer

The counterparty's earlier mail carried extra questions (proactive vs
symptomatic, surgical risk) that the doctor's reply never touched. Leave them
out and say so in the chat summary — inventing or extrapolating a clinical
answer into a draft is the worst failure mode in this class.

## Related

When the attachment is a specialist's second opinion on a relative, the
health-domain discipline (prove which channel the opinion came on, quote it
verbatim, keep unanswered family questions out, name the relay package) lives in
`family-health-operations/references/specialist-second-opinion-relay.md`.
