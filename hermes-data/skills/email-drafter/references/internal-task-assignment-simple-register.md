# Internal task-assignment email — NDR's "keep it simple" register

Applies when NDR says some version of: *email <colleague> about <document/notice>, tell them
what it is, attach the file, ask them to get it done, and I'm marking X to coordinate.*

## The rule (learned the hard way, 29-Oct-2026)

On the Arya Developers IT refund (AY 2026-27, intimation u/s 143(1)) the agent drafted TWO
detailed emails — notice analysis, figures table, password, portal steps, escalation paths,
Rs 4 non-tie flag, 8 numbered asks. **NDR deleted both without sending**, and re-issued:

> "We want a simple email to Eshwari ... Just say, 'Please find out what is to be done to get
> the refund.' ... Very simple English. That's all. Don't worry about giving him the task
> number or any other number; nothing needs to be given."

So the correct shape for an internal task-assignment mail is **3–4 short sentences**:

1. What it is, in one line — including the thing that actually matters ("It is not a demand.
   It is a refund of Rs 6,660").
2. The attachment, one line — and only if true, that the password was removed so it opens.
3. The single ask, near-verbatim from NDR — "Please find out what is to be done to get the refund."
4. One line naming the coordinator — "Sarthak, please coordinate with Eshwari and make sure
   this is done. I am assigning it to you on my task tracker."
5. "Thanks, Nishant".

Nothing else. Specifically **do not include**:
- task/tracker numbers (P-653), refund sequence numbers, Ack numbers, DIN, PAN
- tables of figures, bullet lists of steps, portal navigation paths
- the analysis, the caveats, the "one honest flag" — that belongs in the tracker row or a
  separate note, NOT in the mail
- deadlines NDR did not state

The full analysis is not wasted — it goes in the tracker row's note and in the completion
report to NDR. The *email* stays at the level of "here is the paper, please handle it".

Register: plain Indian business English, no markdown, no headings, no bold. NDR's own words
for the ask are better than a paraphrased "action item".

## Attachments: native tools cannot do it, raw MIME can

`gws_gmail_draft_create` / `gws_dwd_gmail_draft_create` take no attachment argument, and
`tools.gws_skill_bridge.call("draft_create", ...)` builds a bare `MIMEText` — no attachment
support either. The working path (still draft-only, never send):

```python
import sys, base64; sys.path.insert(0, '/opt/hermes')
from email.message import EmailMessage
from tools.gws_auth import build_service

msg = EmailMessage()
msg['To'] = to; msg['Cc'] = cc; msg['Subject'] = subject
msg.set_content(body)                       # plain text — keep it plain for these mails
with open(pdf, 'rb') as f: data = f.read()
msg.add_attachment(data, maintype='application', subtype='pdf', filename=fname)
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
build_service('gmail','v1',service_name='google-draas') \
    .users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
```

Then **read the draft back** and verify the attachment, not just the headers: fetch the draft
with `format='full'`, walk the payload parts for `filename`, pull the `attachmentId` via
`users().messages().attachments().get(...)`, base64url-decode and compare sha256 + byte count
against the local file. Also confirm the PDF really is unlocked first
(`pdfinfo` → `Encrypted: no`, or `qpdf --show-encryption` → "File is not encrypted") and
re-read the operative figure out of the PDF text with `pdftotext -layout` rather than trusting
the previous session's summary.

## Verify the Cc names resolve before drafting

NDR dictates names fast and adjacent names merge. In this session **"Sarita Kroshni" was
"Sarthak + Roshni"** — the same two people his previous instruction on the same matter had
named. Before drafting, resolve every name to a real address:

- `gws_dwd_resolve` for a Workspace account (no match = no such mailbox)
- the contacts sheet's `employees` tab (small, authoritative for staff + telegram ids) and
  `contact_resolver`
- Gmail search for the name / `sarita@draas.com`, Drive `name contains`, tracker assignee
  column, WhatsApp chats

If none of those yield a person, say so plainly in the completion report and state the reading
you used and why (prior instruction on the same thread named those people), and offer to redo
it with the right address. Do not silently invent an address, and do not stall the draft —
it is a draft, NDR reviews before sending.

## Tracker side of "I'm assigning it to you"

If a row for this task already exists and is already assigned to that person, **do not create a
duplicate row**. Append a dated note to the existing row's Last Update Notes and refresh Last
Update Date, recording that NDR scrapped the earlier drafts and re-issued the simpler ask, with
the new draft ID. Say in the completion report that the assignment already existed.
