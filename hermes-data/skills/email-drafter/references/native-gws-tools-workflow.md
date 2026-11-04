# Native GWS Tools Workflow (Primary Path)

**Companion references:** `references/work-order-email-pattern.md` (replicate an existing sent HTML work-order mail + attach an image via MIMEImage; kids'/family WOs), `references/bulk-gmail-archive.md` (mass-archive a sender blast — 237-mail pass workflow, batchModify, search/pagination pitfalls).

**Not drafting for NDR?** A non-NDR session user's mailbox (Bharat / sales1.blr@draas.com →
Gowri Singh `gsingh@draas.com`, Cc Roshini Ranka `rnr@draas.com`) follows the same rules with
different account resolution — see the `messaging-drafts` umbrella,
`references/drafting-from-a-non-ndr-mailbox.md`. Key points: resolve the account with
`gws_resolve_account()` first (in his session the `google-draas` vault key holds
sales1.blr@draas.com, not NDR); `contact_resolver` needs an explicit
`account_email='sales1.blr@draas.com'` for a non-NDR user; and his standing wording rule is
state-scope-honestly ("this was not our scope, treat the attachment as reference only").

The native `gws_gmail_*` tools (not `google_workspace_manager`, not terminal/python bridge) are the **primary sanctioned path** for email operations. They are available in the default Hermes toolset and require no venv, no `sys.path.insert`, no credential plumbing.

## Available Tools

- `gws_gmail_search(query)` — search ndr@draas.com mailbox
- `gws_gmail_get(message_id)` — read full message (headers + body + attachment metadata)
- `gws_gmail_thread_get(thread_id)` — fetch entire thread
- `gws_gmail_draft_create(to, subject, body, cc, bcc, html)` — new email draft (NEVER sends)
- `gws_gmail_reply_draft(message_id, body, cc, bcc, html)` — threaded reply draft (auto-populates To from original sender; DOES NOT send)
- `gws_gmail_draft_delete(draft_id)` — remove a draft
- `gws_gmail_draft_list()` — enumerate drafts
- `gws_gmail_labels_modify(message_ids, add_label_ids, remove_label_ids)` — relabel messages
- `gws_gmail_trash(message_id)` — move to Trash
- `gws_gmail_attachment_get(attachment_id, message_id)` — download attachment as base64
- `gws_resolve_account(account)` — find the correct service_name for any account

For DWD (another person's @draas.com mailbox): use `gws_dwd_gmail_*` equivalents.

## Typical Workflow: Voice Dictation → Reply-All Draft

This is the pattern validated 2026-10-03 for **voice-dictated meeting outcomes** requiring a structured reply-all:

### Step 1: Find the Thread
```python
gws_gmail_search(query='subject:"Meeting Topic" from:sender')
```
Use specific subject keywords + participant names. If the subject is fuzzy, search broader and iterate. The thread's `threadId` is returned on the first result — note it.

### Step 2: Read the Original Email
```python
gws_gmail_get(message_id="...", service_name="google-draas")
```
Read the body to understand formatting style (bold headers, section dividers, bullet style, tone) so the reply matches. The user often says "same formatting" — match the original's HTML structure.

### Step 3: Craft the Reply Draft
```python
gws_gmail_reply_draft(
    message_id="...",   # the message you're replying to
    body="<html>...</html>",
    html=True,          # REQUIRED — without this HTML renders as literal text
    cc="colleague@draas.com"  # reply-all recipients beyond the original From
)
```

Key rules:
- `message_id` (NOT `thread_id`) — the tool handles threading automatically
- `html=True` is **mandatory** for formatted emails; without it the HTML source renders as literal text
- `cc=` parameter adds Cc headers for reply-all
- The tool auto-populates To from the original message's sender

### Step 4: Verify — existence AND recipients

```python
gws_gmail_draft_list()          # confirms the draft exists in Drafts
```

`draft_list` only proves the draft exists — it does **not** prove `to`/`cc` are right. For a
reply-all, read the draft's raw MIME headers before telling NDR it's ready:

```python
import base64
from email import message_from_bytes
d = svc.users().drafts().get(userId='me', id=draft_id, format='raw').execute()
m = message_from_bytes(base64.urlsafe_b64decode(d['message']['raw'].encode('ascii')))
print(m['To'], '|', m['Cc'], '|', m['Subject'], '|', m['In-Reply-To'][:60])
```

Verified 2026-10-05: `gws_gmail_reply_draft` auto-populated **To** from the replied-to message's
sender (Manju) and honoured the `cc=` list (Harsimran, Viraj, Amit Saraf), with `In-Reply-To` /
`References` set so it stayed in-thread. Verify rather than assume — especially the Cc list,
which is the whole point of a reply-all.

**Run the verification from a script FILE, not `python3 -c` (2026-10-07).** Passing multi-line
Python via `-c` with `json.dumps(code)` breaks on escaped newlines
(`SyntaxError: unexpected character after line continuation`). Write the verifier with
`write_file` to `/tmp/verify_drafts.py`, then
`HERMES_SESSION_USER_ID=7449813913 /opt/hermes/.venv/bin/python3 /tmp/verify_drafts.py` —
print WHOAMI first (must be `ndr@draas.com`), then per draft: To, Cc, Subject (decode the
RFC2047 `=?utf-8?q?…?=` if it matters), In-Reply-To prefix, and `threadId` (must match the
target thread for replies).

**gws_ops python bridge: pass raw Credentials, NOT a built service (2026-10-07).**
`gws_ops.gmail_reply_draft(creds, message_id=..., body=..., html=True, cc=...)` expects
`google.oauth2.credentials.Credentials` — get them via
`gws_auth.load_credentials(service_name='google-draas')`. Do NOT pass
`gws_auth.build_service('gmail','v1', ...)` which returns a googleapiclient **Resource** —
that fails inside `gws_ops._build` with `AttributeError: 'Resource' object has no attribute
'authorize'`. Native `gws_gmail_*` tools have no such issue; this pitfall only hits the
execute_code/python bridge path.

**`format='full'` returns header names LOWERCASE — verify case-insensitively (2026-10-13).**
Gmail lowercases the header keys (`to`, `cc`, `subject`) on tool-created drafts, so the usual
`{h['name']: h['value'] for h in payload['headers']}.get('To')` idiom prints `None` and looks
like the recipients were never set. Either read `format='raw'` and use `m['To']` (the
`email.message` API is case-insensitive), or normalise keys with `.lower()`. Verified on the
Manish/Dharmesh Hemant-termination reply-all: `format='full'` showed `to=None, cc=None,
subject=None` while the same draft's headers were in fact correct (`to: Manish Ranka
<mdr@draas.com>`, `cc: dharmesh.ranka1@gmail.com`, `subject: Re: DRA Projects Pvt Ltd …`).
On that false alarm do NOT re-create the draft — just re-read it case-insensitively.

**Stale draft on the same thread — delete it when recreating (2026-10-07).** If a thread
already carries an older draft and ndr asks for a fresh one, `gws_gmail_draft_list()` then
`gws_gmail_draft_delete(draft_id)` the old one so the thread carries exactly one draft.
Leaving both drafts clutters the Drafts view and the wrong one may get sent. Verified:
deleted `r3020989636002646597`, kept `r6665899269779430470` on thread `1a10c71519ac4760`.

**PHANTOM SUCCESS — never trust the return string, always read the draft back (2026-09 to
2026-10, Behera/Jasleen thread).** Repeatedly the bridge (`draft_reply_create`) and even some
raw `drafts().create` calls returned a plausible-looking `r-…` draft id and "Draft created"
message while NOTHING existed in Gmail — NDR could not see the email in his Drafts at all.
Some of these calls had silently gone down a path that did NOT create a draft object. The
ONLY reliable proof a draft exists: `gws_gmail_draft_list()` and read the raw MIME back
(`svc.users().drafts().get(id=..., format='raw')`) and verify To/Cc/Subject/In-Reply-To/
References AND the threadId. If a draft "update" returns a NEW threadId, Gmail has detached
it from the target thread — delete and recreate rather than trust it.

**In-Reply-To must be the anchor's OWN Message-ID, not its In-Reply-To (2026-09-24).** For a
reply to land in the same conversation: `In-Reply-To` = the exact `Message-ID` of the message
you're replying to; `References` = that message's full References chain + its Message-ID
appended. Pointing In-Reply-To at the anchor's *earlier* In-Reply-To produces a new thread on
send. When Gmail returns a provisional `threadId` on the draft, it is recomputed at SEND time
from these headers — verify headers, not the provisional threadId. A sent email that went out
with NO In-Reply-To/References created a brand-new conversation (Behera, 24-Sep-2026) — that
is exactly why NDR saw it "not in the thread".

**Attach the docx/PDF to a threaded draft via raw MIMEMultipart (2026-09-24).** Native and
bridge draft tools have no attachment parameter. Build `MIMEMultipart("mixed")` with the
HTML body as `add_alternative(..., subtype='html')` and the file as
`email.mime.application.MIMEApplication(bytes, _subtype='pdf')` / octet-stream for docx, set
To/Cc/Subject/In-Reply-To/References, `base64.urlsafe_b64encode(mime.as_bytes())`, then
`svc.users().drafts().create(body={'message': {'raw': raw, 'threadId': target}})` — pass
threadId INSIDE the message object. BYTE-VERIFY the attachment after creation (compare sha256
of the embedded part vs the local file) — presence in `parts` metadata is not proof.

**Image attachments use MIMEImage, NOT MIMEApplication (2026-10-12).** `MIMEApplication(bytes, _subtype='jpeg')`
yields `application/jpeg` (cosmetic wrong type); `from email.mime.image import MIMEImage; MIMEImage(bytes, _subtype='jpeg')`
yields `image/jpeg`. Verify the part's content-type in the read-back (`VERIFY_PARTS`) and sha256 the payload.

**Clone the exact HTML of a previously sent email (work-order / template replication, verified 2026-10-12).**
"Exact same formatting as the <earlier> email" means fetch the SOURCE of the sent message, not the plain-text body:
`svc = build_service('gmail','v1',service_name=...); msg = svc.users().messages().get(userId='me', id=<msg_id>, format='full').execute()` —
walk `payload['parts']`, take the `text/html` part, decode `body['data']` base64url, save to a file, then rewrite names/amounts/sections
inside that exact markup (container div, summary box `#1d1d1f` + `#f5a623` h3 borders, etc.). Sent-mail search (`gmail_search "work order"`) finds the prior WO; `gmail_get` alone returns only the plain-text body, which is NOT enough to clone styling.
- **Family / kids' work-order emails (recurring):** NDR issues paid Work Orders to Ruhaan and Rivaan for design/3D-modelling work —
  same WO structure each time (objective & tools → functional requirements → deliverables D1–Dn → budget discipline → acceptance →
  payment → timeline → coordination). Resolution shortcuts: **"Pebli"/"Pebley"/"Mark Pebley" in dictation = Ruhaan (pebblyshark69@gmail.com)**
  (pebbly = his username alias), Rivaan = rankarivaan@gmail.com, Roshini (RNR) = rnr@draas.com.
- **"Send it for my personal email" = nishantranka@gmail.com (`google-gmail`**), even when the earlier work order went from ndr@draas.com —
  a personal/family job drafts from the personal account. Native `gws_gmail_draft_create` may return `draft_id` without exposing To/Cc —
  read the draft back (raw MIME) to verify headers; drafts are NEVER sent.

**`gws_gmail_draft_create(body=...)` does NOT decode HTML entities — pass a bare `&` (2026-10-16).**
With `html` unset/false the body is stored VERBATIM. Writing `Thanks &amp; Regards` in the
`body` argument produced a draft whose text literally reads `Thanks &amp; Regards`; the plain
sign-off needs a plain `&`. Entities are only correct INSIDE an `html=True` body. Detect it by
reading the draft back raw and checking the decoded body for `&amp;` / `&#39;` / `&nbsp;`;
fix by `gws_gmail_draft_delete(draft_id)` + re-create, or by `drafts().update()` in place.

**Shell-guard pitfall building HTML drafts via heredoc.** A terminal heredoc containing `&`
(characters common in HTML entities) is rejected with "Foreground command uses '&'
backgrounding" — write the Python to a file with `write_file` first, then run it
(`/opt/hermes/.venv/bin/python /tmp/create_draft.py`).

(draft-list only proves existence, not recipients):

```python
svc = build_service('gmail', 'v1', service_name='google-draas')
d = svc.users().drafts().get(userId='me', id=DRAFT_ID, format='raw').execute()
from email import message_from_bytes; import base64
m = message_from_bytes(base64.urlsafe_b64decode(d['message']['raw']))
print(m['To'], '|', m['Cc'], '|', m['In-Reply-To'])
```

**Reply-all recipients.** `gws_gmail_reply_draft(message_id=...)` auto-fills To from the
*message's sender* and adds only what you pass in `cc=` — it does NOT reconstruct the thread's
full To/Cc list. So for a reply-all to a message that is not the thread's first, put every
other recipient NDR named into `cc=` (To = the person being answered). If the To cell comes
back as someone unexpected, you replied to the wrong message in the thread.

**"Address to X, marking to Y" = To: X, Cc: Y (2026-10-13).** NDR's Indian-English dictation
uses "marking" for "mark a copy" — it means **Cc**, never To. Verified: *"one email, address to
Dharmesh, marking to Manish and CC"* → To `dharmesh.ranka1@gmail.com`, Cc `mdr@draas.com`; a
follow-up *"add Rajendra as well in the CC"* added `rajendra@draas.com`. Do not promote the
"marked" person onto To.

**"Reply all" = the NAMED set NDR dictates, not Gmail's Reply All (2026-10-13, Oasis P-370).**
Voice dictation example: NDR said "reply all to Sinchana, Bhuvanesh and Anbu" about the
Phase 2 pack for Janice/BNA. The historical thread also carried Janice, Nilanjana (BNA),
Roshini and Gowri on earlier messages — but the draft went ONLY to Sinchana (To, auto) +
Bhuvanesh + Anbu (Cc), because this was an internal correction mail and NDR named exactly
three people. Rule: never reconstruct the thread's historical participants; include exactly
what NDR dictates. Verification then confirmed To/Cc via raw-MIME read-back (see below).

**Ambiguous contact_resolver → lift address from NDR's own prior sent thread (2026-10-13).**
"Bhuvanesh" returned 5 candidates (B Hegde, Bhuvan, Bhuvanesh S Krishnan/Design Form, Anjali
Bhuvanesh, …) with no auto-select and the contacts sheet lacked the active address. Fix:
`gws_gmail_search "from:<candidate-address>"` / search the person's name across sent mail and
lift the header that appears in NDR's own non-bounced thread. Worked example: Bhuvanesh S
Krishnan (Design Form — Ranka Oasis villa designer) is reached at `mail.bskrishnan@gmail.com`
(NDR's direct "URGENT: architect deliverables" thread); Sinchana usually also copies
`bk@findingform.design`. Anbu = Anbarasan M <pm2.blr@draas.com> (verified from daily sent
sign-in mails). Check for a hard bounce (550 5.1.1) before reusing any address.

**Resolving a recipient who is not in the contacts sheet.** `contact_resolver` returns nothing
for people who never made it into the Google Contacts Sheet — common for external
counterparties (developer / agency staff). Get the address from a prior Gmail thread instead:
`gws_gmail_search` the name across sent mail and lift the address from the headers. This also
resolves voice-garbled surnames ("Amit Sharaph" → `amit.saraf@godrejventure.com`, found in
NDR's own sent mail on the same project thread).

**HARD RULE — every recipient must be contact-verified AND bounce-verified (2026-10-13, NDR).**
NDR's standing rule, restated after a bounced reply: *"all emails must be looked up, must be
available in contact, and must have previous emails sent in for that person — not bounced. If
it bounces, you have to look up the contact and get the email from there."* Before drafting to
any address:
1. The address must be in the contacts sheet OR found in a prior non-bounced email thread
   (prefer lifted-from-sent-mail headers over a guessed address).
2. Check whether any mail to that address has bounced. A Gmail hard bounce appears in the
   thread as the original message + a failure notice: `550 5.1.1 The email account that you
   tried to reach does not exist` (or `5.1.10`). A dead address must NEVER be re-used; find
   the correct one via contact_resolver / sent-mail headers before drafting.
3. If a bounce is found, do not silently substitute — tell NDR which address bounced, what it
   was replaced with, and that the replacement is verified from contacts/prior mail.
Concrete case (Oct 2026): `arvindsarchitects@gmail.com` was a guessed address (never in
contacts) and hard-bounced 27-Sep; the correct AJ Architects contact was Shrinidhi GV at
`shrinidhi.a.j.architects@gmail.com` (contacts sheet row 4249, and the address that actually
sent the sanction files on 21-Sep). The reply-all went to Arvind
(`arch_arvind2000@yahoo.co.in`) + Shrinidhi, with the dead address dropped entirely.

**Downloading an attachment.** Prefer `execute_code` + `build_service(...).users()
.messages().attachments().get(...)` writing bytes straight to disk. The native
`gws_gmail_attachment_get` result is large and gets persisted/truncated, so decoding the saved
payload can produce a corrupt file.

## When to Use Native Tools vs Terminal/Python Bridge

| Situation | Path |
|---|---|
| Search, read, reply-draft, draft-list | **Native tools** — always prefer these |
| Attachments (attach files to draft) | Terminal/python with `build_service` + `drafts().create()` raw MIME |
| Complex MIME manipulation | Terminal/python raw Gmail API |
| Bulk operations > 100 messages | Terminal/python loop (batch silently partial-fails) |
| Another user's mailbox (DWD) | `gws_dwd_gmail_*` native tools |

The native tools handle ~80% of NDR's email workflow. Only drop to terminal/python for:
- Attaching files to a draft (need MIMEMultipart construction)
- Reading raw MIME headers (need `format='raw'` endpoint)
- Complex body editing in place

## Voice Dictation → Structured Meeting-Outcome Drafts

When the user voice-dictates a multi-topic meeting outcome:

**Structure:**
- Bold `**Point N: Topic**` headers matching the original email's formatting
- Bullet points under each header (not paragraphs)
- Clarifications / corrections inline (user says "this is for S7, not S30" → keep original name in body but annotate the correction)
- Number sections cleanly so the recipient can reference them
- End with a concluding statement about outstanding items

**Format:** Match the original email's visual style — if it uses blue `<hr>` dividers, bold numbered headers, and bullet tables, replicate that. The user said "same formatting, you know, with all the appropriate bolding" — this is the explicit instruction to reproduce.

**Content processing:** The user's voice dictation will contain:
- Self-corrections ("this is for S7, not S30") — keep the correction, not the original
- Filler sounds ("PAG", "PAY", "UHM") — strip them
- Run-on sentences — break into clear bullets
- Ambiguous references ("it", "that") — resolve from context and make explicit

---

Family/kids' design work orders (Ruhaan/Rivaan, 914 Embassy Habitat — "same formatting" emails with attachment): template `templates/kids-work-order-email.html` + full class notes `references/family-design-work-orders.md` (voice-alias traps: "Pebby"=Ruhaan; Rivaan's dual email spellings; MIMEImage for image/jpeg; personal → google-gmail routing).