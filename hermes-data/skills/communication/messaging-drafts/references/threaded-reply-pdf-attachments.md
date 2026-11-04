# Threaded reply with PDF attachments (hand-built MIME) + in-place attachment swap

Use when the user asks to draft a follow-up email in an existing thread (reply to a
SENT email or earlier intro mail) and attach files — especially PDFs that already
live on Google Drive. Worked 2026-10-07 for the Ranka Amber → Squarefeet Realtors
channel-partner follow-up on psingh@draas.com.

## When this applies

- User says "draft an email to <partner>, we already sent an intro earlier, add to same thread".
- Attachments include a Drive-hosted PDF (e.g. a cost sheet shared via a
  `drive.google.com/file/d/<FILE_ID>` link).
- User later says "change attachment 1 to <another Drive link>" — swap in place, keep draft ID + threading.

**Durable rule: the native tools (`gws_gmail_draft_create`, `gws_gmail_reply_draft`)
accept NO attachments. Any draft that must carry a file goes through hand-built MIME
(`build_service('gmail',...)` + `drafts().create`/`.update` with a raw base64 message).**

> Runnable parameterised builder (anchor + thread + to/cc + body file + attachments, prints
> draft id and a verified flag, never sends): `email-drafter/scripts/draft_reply_with_attachment.py`.
>
> **When the body comes from a colleague's internal mail, read
> `internal-briefing-to-external-mail.md` FIRST** — the mechanism here is only half the job;
> that file carries the leak checklist (a verbatim forward ships "keep this mail on hold with
> <client>"), the compiled-mail shape, and the absence-grep verification that catches a leaked
> body which every presence check passes.
> The email-drafter side of this class (anchor choice, `Re: Fwd:` trap, WhatsApp image → PDF,
> reading a photographed spreadsheet) is in
> `email-drafter/references/threaded-reply-with-attachment.md`.

## Compact variant — a NEW draft (no threading) with one PDF

The `MIMEMultipart` recipe above is the general one. When the draft is a **brand-new
message** (no thread to join), the body is plain text, and there is a single
attachment, `EmailMessage` is ~10 lines shorter and needs no manual part headers
(verified 2026-10-23, forwarding a scanned court notice to ndr@draas.com):

```python
from email.message import EmailMessage
msg = EmailMessage()
msg['To'] = 'ndr@draas.com'
msg['From'] = 'admin3.blr@draas.com'
msg['Subject'] = '...'
msg.set_content(body_text)
msg.add_attachment(open(PDF, 'rb').read(),
                   maintype='application', subtype='pdf', filename='Display Name.pdf')
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
```

`add_attachment` sets the Content-Type/Content-Disposition headers and base64-encodes
the payload for you. It does NOT add `Date`/`Message-ID` — fine for a draft, but a
threaded reply still wants the `In-Reply-To` / `References` headers, so reach for
`MIMEMultipart` the moment threading matters. Verify either route by reading the draft
back (`gws_gmail_get`) and checking `attachments[].filename` and `.size`, not just that
the call returned an id.

## The attachment on hand is a legacy `.doc` — convert via Drive, don't ship a Word file

Lawyers and vendors send `.doc` (and `.docx`) opinions/letters that must go out as PDF, and the
box has no `antiword` / `catdoc` / `wvText` / `soffice`. Do **not** attach the `.doc` and do not
hand-build a text-only PDF (you would lose the letterhead, which is often the evidence). Google
Drive does the conversion, and it is one sanctioned round trip:

```python
from googleapiclient.http import MediaFileUpload
g = build_service('drive','v3', service_name='google-draas')

created = g.files().create(
    body={'name': 'TMP_CONVERT', 'mimeType': 'application/vnd.google-apps.document'},  # ← conversion
    media_body=MediaFileUpload(src_doc_path, mimetype='application/msword'),
    fields='id').execute()
pdf = g.files().export(fileId=created['id'], mimeType='application/pdf').execute()
open(out_pdf, 'wb').write(pdf)
g.files().delete(fileId=created['id']).execute()      # leave no temp Doc in the mailbox's Drive
```

- Naming the new file's `mimeType` as a **native Google Doc** is what triggers Word→Docs import;
  the export then renders the real layout, letterhead included.
- Delete the temporary Doc immediately — the user's Drive is not a scratch space (§ pitfalls).
- `pdftotext -layout` the exported PDF: it is both the deliverable **and** a verification read.
  Verified 2026-10-23, a `.doc` legal opinion exported this way gave an 11-page PDF whose
  letterhead independently confirmed the advocate's name, court address and cell number — the
  phone the user had dictated garbled. Tell the user when a document has confirmed a fact.
- If the source is `.docx`, the same trick works (`mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'`),
  or read the text directly out of `word/document.xml` in the zip when you only need the words.

## Attachment sources

- **Drive file** → `drive.files().get_media(fileId=...)` (below).
- **A Gmail message's own attachments** (forward-style: "send the statements that
  Kotak emailed me"): walk the message payload for `body.attachmentId`, then
  `gmail.users().messages().attachments().get(userId='me', messageId=MSG_ID, id=ATT_ID).execute()`
  → `base64.urlsafe_b64decode(data['data'])` → write to disk → attach via the same
  `MIMEApplication` helper. One message id may yield multiple attachment ids; download
  each and attach in the order shown in the email.
  - **⚠ Do NOT use the `gws_gmail_attachment_get` TOOL for anything over ~1 MB.** It returns
    the whole file as `dataBase64` inside the tool result; a 3.2 MB PDF → 4.36 MB result →
    harness truncates to a persisted file (~4.2 MB) → `binascii.Error: Invalid
    base64-encoded string: number of data characters (3934753) cannot be 1 more than a
    multiple of 4`. That is truncation, not a padding bug — retrying or repairing the string
    cannot work. Call `attachments().get()` from `execute_code` and write the decoded bytes
    straight to disk (verified 2026-10-26, a 58-page registered deed). Then prove the file:
    `pdfinfo` for page count + `Encrypted: no`, and `pdftoppm -f 1 -l 2 -r 110 -png` +
    `vision_analyze` when the scan has no text layer.
- **Multiple recipients**: `msg['To'] = "a@x.com, b@y.com"` (comma-separated) works for a
  new batch forward; no threading headers needed when it's a new draft (verified 2026-10-08:
  6 Kotak RERA statement PDFs → draft to Eshwari + Sarthak, all attachments present in Drafts).

## Recipe

```python
import sys, os, base64
from email import policy
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate, make_msgid

sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')  # resolve service_name via gws_resolve_account; run with HERMES_SESSION_USER_ID=<session-slug>
drive = build_service('drive', 'v3', service_name='google-draas')
USER = 'me'
```

### 1. Preserve threading headers from the last message in the thread

Copies To + Subject + In-Reply-To + References so the draft lands under the existing
conversation (not a new one). If the draft ALREADY exists (we created it earlier),
fetch the draft itself and reuse its headers instead:

```python
# A) From a sent/received message:
orig = gmail.users().messages().get(userId=USER, id=MSG_ID, format='metadata').execute()
hh = {h['name']: h['value'] for h in orig['payload']['headers']}
# B) From an existing draft (to preserve headers when rebuilding):
draft = gmail.users().drafts().get(userId=USER, id=DRAFT_ID, format='raw').execute()   # ⚠ format='raw' REQUIRED
msg = email.message_from_bytes(base64.urlsafe_b64decode(draft['message']['raw']), policy=policy.default)
to, subject, in_reply, refs = msg['To'], msg['Subject'], msg['In-Reply-To'], msg['References']
```

**PITFALL: `drafts().get()` / `messages().get()` only return `message['raw']` when
passed `format='raw'`.** `format='full'` gives structured payloads without raw →
`KeyError: 'raw'`. `format='metadata'` omits `raw` too (only headers). Always decode
via `base64.urlsafe_b64decode(message['raw'])`.

### 2. Download the Drive PDF and build the MIME

```python
FILE_ID = '1UsEIsuAyG1vtS-7UjuKiRPVxjN7897Qy'   # from the /file/d/<ID>/view link
with open(OUT, 'wb') as f:
    f.write(drive.files().get_media(fileId=FILE_ID).execute())
```

`get_media` works for `application/pdf`. Only use `files().export(fileId=..., mimeType='application/pdf')`
for native Google formats (Sheets/Docs). Drive links arrive as `view` URLs — extract
the file id between `/file/d/` and `/view`.

```python
outer = MIMEMultipart()
outer['To'] = to; outer['Subject'] = subject
outer['In-Reply-To'] = in_reply
outer['References'] = refs if refs else in_reply
outer['Date'] = formatdate(localtime=True)
outer['Message-ID'] = make_msgid(domain='draas.com')
outer.attach(MIMEText(body_text, 'plain', 'utf-8'))

def attach(path, disp_name):
    with open(path, 'rb') as f:
        part = MIMEApplication(f.read(), _subtype='pdf')
    part.add_header('Content-Disposition', 'attachment', filename=disp_name)
    outer.attach(part)
```

### 3. Create OR update the draft

- New draft: `gmail.users().drafts().create(userId=USER, body={'message': {'raw': enc, 'threadId': THREAD_ID}}).execute()`
- **Swap attachments in place**: `gmail.users().drafts().update(userId=USER, id=DRAFT_ID, body={'message': {'raw': enc}}).execute()` — keeps the SAME draft id, stays in Drafts, threading preserved because the MIME headers carry it.

## Size the attachments BEFORE you create the draft

A scanned deed that comes out of a redaction pass is often *bigger* than the original
(`apply_redactions()` re-encodes the page images). 2026-10-29: a 2.2 MB, 6-page deed produced a
**3.18 MB** 4-page extract, which went into a draft and had to be thrown away and rebuilt at
510 KB (re-render 130 dpi / JPEG q68 — see `pdf-redaction` →
*"Shipping only the necessary pages of a scanned deed"*).

- Check the byte size of every attachment (`os.path.getsize`) **before** building the MIME, and
  aim for a total well under ~5 MB for a lender/authority recipient.
- Read the payload sizes back after creating the draft and report them, not just the filenames:

```python
def walk(part, out):
    if part.get('filename'):
        out.append((part['filename'], part.get('body', {}).get('size')))
    for p in part.get('parts') or []:
        walk(p, out)
atts = []; walk(draft_msg['payload'], atts)
print(atts, sum(s or 0 for _, s in atts))
```

- **Already created it with the wrong file?** `drafts().delete(userId='me', id=DRAFT_ID)` then
  create the corrected draft, and say which draft id is live. Do not leave two drafts of the same
  mail in the folder — the user will send the wrong one. (`drafts().update` also works when you
  want to keep the id; use delete+create when the *shape* of the message changed.)
- Name the attachment after its substance, not its temp path:
  `"Sevaganapalli Land Partners - Reconstitution Deed 27-02-2025 (extract).pdf"`, and make the
  body's numbered attachment list match those filenames exactly.

## Verification (always re-read the draft)

```python
d = gmail.users().drafts().get(userId='me', id=DRAFT_ID, format='raw').execute()
msg = email.message_from_bytes(base64.urlsafe_b64decode(d['message']['raw']), policy=policy.default)
for part in msg.walk():
    fn = part.get_filename()
    if fn:
        print(fn, len(part.get_payload(decode=True)))
body = msg.get_body(preferencelist=('plain',)).get_content()
```
Check: To = partner, Subject starts `Re:`, In-Reply-To + References present, attachment
names + sizes match expectations.

## Body convention

When the user swaps an attachment ("change attachment 1 to this sheet"), ALSO update
the numbered list in the body to match the new filename (e.g. item 1 →
"Ranka Amber - Cost Sheet (Developer Share Units) (Oct 2026)"). The numbered list
and attachment filenames should read consistently.

## Pitfalls

- **Anchoring on a FORWARD gives you `Re: Fwd: <subject>`.** Strip all leading `re:`/`fw:`/`fwd:`
  tokens and add a single `Re:` before creating. If the draft is already made, delete and recreate,
  then confirm only one live draft remains. (Same trap for a stale chain re-used months later —
  it is still the right home if the substance lives under it. Verified 2026-10-27.)
- **Attachment source may be WhatsApp, not Drive/Gmail.** `wa_get_media(messageId=…)` →
  `/data/hermes/tmp/wa-media/`; for a photo of a screen, convert to a one-page PDF
  (`Image.open(p).convert('RGB').save(out,'PDF',resolution=150)`) and name it after the substance
  (`<person> - fee split <date>.pdf`), never `IMG-….jpg`.
- **Draft update vs create**: `drafts().update` is the in-place path; `drafts().create` always makes a NEW draft. Rebuilding from the old draft's own headers keeps threading intact.
- **/opt/hermes is not writable from the agent session** — write scripts to /opt/data, run with `/opt/hermes/.venv/bin/python3` and `HERMES_SESSION_USER_ID=<session-slug>` (e.g. psingh-8502281203). Verify the acting user with `drive.about().get(fields='user(emailAddress)')` / `gmail getProfile` — files land in the SESSION user's Drive, not ndr's.
- Attachment order in MIME = order shown in the Gmail UI; attach in the order the body lists them.