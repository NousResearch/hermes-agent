# Finding a document someone emailed us, verifying it, and drafting it on to someone else

Ask shape: *"check if any email from <company/person> has the <document> and if so share it with
<person> on email."* Verified 2026-10-26 — the **registered Deed of Declaration of Prestige
Hermitage** (58 pp, 3.27 MB) located in Chinmay Pimpalkhute's 25-Jan-2025 mail and drafted on to
Dharmesh. Sibling recipes: `new-email-html-attachment-nonascii.md` (brand-new mail + file),
`forward-existing-sent-mail.md` (forward a SENT mail to a new person),
`threaded-reply-pdf-attachments.md` (reply onto a live thread).

The hard part is never the draft — it is **proving the document exists, that it is the right
document, and getting its bytes onto disk without blowing the context window.**

## 1. Search ladder — the subject line is the last place to look

`gws_gmail_search` first, on the session user's own account (`gws_resolve_account` to confirm the
`service_name`). Run the rungs in this order and stop when the census in §1.3 shows the file.

**1.1 Name the sender, not the document.** Corporate documents come from **role mailboxes**, so
the human name in the ask usually is not the address:

| Ask says | Actual address |
|---|---|
| "Prestige Chinmaya" | `pmhermitage@prestigeconstructions.com` (Chinmay Pimpalkhute, Property Manager) |
| "Prestige front desk" | `fo.hermitage@prestigeconstructions.com` |
| association circulars | `prestigehermitageaoa@gmail.com` |

So: `from:prestigeconstructions.com has:attachment`. Searching the person's display name finds
replies *to* him; searching the domain finds everything he ever sent.

**1.2 Do not trust `filename:`.** `filename:deed OR filename:DOD OR filename:declaration` returned
Std-9 "Domains of Development" school mails, partnership deeds and lease "Deed of Adherence" —
OR-chained short tokens are effectively a fuzzy full-text search. `filename:` is only useful with
one long distinctive token (`filename:nomination`, `filename:3825`).

**1.3 The attachment census — `gws_gmail_thread_get(thread_id=...)`.** This is the tool that
matters: it returns, per message, an `attachments[]` array of `{filename, mimeType, size,
attachmentId}`. Search snippets and `gws_gmail_get` summaries never show attachments. One call on
the right thread settled this session in a single round trip.

**1.4 The document may be item 4 of 4, named only in the body.** This one was attachment #4 in a
mail whose subject was *"First Intimation: Formation of Association"*; the body said
`4. DOD - Attached`. No subject or filename search could have found it. **Read the `body` of
candidate mails, not just `subject`.** Government/association mail routinely bundles
notice + proxy + nomination + the actual instrument under one prosaic subject.

Alternative tell: a sibling of the same batch will name it — Manish's 9-Jul mail *"send me the
DoD"* and Dharmesh's *"Can you get hold of the Deed of Declaration?"* confirmed the document
existed and that nobody had circulated the file itself.

## 2. Get the bytes to disk — in code, never through the tool result

`gws_gmail_attachment_get` returns `dataBase64` **inline in the tool result**. A 3.27 MB PDF became
a 4.35 MB result, which was auto-persisted and truncated at ~4.2 MB — and then
`base64.b64decode` failed with:

```
binascii.Error: Invalid base64-encoded string: number of data characters (3934753)
cannot be 1 more than a multiple of 4
```

That error signature means **truncated payload**, not corrupt base64. Do not retry the tool —
fetch through the API and write straight to disk, so nothing large enters the transcript:

```python
import sys, base64, os
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
svc = build_service('gmail', 'v1', service_name='google-draas')

att = svc.users().messages().attachments().get(
    userId='me', messageId=MSG_ID, id=ATTACHMENT_ID).execute()
data = base64.urlsafe_b64decode(att['data'])
assert data[:5] == b'%PDF-'                  # cheap sanity check before trusting the bytes
open(OUT, 'wb').write(data)
print('bytes', len(data), 'on disk', os.path.getsize(OUT))
```

Use the tool-based fetch only for small attachments (< ~1 MB) where you want the content in
context anyway (a one-page scan to OCR inline).

## 3. Prove it is the document before you forward it

A 58-page scan with **no text layer** (`pdftotext` returns nothing) can be the wrong scan, a
part-only copy, or page one of something. Render and read before drafting:

```bash
pdfinfo X.pdf | head -20                     # pages, producer, creation date
pdftotext -f 1 -l 3 X.pdf - | head           # empty ⇒ scanned, go visual
pdftoppm -f 1 -l 2  -r 110 -png X.pdf pg     # front matter
pdftoppm -f 55 -l 58 -r 110 -png X.pdf end   # endorsement / annexures
```

then `vision_analyze` on the renders. What each page gives you:

- **p1** — the instrument's own title, the Act, the project, and (for a Deed of Declaration) the
  **owner schedule**: which entity owns which apartments, SBA and UDS per unit. This is what makes
  the covering note worth reading.
- **p2** — the registration artefact in these Karnataka packets: a **Kaveri Online Services**
  endorsement print with challan reference and date (here `RG0125000014755667`, 16/01/2025).
- **last pages** — ANNEXUREs (proxy forms, bye-laws), *not* the endorsement. Do not assume the
  registration page is at the back.

Then state the verified particulars in the draft — and only those.

## 4. Draft it on — with two judgement calls

Attachment mechanics: `EmailMessage` + `add_attachment` + `drafts().create` — recipe in
`threaded-reply-pdf-attachments.md` §"Compact variant". Never `gws_gmail_send`, never
`messages().send`.

**Call 1 — was the recipient already on the original?** Dharmesh was a To: recipient of the
Jan-2025 mail. Do not write as if he has never seen it; frame it as *"this is the copy Chinmay
circulated on 25 Jan"*. It also means he may already hold it — say what is new (the filename, the
fact that it is findable at all), not that the document is news.

**Call 2 — which of several addresses?** Use the last-used one, not a scatter. The house rule
(`forward-existing-sent-mail.md` §5.3) is ONE address per person; if you end up adding a work ID as
Cc, present it as an open question in the report rather than as settled.

**Never assert what a scanned document does not contain.** Dharmesh and Manish have argued since
July over whether the car parks appear in this DOD. With no text layer, silence is not evidence —
the draft asserted nothing either way and the reply to NDR offered to OCR all 58 pages. If the user
needs a negative finding, OCR the whole document first.

## 5. Verify the draft, then report

```python
d = svc.users().drafts().get(userId='me', id=DRAFT_ID, format='full').execute()
hdrs = {h['name']: h['value'] for h in d['message']['payload']['headers']}
for p in d['message']['payload'].get('parts', []):
    print(p.get('mimeType'), p.get('filename'), p['body'].get('size'))
```

Assert the attachment's `size` equals the bytes written in §2 (here 3,266,645 on both sides). A
draft id alone proves nothing.

Report in this order: **where it was** (sender + address, date, subject, attachment filename, size,
pages) → **what the document actually is** (title/Act/owner schedule/registration date) → **the
draft headers** (To/Cc/Subject) → **the flags** (address choice, anything not asserted, anything
still open). Lead with the finding, not with the process.

## Pitfalls

- Searching only the user's main mailbox: check all four vault identities before concluding
  "not received" (`gws_resolve_account` with no args lists them). Here only `google-draas` had it.
- A subject-only search on a prosaic subject ("First Intimation…") buries the real instrument.
- `filename:` OR-chains are noise; `has:attachment` + thread census is signal.
- Trusting a returned draft id without re-reading the MIME parts — the attachment can silently be
  absent when the raw build threw.
- Forwarding page-one-only photos as if they were the instrument; check `pdfinfo` page count
  against what the sender claims.
