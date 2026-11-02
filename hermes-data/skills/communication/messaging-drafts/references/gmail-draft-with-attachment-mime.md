# Gmail draft WITH an attachment (MIME) + Cc verification

Verified 2026-10-05 sending-adjacent work: BHC / Prestige Hermitage Flat 601 rent invoice +
maintenance demand note, drafted from Bharat's own account (sales1.blr@draas.com).

## Why the native draft tool is not enough

`gws_gmail_draft_create` takes `to` / `cc` / `subject` / `body` only — **no attachments**.
`gws_gmail_reply_draft` likewise. For a draft that must carry a PDF (invoices, demand notes,
signed docs, decks) build the MIME message yourself and create the draft via the API. This is
still **draft-only** — never call `messages().send()`.

## The recipe (draft only — never send)

```python
from tools.gws_auth import build_service
import base64
from email.message import EmailMessage

gmail = build_service('gmail', 'v1', service_name='google-draas')   # session user's OWN token

msg = EmailMessage()
msg['To']      = 'C-R.Priya@fcdo.gov.uk'
msg['Cc']      = 'echamundeshwari@draas.com, rnr@draas.com'
msg['Subject'] = 'Rent Invoice & Demand Note, against Flat No. 601 @ Prestige Hermitage'
msg.set_content(body)                       # plain text; use set_content(html, subtype='html') for HTML

with open(pdf_path, 'rb') as f:
    msg.add_attachment(f.read(), maintype='application', subtype='pdf', filename=fname)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
print(draft['id'])          # -> 'r913530658586936028' (draft id, NOT the message id)
```

- Rename the attachment on the way in (`filename=`) — chat uploads arrive as
  `202610051529.pdf`, which is useless to the recipient.
- `msg['Cc']` takes a comma-joined string; multiple addresses in one header is fine.

## HTML body + attachments (NDR preference, 2026-10-14)

NDR explicitly REJECTS plain-text emails for authority/grievance correspondence
("This email is pure text and it's not easily readable... the email body itself
should have all of the facts... highlighted with bullets, bold, maybe use HTML
CSS"). Pattern that satisfies him — used for the Kanta Ranka passport grievance:

- `msg.set_content(plain_text)` (accessible fallback) then
  `msg.add_alternative(html_body, subtype='html')` — multipart/alternative with
  both views, then `add_attachment(...)` for each PDF/JPG. Works in one
  EmailMessage; the native `gws_gmail_draft_create` (html flag) cannot carry
  attachments, so build MIME yourself via `build_service('gmail','v1',...)`.
- HTML: inline CSS only (Gmail strips <style>), Georgia/serif ~14px, a
  light-background "request in one line" summary box at top, bold section
  headers, bullet lists for facts/appointments, and a TABLE of every attached
  document (number | filename | what it establishes). All facts/ARNs/dates live
  IN the body, attachments carry the evidence.
- Verified 9-PDF/JPG attachment build: downloads via drive `files().get_media`,
  then raw-base64 `drafts().create`. Verify by re-reading the draft with
  `messages().get(format='raw')` and walking `msg.walk()` parts — count
  attachments + confirm `text/html` part present. Then delete any stale
  plain-text drafts.

## ⚠️ `gws_gmail_get` does NOT show the Cc header — verify with `format='raw'`

A created draft read back through `gws_gmail_get` returns `to`, `subject`, `attachments` and
`labelIds: ["DRAFT"]` — **no `cc` key at all**. That is an omission in the reader, not a
missing header. Never tell the user the Cc is set (or missing) off that output.

Verify the real headers from the raw MIME:

```python
m = gmail.users().messages().get(userId='me', id=<msg_id>, format='raw').execute()
raw = base64.urlsafe_b64decode(m['raw']).decode('utf-8', 'ignore')
for line in raw.split('\n')[:20]:
    if not line.strip():
        break
    if line.lower().startswith(('to:', 'cc:', 'subject:', 'from:', 'date:')):
        print(line.strip())
```

Claim "Cc verified" only after this read. `labelIds: ["DRAFT"]` is the proof nothing was sent
— say it explicitly when reporting.

## Resolving a voice-garbled CC name — mine the thread, don't guess

Bharat said *"keeping **Echanumdeshewari** in CC"*. No Gmail hit for that spelling and no
contact-resolver hit. Resolution came from the **document trail**, not the name:

1. `gws_gmail_search(service_name='google-draas', query='Prestige Hermitage')` and again on
   `query='fcdo.gov.uk'` — search the **document/project subject**, not the person.
2. The recurring thread *"Rent Invoice & Demand Note, against Flat No. 601 @ Prestige
   Hermitage"* showed the actual participants: `Eshwari Chamundeshwari <echamundeshwari@draas.com>`
   → `C-R.Priya@fcdo.gov.uk`. That is the CC Bharat meant.

Generalise: for a **recurring external document mail** (rent invoice, CAM demand note,
statutory filing), the previous cycle's thread is the authoritative source for To/Cc, the
house subject line, and the body shape. Search the subject; mirror the format.

## Cross-check the attached documents BEFORE handing the draft back

An invoice mail is only as good as the numbers in it. Read every attached page first (OCR via
`vision_analyze` on rendered pages when the PDF has no text layer) and reconcile:

- **Periods must agree.** Found in this case: the tax invoice covered **2 months** (Oct–Nov)
  while the maintenance demand note covered **3 months** (Oct–Dec) — whereas the previous
  cycle had both on the same 3-month window. Flagged to the user rather than silently sent.
- **Arithmetic**: rate × months = amount. (₹6,07,753 × 2 = ₹12,15,506. The OCR of the
  particulars cell read `12,15,504` — a digit misread; the total was the trustworthy figure.)
- **Series/quarter labels**: a demand note numbered "Q3" covering Oct–Dec (Q4) is a
  carry-over from last year's "Q4" doc — worth a light flag.
- **Unit identity**: "Flat 601" = 6th floor. Confirm the stated floor matches the unit number
  the user described.

Report the flags as *"check before you send"* items. Do not silently "fix" a source document.

## House format for DRA Projects → external lessee (BHC)

Subject: `Rent Invoice & Demand Note, against Flat No. <unit> @ <project>`
Body: greeting → "Please find attached..." → numbered list of each document (type, number,
date, period, rate, amount) → **total payable** → "Kindly process the payment as soon as
possible." → "Thanking you, Regards," → signature block (name, DRA Projects Private Limited,
phone, email).

Issuer identity on these documents: **DRA Projects Private Limited**, #2 Ranka Chambers,
31 Cunningham Road, Bangalore 560052 · GSTIN 29AACCD4378F1Z1 · CIN U70102KA2006PTC039826 ·
Kotak Mahindra Bank, Frazer Town, IFSC KKBK0008061 (A/c 165011009438).

Who normally sends it: **Eshwari Chamundeshwari (echamundeshwari@draas.com)**, DRA
TEAM-ACCOUNTS. When the user asks for the mail under their own name while Accounts is on CC,
create it as asked but flag the alternative sender — do not reassign silently.

Platform note: this is the **own-token** path (`service_name='google-draas'` = the session
user's own mailbox). Never DWD here.
