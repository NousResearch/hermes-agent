# Fresh Gmail draft WITH attachments (no native tool param)

The native `gws_gmail_draft_create` tool has no attachment parameter. When the
email needs files attached (logo, GST cert, PDFs) — e.g. sending a brand logo +
GST certificate to a vendor for a stall booking — create the draft directly via
`tools.gws_auth.build_service('gmail','v1', service_name=<resolved>)`.

**HARD RULE: draft only. Never `messages().send()`.** This builds a real Gmail
draft the user reviews and sends themselves.

## Recipe (verified 2026-10-09, Bharat/sales1.blr @ google-draas)

```python
from tools.gws_auth import build_service
import base64, mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

svc = build_service('gmail', 'v1', service_name='google-draas')  # resolve first via gws_resolve_account
msg = MIMEMultipart('mixed')
msg['To'] = 'vendor@example.com'
msg['Subject'] = '...'
msg.attach(MIMEText(body, 'plain'))

for path, disp_name in attachments:          # [(path, display_filename), ...]
    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = 'application/octet-stream'
    maintype, subtype = ctype.split('/', 1)
    part = MIMEBase(maintype, subtype)
    with open(path, 'rb') as f:
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment', filename=disp_name)
    msg.attach(part)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = svc.users().drafts().create(userId='me',
        body={'message': {'raw': raw}}).execute()
print(draft['id'])
```

Verify after creation with `gws_gmail_draft_list(service_name=...)` — the new
draft id must appear.

## Prep the attachments first

- Transparent PNG logos can render with a black box in Gmail dark mode /
  previews. Make a white-background copy before attaching:
  `Image.open(png).convert('RGBA'); paste onto white; save as RGB`.
- For a PDF cert, sanity-check contents + pull key fields (GSTIN etc.) with
  `pdftotext file.pdf -` before sending.
- Confirm the attachment entity matches the vendor/invoice context — e.g.
  DRA Thindlu Land Partners (GSTIN 29AAXFD2296G1ZS) vs DRA Realty Pvt Ltd
  (29AALCD9962L1ZW) are different GST entities; pick the right certificate.

## DRA brand asset vault (find the RIGHT logo before attaching)

Finding the correct logo is the slow part of these drafts. Verified locations + tagline trap
(09-Oct-2026):

- **The official "HOME OF PRIDE" logo** — DRA serif wordmark + gold wing mark + gold rule +
  `HOME OF PRIDE` tagline, TRANSPARENT background (RGBA 800×544):
  `/data/hermes/cache/analysis/dra_logo/dra-logo.png`. Same art as every DRA Realty letterhead.
  **Reuse this file — don't re-hunt Drive.** Make the white-background copy for the attachment.
- **Drive holds several DIFFERENT DRA logos — a filename match is NOT a content match:** the file
  named `DRA LOGO.pdf` carries the tagline **"EXPERIENCE NICE"**, `DRA-logo.jpg` reads **"BY DRA"**,
  and `DRA logo.svg` is the mark only (no tagline). When the user asks for "the logo with HOME OF
  PRIDE", **verify the tagline with `vision_analyze`** before attaching — never attach off the name.
- **GST certificates are entity-specific.** `DRA THINDLU GST CERTIFICATE (1).pdf` (Form GST REG-06,
  DRA THINDLU LAND PARTNERS, GSTIN **29AAXFD2296G1ZS**) lives in the Thindlu project folder;
  DRA REALTY PRIVATE LIMITED's certificate (GSTIN **29AALCD9962L1ZW**) is a separate file. Pick the
  entity that is actually billing/contracting — for Ranka Udaya stall/vendor paperwork that is
  usually DRA Thindlu Land Partners. Sanity-check with `pdftotext file.pdf -` before sending.

## Voice-dictated names in the draft

When the user's voice note garbles a venue/project name ("Sukruti" →
"Sukriti Cultural Association, opp. TISB"), cross-check against the Drive
folder in the user's My Drive before finalising the draft. Flag the
discrepancy to the user (they can confirm), and never claim an email was
"sent" — it is a draft in their Drafts folder.