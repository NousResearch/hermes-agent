# Gmail attachment download + invoice verification (for email drafting)

When an email must reference an invoice/attachment (e.g. "clear this invoice"),
verify the attachment's facts FIRST so the draft is accurate (invoice no, date,
amount, GST split, name the invoice is raised in).

## Recipe 1 — download a Gmail attachment to disk (sanctioned path)

1. Find the message with `gws_gmail_search` / `gws_gmail_get`. Note the
   attachment filename + `attachmentId`.
2. `gws_gmail_attachment_get` returns base64 — for PDFs this can be TRUNCATED
   in the agent context. Do NOT trust the truncated string.
3. Download via the vault-backed client (never handle credentials in context):

```python
import sys, base64
sys.path.insert(0, "/opt/data")
from tools.gws_auth import build_service
svc = build_service("gmail", "v1", service_name="google-draas")  # resolved via gws_resolve_account
att = svc.users().messages().attachments().get(
    userId="me", messageId=MSG_ID, id=ATT_ID).execute()
open("/tmp/invoice/file.pdf", "wb").write(base64.urlsafe_b64decode(att["data"]))
```

Run with cwd=/opt/data. `service_name` must be the resolved vault key
(e.g. `google-draas` = sales1.blr@draas.com for Bharat).

## Recipe 2 — read a scanned PDF invoice

- `pdftotext -layout file.pdf -` → empty output = scanned/image PDF.
- Render: `pdftoppm -png -r 150 file.pdf /tmp/pages/pg` → `/tmp/pages/pg-1.png`.
- `vision_analyze` on the PNG with a specific extraction question (bill-to
  name, invoice no, date, HSN/SAC, amount + tax split, GSTIN, supplier).

## Workflow — invoice-clearing email (draft-only, NEVER send)

Worked example (2026-10-09): Bharat dictated "clear the invoice Chinmay sent,
email admin.blr@draas.com, Mr. Sartak, stating invoice is raised in Nishant Ranka's name."

1. Search Gmail for the sender — invoice lives in the same thread as the
   service proposal ("Chinmay OR Prestige Hermitage" → the 8-Oct mail from
   pmhermitage@prestigeconstructions.com with the PDF attached).
2. Download + OCR the invoice → confirm Invoice 5700000477, 08.10.2026,
   ₹5,900 (₹5,000 + CGST ₹450 + SGST ₹450), billed to Nishant Ranka,
   supplier Falcon Property Management Services, service = flat cleaning
   Unit 2201 Prestige Hermitage (Sept 2026).
3. Use the recipient the user TYPED, not the voice garble — voice said
   "admin3.plr@drwsdras.com", user typed admin.blr@draas.com → typed wins.
4. Body: full invoice details + explicit statement it is raised in the name
   of Nishant Ranka + request to clear/process.
5. Create the Gmail draft ONLY (`gws_gmail_draft_create`) — report draft id,
   tell the user it is in their Drafts to send. `gws_gmail_draft_create` has
   no attachment param; put the reference number + details in the body and
   offer to upload/share the PDF separately.

## Pitfalls

- Voice dictation garbles email addresses — always defer to the user's typed
  correction when the two differ.
- An invoice addressed to the owner (not the company) is often intentional —
  the clearing email may need that exact statement for accounts to process it.
- Prestige Hermitage Unit 2201 is NDR/Nishant's personal flat; cleaning is
  billed by Falcon Property Management Services via property manager Chinmay
  Pimpalkhute (pmhermitage@prestigeconstructions.com). DRAAS accounts clearing
  contact for such invoices: admin.blr@draas.com (Mr. Sartak).