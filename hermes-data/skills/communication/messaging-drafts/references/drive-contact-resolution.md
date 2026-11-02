# Drive-Based Contact Resolution (When Contact Not in Sheet)

## Problem

The contact (seller, buyer, or counterparty in a property transaction) is **not in the Google Contacts sheet**. No sheet match found.

## Solution: Search Drive Property Documents

Property transaction documents in Drive often contain the counterparty's full name, PAN, Aadhaar, and address — which includes a phone number or at minimum enough detail to construct a wa.me link for initial outreach.

**Workflow:**

1. Search Drive for the property project/flat folder (e.g., "914 Embassy Habitat")
2. List files in that folder — look for:
   - Sale Agreement (PDF, Google Doc)
   - Legal Opinion / Due Diligence report
   - Cover letter or correspondence
3. Download and OCR the document (see `ocr-and-documents` skill — `pdf2image` + `tesseract` for scanned PDFs)
4. Extract the counterparty's details from the text
5. Use the address/location to confirm identity, then construct the WhatsApp message

**This session's example — Ravi Kumar (EH9 Seller):**

| Field | Value |
|-------|-------|
| Name | Ravikumar Kubasing Naik |
| Address | H. No. 01, 3rd B Cross, Gururaja Layout, Doddanekkundi, Bangalore — 560037 |
| PAN | ABWPN6886E |
| Aadhaar | 754125337406 |
| Buyer | Roshini Ranka (wife of Nishant Ranka) |
| Property | Flat No. 914, Embassy Habitat, Palace Road, Vasanthnagar |
| AoS Date | 05.03.2026 (GNR-1-06465-2025-26) |
| Sale Deed | Registered 2026 — Document No. GAN-1-00865-2012-13 (original 2012) |
| Seller's SBI Loan | A/C No. 64186230661 — outstanding at time of AoS |
| Seller's obligations pending | Loan closure NOC, tenant vacation NOC, society NOC, electricity transfer, original docs |
| Drive folder | `1rvnnl3168-YrvGQcUsD71aSYoVpmcChH` ("914 Embassy Habitat Title Documents") |
| Key Drive files | AoS PDF (`1ajkYQp7kWi14jGrt0uBdfpYG-lSXF2b8`), Sale Deed Google Doc (`1j195TcQyGCwzZteaUHHSEomJZlMQXMGs3DJLgJc83Fk`) |

**Note:** Ravi Kumar was not found in the contacts sheet. His details were extracted from the Drive documents.

## Drive Search Pattern for Property Transactions

```python
service = build_service("drive", "v3")
results = service.files().list(
    q="name contains '914' or name contains 'EH9' or name contains 'Embassy Habitat'",
    fields="files(id, name, mimeType, parents)",
    pageSize=20
).execute()
```

Then list the parent folder contents to find all transaction documents:
```python
results = service.files().list(
    q="'{folder_id}' in parents",
    fields="files(id, name, mimeType)",
    pageSize=50
).execute()
```

**Which auth path for Drive search:**
- Use `tools.gws_auth.build_service("drive", "v3")` for per-user OAuth — this also covers shared DRA Drive files the session user has access to
- `tools.gws_sa` does not exist (removed 2026-05-08) — never import it
- Only when a file is reachable *solely* under ndr@draas.com's identity, use `tools.gws_dwd_tools` (admin-gated DWD; returns data only, never a credential)
