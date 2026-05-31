# BlackDuckAi Finance Document Sync Workflow

## Purpose

This note mirrors the non-sensitive Obsidian wiki workflow for BlackDuckAi finance document intake. It documents how Red should keep local archive files, Google Drive, Google Sheets, and review notes aligned without exposing secrets or sensitive identifiers in GitHub.

## Scope

Use this workflow when TOP sends finance/accounting documents such as:

- invoices or receipts
- payment slips
- accounting service fees
- VAT / WHT support documents
- company compliance documents

## Storage model

1. Preserve the original image/PDF in the local finance archive.
2. Create a paired `_ocr.txt` text file with extracted visible fields.
3. Sync both the evidence file and OCR file to the matching Google Drive company/month/category folder after TOP approval.
4. Update the company Google Sheet:
   - `Document Inbox` receives one row per source document.
   - the relevant category tab receives the transaction row, e.g. `Service Fees`.
5. Update Obsidian/Wiki with only durable, non-sensitive workflow facts.

## KST example pattern

For KST accounting service fee documents:

- Company: KST / KIRA SOLUTION TECH
- Category: `Service_Fees`
- Sheet tabs: `Document Inbox`, `Service Fees`
- Pair invoice/receipt evidence with the transfer slip when the net paid amount matches the invoice after withholding tax.
- Keep local paths and Drive URLs aligned in the Sheet.

Amount handling example:

```text
service fee: 13,000 THB
withholding tax: 3%
net payment: 12,610 THB
```

## Safety rules

- Do not commit tax IDs, bank account numbers, slip reference values, tokens, browser session values, or raw credentials.
- Do not paste full original OCR when it contains sensitive identifiers; summarize visible fields only.
- Do not share Drive folders externally or send accountant packages without explicit CEO approval.
- If VAT or tax fields are not clearly visible, leave them blank and mark review required or add an accountant note.

## Verification checklist

- [ ] Local original files exist in the correct company/month/category folder.
- [ ] OCR sidecar files exist and are readable.
- [ ] Drive upload returned stable file URLs for every file.
- [ ] Google Sheet `Document Inbox` rows include matching local paths and Drive URLs.
- [ ] Category tab includes the canonical transaction row and accountant note.
- [ ] Obsidian/Wiki note was updated without sensitive identifiers.
- [ ] GitHub docs mirror only non-sensitive workflow facts.
