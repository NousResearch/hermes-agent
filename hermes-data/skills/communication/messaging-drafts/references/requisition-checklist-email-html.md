# HTML Requisition / Document-Collection Checklist Emails

**When:** a legal/title opinion (or bank / advocate requisition) is conditioned on a
document list, and an assignee must collect each item, tick it off, and commit a
delivery date. Verified 2026-09-04 on Gunjur Sy 40 (BRK Law Preliminary Title
Report, 22-item checklist, assignee Rahul = Vinod Kumar Das, task P-119).

## Workflow

1. **Pull the authoritative requirements list from the opinion itself.**
   - Download the opinion attachment (Gmail `messages().attachments().get` via
     `messageId` + `attachmentId` from `payload.parts[].body`).
   - `pdftotext -layout` first — scanned reports often carry an embedded text
     layer and OCR'able text this way; if empty, `pdftoppm -r 300` + vision.
   - The opinion's "documents to be furnished / notes" section IS the checklist;
     do not summarize it away. Read it item by item.

2. **Build the HTML email** around NDR's explicit requirements:
   - `MIMEText(html, "html", "utf-8")` — never plain text for a 15+ item list.
   - `☐` checkboxes, numbered items grouped A–F under colored section headers.
   - **Golden-rule callout box** (yellow): for EVERY item, deliver either the
     actual document OR a department endorsement that the record is not
     available / does not exist — no gaps.
   - **Deadline callout box**: commit a date against every item; deliverables due
     "no later than Friday, <next Friday's date>" (compute from NDR's "5 days /
     next Friday" phrasing — NDR sometimes says both; anchor on the named day).
   - Keep exclusions explicit (e.g. "public notice + original verification NOT in
     this list — deferred to the developer's detailed opinion").
   - Sign-off "Thanks, Nishant".

3. **Update the existing draft IN PLACE** (`drafts().update` with rebuilt raw
   MIME, same DRAFT_ID) rather than creating a duplicate — see messaging-drafts
   umbrella pitfalls. Then verify: `drafts().get` → headers + `payload.body.data`
   (check `body.data` BEFORE walking `parts[]`); assert checkbox count, deadline
   string, and every group keyword present.

4. **Mirror to WhatsApp + tracker** (same session pattern):
   - WhatsApp: same checklist, numbered, bold section headers, `whatsapp_link`
     tool, split-parts delivery per the `messaging-drafts` → `references/messaging-links-workflow.md` (one bubble per
     part, short labels).
   - Tracker (personal-task-tracker): Deliverable column = compressed full list;
     **Committed Date (K) = the deadline NDR named**; append dated note with the
     both-doc-or-endorsement rule.

## Pitfalls

- Opinion caveat lists mix executable collection items with deferred items
  (public notice, original verification). Keep deferred items OUT of the
  assignee checklist but NAME them so he knows they're not his.
- Report names are authoritative over earlier drafts (the Gunjur report says
  "B. Thippaiah / T. Thippaiah"; an earlier internal draft had said "D.
  Thippaiah" — flag the discrepancy, use the report).
- Verify the draft's checkbox count equals the intended item count — one missing
  line = one uncollected document.