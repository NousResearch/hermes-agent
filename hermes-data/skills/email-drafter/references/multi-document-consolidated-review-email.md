# Multi-document consolidated review mail — "put all of it in ONE draft"

Trigger (Bharat, 2026-10-26): *"analyse all three … put it under one single draft email … I just
want the draft and the summary of whatever things we have done."*

When several related documents about ONE subject already exist (a unit's allotment letter, cost
sheet and sale agreement; a project's three quotations; a filing's annexures), the ask is not
three mails. It is **one consolidated review mail plus a chat summary**.

## Rules

1. **One draft, one recipient, one body.** Numbered sections (1, 2, 3), each opening with the
   document name as a hyperlink — never three drafts, never a mail with three attachments when
   Drive links exist.
2. **A shared-facts block before the per-document content.** Whatever is common to all of them
   (unit, area basis, price chain, dates) gets stated once as a bullet block, so the reader does
   not have to reconcile three sections. Per-document sections then carry only the link.
3. **An explicit "points that need your decision" block, last.** Every inconsistency is written
   `from → to → fallback → "please confirm"`. The reader must be able to reply with one word per
   item. Open items (placeholders, KYC gaps, boundaries TBC) go here as a list, not as defects.
4. **Draft to the reviewer, not the counterparty.** Internal review of a customer pack goes to
   the person who signs off (here ndr@draas.com). Do not copy the allottee/customer on a review
   mail of drafts.
5. **Chat gets the summary, not the mail body.** After creating the draft, reply with: draft
   saved + location, the verified figures as bullets, the numbered defect list with the
   recommended fix, and one line offering to amend. No pipe tables, no re-paste.
6. **Never send.** `gws_gmail_draft_create` (or `gws_gmail_reply_draft` if it answers a live
   thread), then verify with `gws_gmail_draft_list` before reporting it as saved.
7. **HTML body.** Inline-CSS-free `<p>/<ul>/<ol>/<a href>` is enough; a plain-text body renders
   the links dead and the numbered blocks collapse.

## Worked instance

Ranka Amber Unit 203 — three drafts (allotment letter, ₹11,499 all-inclusive cost sheet,
agreement for sale) tied out against each other, one mail to ndr@draas.com with the rate-base
delta (₹10,443.88 vs ₹10,443.80 → ₹125.78 / ₹128.43), the 1,572.30 vs 1,380 sft area-basis gate
and the placeholder list. Full recipe, figures and the Docs-API read snippet:
`real-estate/unit-area-and-cost-sheet/references/unit-pack-cross-document-tieout.md`.
