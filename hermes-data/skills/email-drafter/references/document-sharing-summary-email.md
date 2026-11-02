# Sharing documents with an external counterparty — the SHORT email pattern

Use when the email's job is to hand over documents (title report, DD pack, PE
file, drawings, valuation) to an external counterparty — a co-investor's
diligence team, a JV partner, a lender, a consultant.

## The correction that defines this pattern

NDR, on a draft that shared two due-diligence documents with a co-investor:

> "the email is too extensive. You don't need to give that many details. Maybe
> just share the documents with him and give him a short summary of what is
> clean about this and what are the other documents we have requested and why.
> A very short summary."

He had the full internal analysis in hand and did not want it in the email. The
internal report is for him; the email unblocks the counterparty. Two different
documents.

## Shape (aim 200–280 words total)

1. One line of greeting / acknowledgement of the thread.
2. **Numbered document list**, each line: what it is, its date, and a view link
   with access already granted. Nothing else per line.
3. **One short paragraph — what is clean.** The single strongest finding, in
   plain words, with the numbers that prove it (dates, extents, multiplier).
4. **Four to six short lines — what we have asked for, and why.** One line per
   item, one clause of reason. This is a summary, not a requisition table.
5. One line: the other side's request is being pursued / will follow.
6. **The ask:** "Please go ahead and start your own due diligence now."
7. Sign-off.

## Do NOT include

- Head-by-head comfort grading or percentages
- A ranked red-flag list
- The internal reasoning, legal citations, or statute references
- A per-item requisition table with "from whom / why" columns (that belongs in
  the counterparty extract of the internal report, or in a WhatsApp to the
  broker — not in this email)
- Clause-by-clause answers to the counterparty's request list (compress to one
  line if it must be mentioned at all)

## Mechanics

- **Grant view access BEFORE drafting**, then say so in the email. Verify with
  `drive_permissions_list` first in case access was already granted earlier.
- **Thread it.** Use `gws_gmail_reply_draft` on the counterparty's latest message
  so In-Reply-To/References and threadId are correct. Do not start a new mail.
- **Only one draft in a thread.** If a longer draft already exists from an
  earlier attempt, delete it (`gws_gmail_draft_delete`) before creating the
  replacement, so NDR is not choosing between two versions.
- **Never send.** Draft only. Tell him the draft is in Drafts.
- If a document is still awaited (e.g. a certificate being re-called), say so in
  one line and say it will follow. Do not build a section around its absence.
- Do not attach the internal report to the email unless he asks for it
  specifically — share source documents, not our analysis.
