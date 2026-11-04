# Reply Formatting Preferences for Commercial / Negotiation Emails

NDR's explicit instructions for detailed replies to counterparty proposals
(verified 2026-09-09 on the PropLion/Ranka Amber sole-mandate negotiation),
given AFTER reviewing a draft. Apply to any "detailed reply" / "respond to
him fully" email against a counterparty's counter-proposal.

## Rules

1. **Mirror the counterparty's formatting.** If they bolder key phrases,
   use numbered points, and bullets in their reply, do the same in ours —
   bold the key terms/rationales, numbered sections, bullets for rate
   ladders. Purpose: "so our points are very clear to him."

2. **Complete paragraphs — never wrap mid-sentence.**
   NDR verbatim: "please don't truncate the lines and put them into new
   lines in the middle of a sentence... Complete the paras and new line
   characters only between the paras."
   - HTML drafts: one point per `<p>` block (automatic).
   - Plain-text drafts: write each point as ONE flowing paragraph; blank
     line only between points.

3. **Machine-readable term structure.** Structure the reply as numbered
   points that mirror the counterparty's own numbering where possible
   (1. Pricing, 2. Performance gates, ...), so side-by-side review is easy.

## Handling garbled dictation (material terms)

When the dictation is ambiguous on a MATERIAL term, don't silently drop it
or invent a resolution — draft your best interpretation and FLAG it in the
delivery message. Real cases from the PropLion reply:

- "you keep 500 rupees towards water meter, backup power..." was garbled;
  the draft left the all-inclusive definition clean and NDR was asked what
  he wanted for the ₹500 buffer.
- "completion targeting in March" — the draft assumed March 2027 (mandate
  ends Feb 2027) and the assumption was flagged for confirmation.

The user reviews the draft before sending, so a flagged open question is
cheaper than a wrong assumption baked into a sent email.

## PS / footnote pattern

When NDR wants a critical commercial qualifier added late, he often asks
for it as a "PS" after the sign-off. Put it as its own bold-led paragraph
(**PS — <topic>. <Full sentences>**) with the same no-mid-sentence-wrap
rule. Example (2026-09-09): PS on brokerage basis — computed strictly on
basic rate + car park; no brokerage on GST, pass-through water/electricity,
or optional client upgrades; invite the counterparty to raise anything else.

## Draft lifecycle note

The threaded-reply draft is created via raw MIME (`EmailMessage` +
`add_alternative(html)`, In-Reply-To + sanitized References, `threadId` in
the draft body) and verified with `drafts().get` + `drafts().list`.
Update-in-place = DELETE the old draft resource id, then recreate with the
new body (same threading headers); re-verify DRAFT label and presence in
`drafts().list()`.