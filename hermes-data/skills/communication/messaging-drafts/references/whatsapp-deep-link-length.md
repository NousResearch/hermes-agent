# WhatsApp Deep-Link URL Length — Auto-Split Can Still Break Links (verified 2026-09-22)

## Symptom (NDR, 2026-09-22)
A 2,474-char WhatsApp message (Anbu task follow-up) run through `whatsapp_link_tool`
came back `split: true` with 2 parts:
- Part 1 URL = **2,166 chars** — past WhatsApp's ~2,048-char URL cap → link fails / truncates
- Part 2 URL = 1,508 chars — borderline

On top of the over-cap URL, both links were **stacked in one Telegram message**, so the
platform's own splitter cut the second link mid-URL. NDR's report: "the links are not
working, they continue to break across messages."

## Root cause
The tool's auto-split is character-count based (~1,400 text chars), NOT URL-safety based.
A part can still URL-encode past 2,048 once bullets, em-dashes (—), and special chars
are percent-encoded. Long links also break when multiple are concatenated in one
Telegram message.

## Fix recipe (what worked)
1. **Pre-chunk the text yourself** at section/item boundaries before calling the tool —
   each part ≤ ~1,000 chars of raw text (that encodes to ~1,300–1,500 URL chars, safely
   under 2,048). Never rely on the tool's auto-split for long briefs.
2. Call `whatsapp_link_tool({"phone": "...", "text": chunk})` **once per chunk**.
3. **Verify every returned URL**: `len(p['url']) < 1900` for each part. If any part is
   over, split that chunk at a sentence boundary and regenerate.
4. **Deliver each part as its own Telegram message** — never concatenate multiple deep
   links in one message (the platform splitter will cut the trailing link).
5. When re-doing broken links: regenerate fresh, don't re-send the old URLs, and tell the
   user what was wrong (over-limit URL length / single-message stacking).

## Verification output (working example, 2026-09-22)
Same 2,474-char message split deliberately into 3 semantic parts:
- Part 1: 873 chars text → URL 1,317 chars (Udaya/Oasis/Iris)
- Part 2: 996 chars text → URL 1,466 chars (Northstar/feedback)
- Part 3: 601 chars text → URL 933 chars (daily updates/budget/closing)

All three delivered as separate messages, all worked.