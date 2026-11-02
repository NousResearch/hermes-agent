# Vendor quotation gap analysis vs our own render + spec, then the gap-closure email

Class of task NDR dictates as: *"check if the quotation already covers everything we need to
deliver the rendered office space we sent them — whether the materials they talk about, the
methodology, the tables, chairs, wall treatment, glass frontage, the steps, everything. If anything
is missing or not clear, we need a follow-up email itemising the gaps."*

Worked 2026-09-15/16 (AR Portable Cabin, quote ref `ARPC/PMO/7042-1/26-27`). Same family as
`fee-proposal-comparison.md` (architect fee revisions) — answers "does this quote match what we
asked for, and what do I reply".

## Method

1. **Pull the quote PDF off the email, not Drive.** `messages().attachments().get(userId='me',
   messageId=..., id=...)` with the base64 padding guard (`+ '=' * (-len(data) % 4)`), then
   `pdftotext` the saved PDF. Vendor quotes are text PDFs; their brochure may be image-only
   (no text layer) — don't waste a cycle trying to read it, ask for the numbers in the quote.
2. **Read OUR spec back, section by section** — the spec PDF we originally sent is the yardstick.
   A long spec (this one was 46 numbered sections) yields far more gaps than skimming the render.
3. **Bucket every line into three lists:**
   - **Covered well** — name it, so the reply is not a wall of complaints.
   - **Missing / explicitly excluded** — quote the vendor's OWN T&C line as the evidence
     ("loose furniture and ACs are not included", "freight extra at actual", "crane to be provided
     by client", "LAN extra"). Their own terms are the strongest possible citation.
   - **Contradicts the design** — e.g. quote offered plywood + cement-fibre + paint where the design
     calls for a vertical wood-slat feature wall; vinyl carpet where the design wants wood-look
     SPC/LVT plank.
4. **Hunt the quote's internal contradictions** — the highest-value findings. On this quote
   "Transportation" was ticked as *included* on the option sheet while the T&C said freight was
   *extra at actual*; the standard cabin spec carried toilet fittings while the approved design has
   no toilet. Both went into the reply as direct questions.
5. **Mine the vendor's separate clarification sheet** for timeline / methodology / references claims
   — that answer doc (not the quote) is where "we can deliver in 60 days", "prefab structural over
   container", "95% factory, only AC at site" live. Carry them into the handoff pack.

## Dimensions to check on ANY prefab/container office quote

Furniture (lounge sofa, coffee table, rug, meeting table, chairs) · AC supply + install + concealed
routing · TV/digital display + concealed power & HDMI · branding/logo + exterior signage · printed
project imagery + location map · entry steps/ramp/landing · wall panelling + the feature wall ·
flooring plank spec · glass frontage (fixed vs openable, panel count + dimensions, glass thickness,
frame finish, entrance door + hardware) · false ceiling + lighting scheme (colour temp, shelf LED
strip, no exposed dots) · built-in joinery (display/storage wall, credenza, pantry counter, carcass
material + hardware) · electrical point schedule · data/network points · itemised BOQ with brand,
model, quantity, unit rate, labour, transport, taxes, warranty · exclusions (GST, freight, crane,
DB/power at site, site civil) · a dated delivery schedule (design → factory → transport → install)
· 2–3 client references and 2–3 photos.

## Reply shape (HTML)

Thanks for the prompt quotation → restate that the requirement is a turnkey office that reproduces
the render's look and feel as closely as possible → numbered sections: (1) scope items missing and
must be added, (2) specifications that must match the render, confirm or itemise, (3) commercial &
methodology clarifications (timeline, transport, crane, plinth, toilet, itemised BOQ, exclusions),
(4) references & proof of capability, (5) our commitment — *"once we receive the itemised quotation,
the references/photos and a firm 90-day timeline, we will confirm within 48 hours and place the
order"*. A lump sum on the top option is explicitly not enough to approve and pay.

## Pitfall — the vendor may reply from a different address than the one you mailed

The AR thread's quote came in from `info@arportablecabins.com` but the person later engaging was
Sayed from `sayed@arportablecabins.com`. Search the whole domain (`q='arportablecabins.com'`), not
the single address, and confirm which mailbox actually carries the human replies before drafting.
Also: the vendor's first "please find attached quote" mail arrived with the quote **not actually
attached** — verify attachment presence on the message payload before assuming the quote exists.
