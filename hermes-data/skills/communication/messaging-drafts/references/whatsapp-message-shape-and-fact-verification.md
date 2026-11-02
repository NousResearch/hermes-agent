# WhatsApp message shape — what NDR actually asks for

Verified 2026-09-25 (DR Villa / Ranka 99 brief to Roshni; and the Ranka-99/Nostos
project-name family).

## The dictation instruction is about FORM as well as content

NDR's voice memo for a recipient-facing brief is explicit, and the form half must be
taken as literally as the content half:

> *"put it in a nice easy to read manner, limiting it to about a hundred words, 150, 200
> words, tops. To capture everything neatly and clearly, formatting it appropriately,
> point-wise, with WhatsApp markdown that has all the bolding, pointing, everything,
> right, bulleting, whatever is required."*

## The working shape

- **WhatsApp markdown, NOT Telegram markdown.** Bold is `*single asterisk*` — never
  `**double**`. Bullets are `•` or `-`. Numbered runs as `*1. Heading*` then `•` bullets.
- Pass the text to `whatsapp_link` with those characters already in it. **Do not
  pre-escape or strip anything** — the tool handles `&`, `#`, `(`, `)`, `'`, `+`, `—`
  and newlines correctly. Stripping is what breaks the link.
- **Lead with a one-line bolded title**, then ALL-CAPS bold block headings
  (`*DEVELOPMENT — CLOSED.*`, `*VALUATION — TWO PATHS*`), then bullets. A long brief
  with no headings is unreadable on a phone.
- **Respect the word budget.** "150, 200 words tops" is a CAP, not a target to exceed
  for completeness. If the dictated content genuinely cannot fit, say so and propose
  which blocks to cut — do not silently deliver 600 words because the dictation was
  600 words.
- **Split parts go out as SEPARATE messages.** Over ~4096 chars the tool returns
  `split: true` with a `parts[]` array, each a complete link. Send each as its own
  Telegram message via `send_message` — never concatenate, because Telegram's own
  splitter will cut a link in half. Label them (`*Part 1 of 2 …*`) so he knows to tap
  both.
- **Also render the text in the reply**, not just the links — he reviews wording before
  tapping.
- Say plainly it is a **generated link**, never "sent" / "delivered".

## Verify asserted FACTS in the dictation against the source artifact BEFORE building the link

NDR's dictation supplies the message content, but it can also carry a **factually wrong
identifier**. Unlike a voice-garbled name, this arrives clean and plausible — it is an
error of recall, not of transcription, so nothing in the audio signals doubt.

**Worked case (2026-09-25).** He dictated a WhatsApp for the DRA Realty group
congratulating Prakash on *"successfully chasing and obtaining the APF for Ranka
**Iris**"*, and a follow-up to Manohar. The APF certificate the agent had filed an hour
earlier — and the entire ICICI thread — was **Ranka Amber**:

- certificate: *"This is to certify that **RANKA AMBER** has been approved by ICICI Bank"*,
  APF `KA/BAN/26/61572`
- ICICI's own covering mail (Kshitij Saurabh): *"the attached approval letter for
  **RANKA AMBER**"*
- thread subject: *"Greetings from ICICI Bank- **Ranka Amber** - Whitefield"*
- zero ICICI APF hits for Iris; Iris's last pre-approvals were 2016 (SBI/HDFC/Axis)

**Rule:** when the dictated message asserts a fact about a specific artifact — APF /
approval number, RERA number, sanction, payment, project name — **verify it against the
artifact itself and the live Gmail thread before calling `whatsapp_link`.** Then generate
with the corrected value and **flag the discrepancy up front** so he can overrule in one
message.

Never silently honour a project name you have evidence is wrong. A message that misnames
the project to staff, a landowner or a bank is a credibility cost — and it will cost a
correction round anyway. The two-message cost of asking is strictly cheaper than the
wrong-name cost of committing.

**Worked case 2 — an uncertain DATE (2026-10-05, SCA / Durga Puja stall, Bharat).** The tracker
carried the puja as **15–21 Oct**; the organiser's own sponsorship deck said **16–21 Oct**. Rather
than asserting either one in the outbound WhatsApp, the message asked
*"Please confirm the dates — 16 to 21 October 2026?"*. An asserted wrong date in a message to the
counterparty is the same credibility cost as an asserted wrong project name — and here the
counterparty is the only authority who can settle it. **Rule: when the user's own record and the
counterparty's document disagree, put the disagreement IN the message as a question; keep asking
for facts the counterparty owns (dates, rates, permissions, inclusions) as a numbered ask-list.**

**Worked case 3 — a garbled CC recipient in an EMAIL ask (same session).** "keeping
**Echanumdeshewari** in CC" resolved to **Eshwari Chamundeshwari <echamundeshwari@draas.com>**
only by mining the recurring thread, not the name. Method + the draft-with-attachment recipe:
`references/gmail-draft-with-attachment-mime.md`.

This is the messaging-side companion to `contact-phone-lookup`'s rule on the same
project-name failure family (`Nostos`→`Northstar`, `Iris`→`Amber`).

## Channel choice: offer a written message, not a call script

For outreach to an external party (vendor, event committee, channel partner, society office,
landowner group), the house preference is a **WhatsApp message the user can forward**, not a call
script. Bharat, verbatim, when offered "call script / message": **"Better what's app message."**
Lead with the WhatsApp draft and treat a call script as the fallback. Practical shape that worked:
one self-contained forwardable message, ask-list numbered and capped at ~5 items, related asks to
the same counterparty folded in (the stall booking AND the gate-branding rate went in one message
because they are the same committee and the same venue), word budget ≈150–200, and an explicit
offer to "visit the site or connect on a call" as the closing line.
