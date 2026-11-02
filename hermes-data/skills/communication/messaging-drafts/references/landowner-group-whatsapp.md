# WhatsApp Messaging to DRA Landowner Groups

Register for drafting WhatsApp messages for landowner groups (e.g. Ranka North
Star / Allalsandra landowner group, and similar group chats per project). Used
heavily in the Sep–Oct 2026 North Star engagement — NDR dictates, then taps
deep links himself. These are EXTERNAL recipients (the group includes the
landowners), so the tone is warm-respectful, not the terse internal DRA tone.

## Core rules

- **Never use `phone` in the link** — the group has no number; the link must
  open WhatsApp's contact picker (`whatsapp_link` with only `text=`, platform
  `telegram`). NDR taps it and picks the group himself.
- **Long messages MUST be delivered as the `parts` array from `whatsapp_link`** —
  one link per part, send each part as its own Telegram message, numbered
  "Part 1 of X". Do NOT merge parts; do NOT hand-encode a mega-URL.
- **Bold `*HEADINGS*` for each section** — WhatsApp renders `*...*` as bold.
  Structure long messages as title line → greeting → section headers → bullets
  (`•`) → numbered points (`1)`) → closing. NDR asks explicitly for this
  ("heading being bold to bulleting, numbering, separating the paragraph").
- **No caption rule for group/external:** NDR's standing rule — landowner
  messages get NO bold caption line (`*Project: Subject*` is for internal DRA
  work messages). The exception is when he explicitly asks for one.
- **Apostrophes are safe to drop on long group messages** — he accepted
  "everyones consideration" (no apostrophe) when the alternative was breaking
  the link; when in doubt keep the apostrophe, it encodes fine.
- **Verify phone numbers via contact_resolver for 1:1 links; group links need
  no number.** For direct messages to individual landowners NEVER guess digits —
  resolve from the contacts sheet and strip spaces for the link (Vaidyanathan
  incident: a wrong number produced a dead link; correct was +91 93425 57120
  from the contacts sheet, not a remembered/dictated variant).

## Register and psychology (NDR's explicit asks)

- **Lead with the good news.** "Light at the end of the tunnel", "breakthrough
  we have all been working towards", "critical milestone reached". Positive,
  forward-moving framing throughout.
- **Acknowledge patience and trust** — "decades of trust", "I hold that with
  the deepest respect". Use generously.
- **Two-pronged objectives framing** (when relevant): 1) legal/registration
  milestone, 2) clearing the air between landowners. Keep each short and bolded.
- **"Ganesha Chaturthi greeting" pattern:** on auspicious days, open with a
  greeting and the serendipity framing ("this good and final confirmatory news
  has been received on this very day... my prayer to Lord Ganesha"), then
  "With that, let me share where we stand:".
- **Developers' accountability framing:** "As a developer, for me, you are
  collectively responsible as a group" — NDR wants the group to resolve
  landowner-internal disputes; DRA's position is "captured in the addendum,
  from our side there is no outstanding matter".
- **Direct quotes of the signed document** — e.g. "the signed addendum clearly
  provides for proportionate recovery" — he wants the legal position stated
  plainly and repeatedly.
- **Meeting logistics messages:** set the date, the time (10 AM at office),
  lunch arranged, transport to Sub-Registrar and back, overnight accommodation
  offer for out-of-town landowners (fly-in from Bombay), heirs to attend.
- **End with gratitude + signature** — "With gratitude, *Nishant Ranka*".
- **Let NDR approve before generating links** — he iterates on wording between
  drafts; present the text first, generate links only once he says go.

## Recurring content anchors (North Star / Allalsandra)

Khata vs surveyed area (53,148 e-Khata vs ~43,000 surveyed), 1.75 FAR milestone,
Addendum 2 (30-11-2024) registration, Kankshita's registered GPA (consulate
route), 2× TDR on 19-12-2017 BBMP lake-road acquisition, 67:33 sharing model,
7% sales & marketing fee, refundable ₹2.90 Cr proportionate recovery, same-day
registration + puja/groundbreaking at sanction. Draft each message against the
LATEST state (check Gmail/Drive for the newest email or draft; never recite a
stale figure like the old 43,000/53,089 framing).

## Pitfalls

- Do NOT invent details (dates, units, beneficiaries) the user didn't state —
  put them in only when NDR dictates them or they are verified from the signed
  documents (e.g. exactly "5 decree-beneficiary units (~1,500 sq ft each SBA)",
  beneficiaries Ashwathamma / Murali Mohan / Vijaya @ Thayamma / Shwetha /
  Guruswamy).
- If a recipient replies with a correction or new constraint (e.g. "wife will
  not sign, only GPA"), the next group message must address that precisely and
  respectfully — NDR wants the counterparty's concern acknowledged head-on.
- For 1:1 landowner messages, use the canonical spelling from the JDA/addendum
  (Kankshita Swaminathan), not the STT variant (Kanshika/Kaushika).
- When the Sub-Registrar/consulate process contradicts, be transparent with the
  group about the delay and the process steps; NDR prefers honesty over
  glossing (GPA process shift of a week to 10 days was communicated plainly).