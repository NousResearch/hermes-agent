# Vendor invoice-request messages (class template)

**Trigger:** the user dictates something like *"he has to raise two invoices … give me text
messages so I can send it in WhatsApp"*. Anbu asks for these for electrical / BESCOM / civil
vendors at Ranka sites. The deliverable is a **copy-paste WhatsApp text**, never a sent message.

## Message structure that works

1. Greeting — `Good morning Sir,` (see naming pitfall below).
2. One line of context: `We need your kind help with two separate invoices for the work
   completed at <Project>:`.
3. **Numbered invoice list, one block per invoice:**
   - `1. <item description — verbatim from the dictation>`
     `   Value: Rs.X,XX,XXX + GST`
   - `2. <item description>`
     `   Value: Rs.XX,XXX + GST`
4. The ask: **"raise both invoices separately (not clubbed into a single invoice), with GST
   shown clearly, and share the soft copies with us so we can take it forward for payment."**
   The word *separately* is the whole point of the request — it is usually driven by GST /
   input-credit or accounting treatment, so never soften it away.
5. Billing hygiene line: `please confirm the GST number and the company name to be billed, so
   we can keep it ready on our side.` **Do NOT guess the invoicing entity** (DRA Realty vs a
   Ranka entity) — ask the vendor to confirm, or leave it as a bracketed `[entity]` placeholder
   for the user to fill.
6. Close: `Thank you for your support.`

Keep it under ~1,000 characters so it stays a single WhatsApp bubble and the deep link (if you
generate one) stays manageable.

## Before clearing a claim: establish what it actually IS

The mirror-image job — the consultant/vendor is chasing money and NDR says *"he has sent me a
message about 1,50,000, to be collected from X or Y or something like that"*. **His one-line
description of a bill is a hypothesis, not the record.** Reconcile it before drafting anything:

- Read the WhatsApp thread (`wa_read_messages`, page back with `before=`) for the ask in the
  sender's own words, plus any photo of his workings (`wa_get_media`).
- Search Gmail for the counterparty, the amount, and the project. The trail is usually split:
  WhatsApp for the ask, mail for the memorandum.
- **Check for one invoice NUMBER raised twice.** 2026-10-27: invoice `VPA/26-27-65` (Rs 40,000 + GST
  = Rs 47,200) was first addressed to DRA Realty, then re-issued under the same number on
  Sevaganapalli Land Partners at NDR's own instruction ("SLP please"). That is ONE live invoice
  with a settled entity — not two bills, and not part of the Rs 1,50,000 he had also mentioned.
- **Say which part of his recollection the record does not support**, in one line, without
  implying carelessness — *"nothing in the record ties the 1,50,000 to South City; what does exist
  is a different Rs 40,000 invoice"*. Then state the one genuine gap (here: the 1,50,000 carried no
  entity at all).
- **Do not guess the entity. Put the question in the reply, not to NDR.** Ask the sender to state
  which entity each line is billed to and to raise the invoices on that footing — same rule as the
  "do not guess the invoicing entity" line above.
- Distinguish a **fee never invoiced** (blocked on an internal approval someone has to give) from an
  **invoice awaiting payment**. The email changes completely: the first clears the block and asks
  for the invoice, the second chases a payment run.

## Delivery

- **Fenced code block FIRST** — that is what he actually wants (he pastes it into WhatsApp
  himself). Plain copy-paste text, not a deep link.
- Link second, at most. If you generate one, call `whatsapp_link` with `platform` **omitted** and
  deliver the **raw `url`** as `[Open WhatsApp](<raw url>)`. Never paste `display_link` /
  `display_text` (MarkdownV2 backslashes) — see `whatsapp-link-delivery-format.md`.
- Never say or imply the message was sent. It is a generated link / paste-ready text only.

## MANDATORY: flag garbled details below the draft — do not silently guess

Voice dictation mangles exactly the tokens that matter on an invoice. Always list what you
are unsure of under the draft and ask for a one-line correction. Real examples:

- **Item descriptions.** *"procurement of 350m and CTm"* could be 350 m of cable, 350 sq mm
  cable, or 350 kVA — plus a CT (current-transformer) meter. The user later summarised the same
  invoice as *"procurement of meter"*. **Echo his wording verbatim in the draft** (`350 m and
  CT meter`) and flag the line; do not invent a spec.
- **Project name.** *"IRIE"* → read as **Ranka Iris**; flag it rather than silently substituting.
- **Person name.** *"Nandish AE"* → **Nanjesh, BESCOM AEE** (+91 99002 49798). If unsure, keep the
  greeting as a neutral `Good morning Sir,` so the draft is safe either way, and ask which spelling.
- **Amounts.** Repeat both figures in the flag list (`₹2,00,000` and `₹30,000`) so a digit slip is
  caught before he sends.

## Known instance — Ranka Iris, Oct 2026

Vendor: **Nanjesh (BESCOM AEE, +91 99002 49798)** — Anbu says "Nandish"/"Nandesh"; same person.
Project: **Ranka Iris** (Domlur 2nd Stage). Workstream: LT/HT panel, DG cable, meter installation,
BESCOM coordination.

Invoice pair requested (raise separately):
1. Procurement of material — `350 m` and CT meter — **₹2,00,000 + GST**
2. Electrical AutoCAD drawing prepared by his team — **₹30,000 + GST**

Anbu signed off on the wording with those two flags open; if he comes back with the corrected
item description, update this block rather than re-deriving it.
