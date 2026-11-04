# Replying point-wise to a long multi-item review mail

## Trigger

The boss (NDR) sends one long mail that reviews **many discrete items at once** — thirteen
vendor invoices, a checklist of approvals, N line items of a scope — and lists what is wrong
with each. The reportee (Anbu) then says, in some form:

- *"Get me the mail from X regarding the 13 invoices — get me the summarized version so that I
  have to reply to the mail."*
- *"I want to respond for each point separately — give me that 13 points so that I can respond
  to it."*

That second sentence is a **format instruction, not a request for more detail**. It means the
first summary was shaped wrong.

## Rule zero — keep the sender's own order and numbering

**Do not regroup.** Not by vendor, not by severity, not by gap type, not into
"themes / systemic / money". The reportee is going to sit with the original mail open and answer
item 1, then item 2, then item 3. Any regrouping forces him to re-map your structure onto the
sender's, which is exactly the work he asked you to remove.

- Preserve the sender's item numbers **and** his item labels (invoice no / file ref / clause).
- If two items share a vendor or share a gap, they still get **two separate points** — say
  "as above" inside the second one rather than merging them.
- Group-level commentary (the sender's own "systemic" section) goes **after** the numbered
  points, never in place of them.

Anbu corrected this in-session: a first pass grouped the thirteen into KPC / KCCB / Lehan /
RGR / I Vak / KTA buckets with a "money he specifically cut" roll-up. Useful to read, useless to
reply into. He asked for the flat 1–13 again.

## Deliverable shape — three artefacts

### 1. Per-item reading sheet (in the chat reply)

One block per item, fixed field order, so the blocks are scannable in sequence:

```
**7. INV-3052 · Lehan Blocks · ₹23,600 · 31-Aug · NO PO**
Qty tallies (500 nos, receipt #55103891).
Flagged: no PO, ₹40/block unverified · completion proof is the invoice PDF itself ·
quality proof is a WhatsApp photo, no block test certificate.
```

- **Header line:** item ref · counterparty · amount · date · PO/WO status. The PO/WO status
  is the single most useful field — it instantly sorts "paperwork fixable" from "no order
  existed at all".
- **The clean half first, in one line.** What the sender explicitly accepted (rates match to
  the paisa, quantity reconciles to the receipt, credit note already correct). The reportee
  needs to know what he does *not* have to argue about — omitting it makes every item look
  equally fatal and he will waste the reply defending settled points.
- **Then "Flagged:"** — every gap, compressed to a clause, separated by `·`. Keep the sender's
  own numbers verbatim. Never re-derive an amount; quote his.

### 2. Copy-paste skeleton with blanks

A fenced block the reportee can paste straight into his reply box and type after each dash:

```
1. 26-22298 (KPC 3,23,440) —
2. 26-24388 (KPC 1,95,685) —
...
13. 016 (KTA 3,27,745) —
```

Cheap to produce, and it is the thing he actually uses. Include it every time the count is
above ~5.

### 3. Threaded Gmail draft, rebuilt as N numbered paragraphs

Each point: **item ref + amount, then the corrective action in one or two sentences.** Restate
the gap in half a line so the mail reads standalone, then the fix. Then:

- the sender's cross-cutting/systemic items as their **own** numbered list ("being fixed across
  every record, not just these thirteen");
- a closing line with the resubmission commitment and any approval you are asking him for
  (e.g. the two PO amendments), because that is the only way the ceiling problem stops
  recurring.

Leave the resubmission date as a literal `[DATE]` placeholder and name it in the chat reply as
one of the "only you can set" items. Same for any commercial choice that is not yours to make
(accept the deduction vs push back, hold the excess vs amend the PO).

## Extraction recipe

Read the mail in **slices**, not one truncated print. A 13-item review runs 10–12k characters;
a `print(t[:6000])` silently loses the tail — including, typically, the systemic section and
the closing instruction. Print `t[:6000]`, then `t[5500:14000]`, overlapping, and confirm
`TOTAL CHARS` first so you know how many slices you owe.

For each item, capture in this order: **ref → counterparty → amount (and accepted amount if
reduced) → date → PO/WO ref and its value → what is clean → each numbered gap (G1..Gn) → the
exact rupee figure attached to each gap.**

Then compute, and **check the arithmetic before quoting it**:

- total at stake across all items;
- subtotals grouped by PO/WO (this is what exposes an over-run);
- a separate list of "money he specifically cut" — the individual deduction figures. These are
  the numbers the reportee will be asked about first, and they are scattered through the body.

## Pitfalls

- **Structure change ⇒ delete and re-create the draft, then prove only one remains.** Rebuilding
  a single-block draft into thirteen points is a wholesale replacement; `draft_reply_create`
  has no update mode. Delete the old draft id, create the new one, then assert:

  ```python
  r = svc.users().drafts().list(userId='me', maxResults=20).execute().get('drafts', [])
  print(sum(1 for x in r if x['message']['threadId'] == THREAD_ID))   # must be 1
  ```

  Two near-identical drafts on the same thread in the reportee's Drafts folder is the failure
  mode. (For same-shape revisions of an NDR draft, `draft-revision.md` says update in place —
  that rule is for edits, this one is for re-structures.)
- **Never invent the corrective action where a decision is required.** "Vendor to redo at the PO
  discount, failing which I accept the reduction" is safe — it is the standard remedy. "I will
  pay the ₹42,381 excess" is not; that is the reportee's/NDR's call. Mark decisions as
  decisions and list them in the chat reply.
- **Do not soften the sender's language.** If he wrote "not a GST invoice", the summary says
  "not a GST invoice" — not "documentation gap". The reportee is answering a specific
  allegation.
- **A wrong-document gap is worth calling out on its own.** "Quality proof is the chamber
  certificate, but the bill is for seals" is a different failure from "no certificate" and the
  fix is different.
- **Watch for an item the sender himself downgraded.** Several invoices had "accepted ₹X after
  credit note" — the reduced figure is the live one; quoting the gross invites a pointless
  argument.

## Reference facts — Ranka Oasis 13-invoice review (Oct 2026)

- NDR's mail: **"Ranka Oasis — 13 invoices from Anbarasan (26-Oct): gaps he must fill before
  payment"**, 28-Oct-2026 18:02, from ndr@draas.com. Thirteen Kelsa bills posted 26-Oct 6:40 pm:
  eleven material bills + the survey bill + the KTA JCB bill.
- Total at stake **₹16,15,410**. By PO/WO: KPC on PO 757 ₹10,42,109 · KCCB on PO 756 ₹88,146 ·
  Lehan blocks ₹51,920 · I Vak on WO 760 ₹64,900 · KTA (no PO) ₹3,27,745 · R G R Traders (no PO)
  ₹40,590.
- Background orders: PO 757 = SLP/OASIS/PO/2026-27/005, 04-Aug-26, KPC Distributors, ex-GST
  ₹8,23,757.36 / gross ₹9,72,034. PO 756 = SLP/OASIS/PO/2026-27/004, 27-Jul-26, KCCB, ex-GST
  ₹1,86,600. WO 760 = SLP/OASIS/WO/2026-27/001, 10-Aug-26, I Vak, ₹5,500 per full working day.
- The five systemic faults (fix once, most bill-level gaps vanish): every material receipt stuck
  at "Arrived At Site" (never moved to Quality Checked / Accepted, so no GRN is complete under
  PO Cl.11); Product Quantity field filled with prose instead of numbers; one photo reused as
  both proof of quality and proof of completion (7 of 13); six of thirteen with no PO/WO at all
  (26-26275, 26-26155, INV-3052, INV-3053, 432, 016); no test certificates for hume pipes,
  solid blocks or TMT bars.
- Specific deductions NDR identified: ₹11,865 chamber billed but "not received" (26-22298) ·
  ₹1,248 rate difference at +₹176/no (26-24981) · ₹31,223 of seals not on PO 757 · ₹42,381
  aggregate over-run on PO 757 (₹10,14,415 invoiced vs ₹9,72,034) · ₹6,195 for 7 cracked 150mm
  pipes still billed (130/2026-27) · WO 760 recorded value ₹6,490 against ₹2,07,680 invoiced
  (yet-to-invoice −₹2,01,190).
- Kelsa record links live in the mail itself as `https://kelsa.io/516/leads?current_item_id=…`
  — one per invoice; do not re-derive them, and do not paste thirteen long URLs into a Telegram
  reply.
