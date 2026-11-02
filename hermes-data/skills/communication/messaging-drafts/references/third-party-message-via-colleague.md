# Drafting when the message must come from SOMEONE ELSE

**Two distinct shapes — pick the right one before drafting.**

- **A. Instruction to our own colleague** (this file): we send X an internal instruction plus a
  ready-to-use block that Y can copy. Deliverable = an *email to our colleague*.
- **B. Message authored for a third party, carried by an intermediary**
  (`references/third-party-message-via-intermediary.md`): the deliverable is a *WhatsApp link*
  whose body is written **in the third party's voice** (a senior official / MP / regulator) and
  sent to an intermediary who forwards it as-is. Different register, different fact-hygiene
  rules — notably, never state a report or approval as completed when only a request exists.

NDR frequently dictates a message that **a third party has to send** — a consultant, a vendor, an
architect, an external advisor. He phrases it as *"write to X asking him to get Y to write to them
that …"*, or *"ask X to tell Y to send this"*.

**The deliverable is NOT an email to Y, and NOT an email to the counterparty. It is an internal
instruction email to our own colleague X, containing a ready-to-use block that Y can copy and send.**

Getting this wrong in either direction is a real failure:
- Sending it to the counterparty leaks an internal instruction and makes the company look like it is
  chasing its own vendor.
- Sending it to Y directly strips the internal owner (X) of the relationship, and NDR explicitly asked
  for X to drive it.
- Drafting a *description* of what Y should say (instead of verbatim text) pushes the composition work
  back onto X and usually produces a vague email that gets ignored.

## Trigger phrases
- "write to <colleague> asking him to get <third party> to write to them / to send them an email"
- "ask <colleague> to talk to <person> and get him to reply"
- "get <person> to write back to <counterparty> that …"
- "I want to write to <colleague> asking him to …"

## Resolution order — do this BEFORE drafting

1. **Find the real thread, not the spelling.** NDR's voice mangles both names and subjects
   ("Harsamran" → Harsimran; the actual subject line may itself be misspelled — the Godrej one is
   literally *"Bux ranka quatation"*). Search by **sender domain** (`from:godrejventure.com newer_than:20d`),
   by person (`manju newer_than:40d`), or by the commercial term (`4.5 FAR`, `advance`) — never by the
   spelling NDR used out loud.
2. **Read the WHOLE thread**, oldest to newest, including who was cc'd on each message. Establish: who
   authorized what, who is waiting on whom, and which message the new draft should hang under.
3. **Pull the source document** (the quotation / PO / approval PDF attached in the thread) and extract the
   hard numbers from it. Never carry amounts from voice into a draft. The Godrej case: NDR said
   "3.5 lakhs advance"; the quotation confirmed ₹3,50,000 out of ₹5,90,000 and showed the DWG + advance
   are a **joint** starting condition, which changed the draft's wording.
4. **Check Sent and Drafts** for an existing artifact on that thread before creating anything — NDR often
   forwards the thread first and only later asks for the follow-up.

## Anatomy of the draft

- **To:** the colleague (our own person). **Cc: none.** An internal instruction must not reach the
  counterparty or the consultant.
- **Threading:** reply on the existing thread — pass `threadId` to `drafts().create()` **and** set
  `In-Reply-To` / `References` to the colleague's own forward, so the counterparty's authorization sits
  directly below the instruction.
- **Subject:** keep the thread's existing subject verbatim (including its typos) so it threads.
- **Body, four parts:**
  1. One line of context — *who* authorized *what*, and that they asked what is needed from their side.
  2. The numbered asks, **anchored to the third party's own quotation/contract terms** ("as per his
     quotation dated 04-09-2026…"). This is what makes the ask unarguable.
  3. **A verbatim block the third party can copy-paste.** This is the highest-value part of the email —
     it converts an instruction into a one-click action. Write it in the third party's voice, addressed
     to the counterparty, first-person, with a salutation and sign-off.
  4. A nudge to send it immediately, plus who to keep in copy.
- Sign as NDR.

## After creating the draft

- **Read the draft back** and verify To / Cc / Subject / `threadId` before reporting — a reply created with
  `threadId` can still pick up unexpected recipients if the headers are wrong.
- **Deliver it in the same turn.** Drafts are inert; withholding one behind a confirmation round-trip is a
  known correction (see `confirm-before-actions`).
- **Report anomalies separately, to NDR, not in the email.** Real examples from the Godrej case:
  the quotation was issued in the name of a different entity ("Excel Consulting Services", S.K. Suresh Babu)
  than the person who emailed it (Manju DS) — a payment-recipient question worth raising before money moves;
  the quotation misprinted the client's own GSTIN; and the colleague had a second address on an older thread
  (`anbarasan@draas.com`) worth offering as a cc.

## Godrej / Manju DS thread map — TWO threads, same Hudson Circle 4.5 FAR deal (verified 2026-10-04)

NDR describes Godrej-Venture work in voice with mixed thread names; both live on `google-draas` and
neither the contact sheet nor `contact_resolver` identifies Manju (38 Manju rows, none godrej-tagged —
he is external, dsmanju15@gmail.com, NOT a DRA person). Resolve him from Gmail threads.

- **Thread A — "FW: 4.5 FAR Drawings"** (threadId `1a038d15540942c9`, ~16 msgs, 25-Aug → 24-Sep-2026).
  Starts with Viraj Majithia sharing a SharePoint folder (25-Aug) that nobody could open (MS sign-in
  AADSTS90072) → Harsimran re-shares 3 separate emails of attachments (27-Aug) because of size limits:
  `APPROVAL 4.5 FAR 19.08.2026_selected_...zip`, `PDF.zip`, `DWG.zip` (35 MB, 36 MB), plus a 33 MB
  `GHC-KPL-PRJ-XX-RPT-GFC-AR-0002-00.dwg` + xlsx. Sep: quotation PDF → Viraj says cost too high. Then
  Harsimran FORWARDS the three sets directly to `manju DS <dsmanju15@gmail.com>` (22-Sep), and Manju's
  24-Sep reply is the ask NDR remembers: "All floor plans, site plans and section and elevation one cad
  drawings required" — i.e. one COMBINED cad drawing file instead of the split sets. That message id:
  `1a0d1f11f054f816`.
- **Thread B — "Re: Bux ranka quatation"** (threadId `1a08ea566eef0676`) — misspelled subject (keep
  verbatim for threading). Manju's quotation ₹3,50,000 of ₹5,90,000, DWG + advance as JOINT start
  condition; NDR's 28-Sep mail: "Harsimran please share 4.5 dwg for pre-DCR as requested by Manju".
  This is the thread the `third-party-message-via-colleague` instructions above were built on.

Finding recipe ("find the email where Manju asked about a zip / combined file"):
search `from:manju godrej` (finds both threads), then narrow with the content term — `from:manju zip`
lands directly on the combined-CAD ask. `from:manju@draas.com Godrej dwg` returns ZERO (he is
dsmanju15@gmail.com, not draas). Deliver the Gmail thread-search URL + verbatim ask, and distinguish
"manju DS the pre-DCR architect" from every other Manju in contacts (Manju Bhua, Manju R Capital9,
Manju Rto, Manju Subbaih AnupShah…).
