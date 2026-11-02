# Querying a provider about an apparent error on their own document — polite-but-urgent tone

**Triggers:** NDR spots something wrong on a prescription, hospital record, invoice, bank
statement, vendor document or bill and wants it "clarified". Usually dictated as *"send them a
message, mark it critical, get this clarified"*.

This is the single most common drafting request in the medical/billing/legal area, and the one
where getting the tone wrong costs the most. NDR's rule, verbatim (22-Oct-2026):

> "let's make sure the message is very polite, cooperative, not trying to in any way sound,
> point out that the doctor was wrong... just make sure that the draft goes out nice and clean."

He also, in the same breath, wanted it flagged **CRITICAL — URGENT CLARIFICATION NEEDED** at
the top. Both are true at once.

## The core rule

**Urgency marker and deference are independent.** The top of the message can shout. The body
must never accuse. You are reporting a discrepancy you noticed, not a mistake they made.

## Skeleton that works

1. **Marker line** — `*🔴 CRITICAL — URGENT CLARIFICATION NEEDED*`, then one line of patient /
   document identifiers (name, UHID/ref, date, consultant, department).
2. **Apology for the hour** — "Good evening, and apologies for writing so late."
3. **Quote the offending text verbatim**, in a block, so there is no argument about what it says.
4. **A self-doubt clause** — "I want to be careful here, because this may simply be me misreading
   the note." This single sentence does more work than anything else in the message.
5. **What you were actually told**, stated plainly and without contrast verbs like "however" or
   "but in fact". "During the consultation Dr. X did not mention a deviated nasal septum to me —
   what she discussed was reflux-related throat clearing."
6. **Why you noticed** — give the comparator, as evidence rather than as an accusation. E.g. the
   son's note from the same doctor, 39 minutes earlier, carries a full
   `EXAMINATION FINDINGS: ANTERIOR RHINOSCOPY` block; NDR's carries none.
7. **State your inference as an assumption, explicitly flagged as an assumption** —
   "So I assume no examination for a deviated septum was done for me, and that this may have
   carried over while one prescription was closed and the next opened — **but I am not certain,
   and I would much rather ask than assume.**"
8. **Numbered asks, max three**, ending in the practical remedy: "would it be possible to issue
   a corrected advice / prescription for UHID … so that the version I file is accurate?"
9. **The no-complaint disclaimer, naming the person positively.** "There is no complaint intended
   here at all — Dr. Ajitha was extremely helpful this evening, and I am only trying to keep my
   records correct."
10. **Sign-off** — name + `+91 98800 55634`.

Ask for the **corrected document**, never for accountability, an explanation of who erred, or an
apology. If the provider did nothing wrong, the same message still works — it reads as a records
query either way.

## Words that must not appear

- "error", "mistake", "incorrect", "wrong", "typo" (in the accusation slot — fine as one of the
  offered possibilities, e.g. "or is it a typographical / copy-paste error in the printed
  prescription?" where the *document* is the subject, not the person)
- "you said", "you told me", "your staff", "this is unacceptable"
- "please do not leave it ambiguous" — this was in a first draft and read as a threat. Cut it.
- Any line implying consequences (escalation, complaint to a council, refund demand) unless NDR
  has separately asked for that.

## Multi-channel

If it goes on WhatsApp and by email, say so inside the message: *"I am also sending this same
query by email to admin@….com, marked for Dr. X's attention. Please treat the two as one request
and reply by whichever is convenient."* That stops two half-answers.

## Addressing when the individual has no published address

22-Oct-2026: Dr. Ajitha Tavvala had **no email anywhere** on her prescription or advice — no
letterhead email, no website, no doctor-level contact. The clinic publishes only
`admin@vaayuchest.com`.

Rule: **do not infer a personal address from a domain pattern**, even when you can see colleagues
on that pattern (`ravimehta@vaayuchest.com`, `anmol@vaayuchest.com`). Send to the generic
clinic/company address with `kind attention <Name>` inside the subject line, and tell NDR that is
what you did and why. Offer to look for a direct address on a different letterhead — but never
present a guessed one as verified.

## Attachments

Attach the provider's own document (scanned by NDR) as the exhibit. Give it a truthful name
including whose document it is, because clinics file by patient:
`20261022_Nishant_Ranka_DrAjitha_ENT_Advice_Vaayu.pdf`.

When the same query needs the paper trail too (vaccination card, prescription, invoice, product
photo), build one merged annexure PDF with A/B/C/D labels rather than sending four forwards.

## Verification before saying "sent"

- WhatsApp: `wa_draft_status(draftId)` → `state=sent`. Do not report delivery from
  `wa_send_confirmed` alone, which only says "queued".
- Email: after creating the draft, fetch it with
  `messages().get(id=…, format='full')` and walk `payload.parts` to confirm the attachment part
  exists and its `body.size` matches the source file byte-for-byte. Report the draft id and the
  byte count; NDR sends it himself.
