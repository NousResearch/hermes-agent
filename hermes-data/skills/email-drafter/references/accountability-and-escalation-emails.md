# Accountability / escalation mail — stern but never offensive

**Trigger:** NDR says some version of *"a little stern email but without offending"*, *"it should not
be offensive to anyone"*, *"be very clever in your language"*, *"I wanted to document it"*, or *"at
your level this is not done"*. The recipient is a **senior colleague**, and the complaint came to
NDR second-hand (usually relayed verbally or on WhatsApp by a third person).

The governing principle: **document the facts and let them carry the rebuke.** The rebuke is one
sentence; everything else is evidence the person can check for himself. That is what keeps it stern
without being offensive.

## Structure that worked (verified 2026-10-16, to Anbu re: a stoppage threat over a Vardhan invoice)

1. **Open by naming the reason for writing, not the offence.** *"I want to raise something with you
   directly, and I would rather do it on the facts than on what was reported to me."* Naming the
   source as "reported to me" preserves their face and keeps NDR out of a he-said-she-said.
2. **State the reported behaviour, then ask two clarifying questions instead of asserting.**
   *"I am told you spoke to X and conveyed that … Before I take that at face value, two questions: is
   that a position coming from <counterparty principal>, or is that something being said at our
   end?"* The "is it from you or from them" fork is the single most useful move — it lets the person
   retreat without losing standing.
3. **Then exactly one plain line of judgement.** *"If it is coming from our end, then I have to say it
   plainly — it is not correct, and at your level it is not the way we work."* One. Not a paragraph
   of reproach. Do not use words like shameful, unacceptable, disappointed.
4. **The numbered facts, pulled from the system of record** (Kelsa, the invoice trail, the thread).
   Dates, times, amounts, stages, who did what when. This is the whole body of the mail and it is
   deliberately neutral in tone. Include the discrepancy NDR himself had found and how it was
   settled, so the record shows he was diligent, not obstructive.
5. **The conclusion from those facts, in one line.** *"So: your approval was three days ago, and mine
   two days ago. Nothing has been sat on."*
6. **The rule going forward, stated as policy rather than as a personal criticism.** *"We have said
   this repeatedly, and it holds for every payment: the approval cycle takes three to four working
   days. We are not going to speed finance up."* Give the *reason* (accounting errors, tax issues) —
   it converts a reprimand into a shared standard.
7. **Close with a face-saving exit and the boundary, in that order.** *"Please go through the above
   and call me. I am raising this with you directly and leaving it there — but at your level, the
   company's work cannot be presented as something that will come to a stop, and certainly not to a
   colleague."* The boundary lands last, softened by "and leaving it there".

## Rules

- **One recipient, no Cc**, unless NDR names someone. A note like this goes to the person alone —
  do not copy their manager or the person who reported them.
- **HTML**, numbered facts as a table or an ordered list. Same reasoning as any multi-point internal
  briefing: plain-text `*bold*` markdown collapses into one running paragraph and reads as a rant.
- **ASCII-only subject** — this applies to subjects *you* author, i.e. a new mail. A threaded reply
  keeps the inherited subject **verbatim**, en-dash and all, RFC 2047-encoded; never ASCII-flatten
  an inherited subject (see `threaded-reply-raw-api.md`, "Non-ASCII subjects in hand-built MIME").
- **Never fabricate a date or a stage.** If you cannot find when something was posted or approved,
  say so and ask NDR rather than inventing a plausible timeline — a wrong date in a mail that exists
  to "document" a matter destroys its purpose.
- **Ask NDR to confirm the item identity before he sends.** State which record you built the facts
  from (e.g. "Kelsa invoice 012 / PO 739") so he can correct you if he meant a different one.

## Addresses confirmed from Gmail headers (do not take these from the contacts sheet)

| Person | Send-to address | Note |
|---|---|---|
| Anbarasan M ("Anbu") | `anbarasan@draas.com` | sheet says `pm2.blr@draas.com` — wrong for mail. **Update 2026-10-24:** his 23-Oct reply on the Shankar Electricals thread arrived FROM `pm2.blr@draas.com`, so both are live. When replying on an existing thread, reply to the address he actually used there; for a fresh mail, use `anbarasan@draas.com`. |
| Prakash Singh | `psingh@draas.com` | |
| Gowri Singh (marketing / content approval) | `gsingh@draas.com` | sheet carries only personal yahoo/gmail |
| Sarthak Sharma | `admin3.blr@draas.com` | voice "Sartak" / "Sartak" |
| Vinod Kumar Das ("Rahul") | `vkdas@draas.com` | |
| Manish Ranka | `mdr@draas.com` | |
| Dharmesh Ranka | `dharmesh.ranka1@gmail.com` | |

## "Present his feedback to him" = PRESENT, do not decide (verified 2026-10-24)

A close cousin of the escalation mail, and the one that catches you. NDR says *"take Anbu's
feedback and present it to him that he is denying it … he is unwilling to take that debit"*. The
mail carries a **third party's position**. It must never be converted into NDR's own conclusion.

First draft of the Bhavik / Shankar Electricals mail (WO 616 debit reconciliation) said *"I find
that difficult to argue with … I am inclined to drop this debit"* — a **decision NDR never took**.
Caught before sending; the draft had to be deleted and rebuilt with:
*"That is his case as it stands. If you have something on record that changes the sequence, this is
the point to put it."*

- **State the colleague's position in their own terms, then hand the rebuttal back.** Open with
  *"<Person> does not accept this debit."* Never *"his point is correct"*, *"I agree"*, or *"we
  should drop this"* unless NDR said it.
- **NDR's own earlier position on the thread IS his** and may be restated — *"my position on the
  work order has not changed"* is legitimate because he wrote it previously. Quote it from the
  earlier sent mail, don't paraphrase it into something stronger.
- **When the colleague and the counterparty quote different specifications from the same work
  order, surface the conflict and make the counterparty settle it off the source document.** Do not
  pick one. Real case: the colleague recorded 2.5 m earthing rods vs 1.5 m supplied; the colleague's
  colleague recorded the WO as a 2 m strip vs 1 m plus brick bat — same WO, two figures, and it was
  the one item everyone agreed was a clean debit. Both figures went to the recipient to settle off
  the BOQ.
- **Do not rule on money.** If NDR has not given a figure or a call, the mail asks for one. Closing
  on a sequence ("clear his accounts, then the PDC, then activation, then the cheque — not before")
  is fine; that is NDR's stated order, not an inference.
- Where the colleague's own reply contains a good systemic point (e.g. *"deductions have to be
  raised on site while the job is executing, not reconstructed at settlement"*), **ask NDR before
  adding it** — it is a criticism of the recipient's team and he may not want it in this mail.

Relaying the colleague's working sheet as an attachment on that reply-all is the
`threaded-draft-relayed-attachments.md` pattern: download the file from the **colleague's** message
(not the one being replied to), and byte-verify the attachment on the finished draft.

## Ambiguous instruction: "reply all … but mark it only to X"

NDR says both in one breath. He means: **reply on the same thread, addressed to that person alone,
no Cc.** Build it as a threaded reply (same `threadId`, `In-Reply-To`/`References`) with `To` = that
person and no `Cc`, then state in your report that you kept it to that one person — so he can widen
it if he meant otherwise.

## Wrong file if they AGREED rather than got caught out

This page is for behaviour that has to be corrected. If instead the colleague has **accepted** extra
scope and hedged with a stated constraint ("happy to help, within my four hours a day"), that is a
different class: praise the acceptance, respect the constraint and redirect inside it, and put the
slipped deadline on notice to a *second* person on the same thread. Use
`deadline-and-capacity-pushback-replies.md`.
