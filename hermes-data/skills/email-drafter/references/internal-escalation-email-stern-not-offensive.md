# Internal escalation email — stern, but not offensive

**Trigger:** NDR asks for "a little stern email but without offending anyone — be very clever in
your language", usually after a colleague has done something he wants *on record*: threatening a
stoppage, escalating past him, overstepping on a payment. Verified 2026-10-16 (mail to Anbarasan M.
about a reported "work will stop at site for a month" message passed to Sarthak).

The email is **documentation, not a telling-off**. Everything follows from that.

## Shape that worked

1. **Raise it directly, and say why.** *"I want to raise something with you directly, and I would
   rather do it on the facts than on what was reported to me."*
2. **Ask before you assert.** Put the reported conduct as a question with two options, naming the
   outside party: *"Is that a position coming from Jayaram sir, or is that something being said at
   our end?"* This is the most important line in the mail — it lets the person save face and turns
   an accusation into a clarification.
3. **Exactly one plain sentence of disapproval.** No adjectives, no repetition: *"If it is coming
   from our end, then I have to say it plainly: it is not correct, and at your level it is not the
   way we work."*
4. **The facts, numbered, with dates and timestamps taken from the source record.** Not prose.
   When it was posted, when he approved it, what the issue was, when NDR approved it. For Kelsa
   invoices those fields are `Received for Approval on`, the stage-change timestamp in Recent
   Activity, and NDR's own note — see `kelsa-mcp` →
   `references/invoice-wo-payment-verification.md`.
5. **Restate the policy once, impersonally**, so it reads as a rule and not a personal rebuke —
   e.g. the 3–4 working-day approval cycle, and why it will not be compressed.
6. **Close with the ask plus one boundary line.** *"Please go through the above and call me. I am
   raising this with you and leaving it there — but at your level, the company's work cannot be
   presented as something that will come to a stop, and certainly not to a colleague."*

## Rules

- **Never accuse.** Report the words, question the source, then state the principle.
- **Facts first, feeling last.** The sternness lives in the closing sentence of each block, never
  in the middle of it.
- **Do not CC.** These go to him alone — "draft it for me to him". If NDR wants someone marked he
  says so; do not add a Cc by inference.
- **No drama in the subject.** A plain descriptive subject naming the matter and what the mail
  covers.
- **HTML with numbered sections**, like his other internal briefings. Plain-text markdown collapses
  into one running paragraph and reads as a wall.
- **Flag your inference.** If the facts block came from a record rather than from his own words,
  tell him which item you took it to be and offer to rewrite if he meant a different one.

## Two companion rules from the same session

- **"Reply all, but keep marking it only to X"** = `To:` X and **no Cc**. Reply-all here only means
  "stay on the thread". When the original message had no other recipients, reply-all and To-only
  are the same thing — say so rather than asking.
- **When a dictation refers to "the item" (the invoice, the approval), the facts live in Kelsa, not
  in email.** Fetch the record for created date, approval timestamps, current stage, open task
  holder and NDR's own notes. Never reconstruct those dates from memory or from the dictation — the
  whole value of the mail is that its facts are checkable.

## Attribution — the counterparty is the actor, the colleague is only the source
(user-corrected 2026-10-24, Ranka Iris / Shankar Electricals debit settlement to Bhavik)

A colleague (Anbu) had sent NDR a point-by-point reconciliation *denying* the debits a second
colleague (Bhavik) had raised. The first draft said so honestly — "Anbu has come back", "Anbu is
denying this debit", "his position is…". NDR rejected the framing outright:

> *"I don't want the email to read saying that Anbu has come back and Anbu is denying it. We don't
> want to even say that we formulated it out of Anbu's response. We just want to say, Bhavik, we
> have spoken to Nanjesh / Shankar Electrical and he is denying all of this and these are his
> responses below. Let's not bring Anbu into this picture."*

**Rule: when writing to colleague B about a position that colleague A supplied on a third party's
behalf, attribute the position to the third party.** A's reconciliation is *source material*, not
the actor. Convert:

- "Anbu does not accept this debit. His position: …" → **"Denied. [The vendor's] basis: …"**
- "Anbu has no factual basis recorded and is getting Bhavik's facts" → **"[He] has given no factual
  basis at all and is disputing it outright."** (then ask B for the facts)
- "I am attaching Anbu's working sheet" → **"the reconciliation on record is attached"**
- Drop A's name from headings, from the reconciliation's title line, and from any "as per A's
  reading" aside. Only A's own operational handover leg stays named if NDR dictated it (e.g. "PDC
  delivered into Anbu's hand").

**Why it matters beyond tone:** naming A hands B a person to argue with instead of a number to
settle, and it exposes A to a dispute he was only asked to quantify. The mail's job is to move the
counterparty, not to referee the internal chain.

**Second rule from the same rewrite — never adopt a decision NDR has not made.** The first draft
said "I am inclined to drop this debit". NDR had only asked for Anbu's denial to be *presented*.
Present it, then put the onus back: *"That is his case as it stands. If you have something on
record that changes the sequence, this is the point to put it, because on the record as it is today
the cost sits with us."* Same for evidence gaps: "the photographs are not evidence of causation" →
"that is his position on attribution; if it is to be pursued we need the site record".

**Third rule — when two sources quote different figures from the same document, do not publish the
one NDR is not asking about.** Bhavik recorded 2.5 m earthing rods vs 1.5 m supplied; Anbu's reading
of the work order was 2 m vs 1 m. After the attribution rewrite the second reading disappears (it
was Anbu's) and the mail keeps only the recipient's figure plus *"settle the specification from the
BOQ and confirm in writing."* Publishing the conflict as a dispute invites the counterparty to argue
the version that suits him.

**Mechanics:** the bridge sends stored text, so a framing rewrite means `drafts().delete()` the old
draft and recreate — never edit prose after showing it. Deleting the superseded draft keeps Drafts
from filling with near-duplicates of the same mail.
