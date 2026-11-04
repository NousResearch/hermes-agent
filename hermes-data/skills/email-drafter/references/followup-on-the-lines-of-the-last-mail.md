# "Send X a follow-up on the lines of the one we sent"

NDR's standing instruction shape for a live negotiation thread (Ranka Iris / Shankar Electricals /
Bhavik, 2026-10-27 is the worked example). He dictates a long block of position and mechanics and
then says *"on the lines of the one we sent"* — meaning: **the previous mail in this thread is the
template, the voice, and the source of every figure.** Do not invent a new format.

## Order of work

1. **Find the mail he means.** `gws_gmail_search(query="to:<party>", service_name="google-draas")`
   — newest `SENT` hit is almost always it. Read it in full with `gws_gmail_get`. It fixes the
   subject line, the greeting, the section order and the numbers, and the follow-up must be
   consistent with it or the counterparty will spot the drift.
2. **Read the counterparty's latest inbound** in the thread (their replies carry the claims and the
   numbers you are answering, and their message id is the correct reply anchor — see
   `references/threaded-reply-and-draft-verification.md`, the no-`to=` trap).
3. Draft as a **reply in the same thread**, draft only.
4. Read the draft back and confirm `to:`, `threadId`, body intact. Report figures he can check.

## Structure that lands (copy from the prior mail)

- `Bhavik,` — first name, comma, no "Dear".
- One-line acknowledgement of what changed since the last mail.
- Numbered objections / positions, each with a bolded heading-phrase, not a neutral label.
- **What I am chasing** / **What I need from you** — split his asks from yours.
- **How we hold the PDC** (or whatever the leverage instrument is) — the commercial mechanic,
  stated as a sequence, unambiguously.
- Optional **Timeline, for your context** — and say explicitly which items are *not* on the
  recipient's site so he does not chase them.
- Closes with ONE ask and a deadline phrase ("Please make the call today and revert").

Keep his rhetorical habits: *"That is the only way this can be done; there is no other way."*
*"That is a given, not a bargaining point."* Short declaratives. No hedging, no apologies,
no bullet-point soup where prose carries the argument.

## Carry the commercial mechanics EXACTLY, and flag ambiguity instead of smoothing it

Dictation on mechanics is often ambiguous about **who does what**. On the Ranka Iris follow-up:
*"prepare the PDC once you have closed the number with him in an 80-20 split, which I will have to
negotiate"* — readable as (a) Bhavik closes the total and NDR negotiates the split, or (b) Bhavik
closes an already-split number. Pick the reading consistent with the leverage logic (the person
who holds the face-to-face meeting negotiates the split), write it plainly, and **say in the chat
summary which reading you chose and what to change if he meant the other**. Silently choosing is
what gets a wrong instruction into a mail NDR then sends.

Same for the instrument: 80% one PDC + 20% another, because *"that way we will at least hold one
against him submitting an invoice"* — record the REASON, not just the split. The recipient needs
it to see why the structure is not negotiable.

## Figures must come from the thread, never from the dictation or memory

Anchor every number to the mail already on record and reconcile the two sides' numbers instead of
picking one:

| Item | Thread (NDR / reconciliation) | Counterparty |
|---|---|---|
| WO 616 value | Rs 35,51,800 | same |
| Extra works (Altimeter cubicle + fire Rs 2,80,000; DG Rs 4,22,204) | Rs 7,02,204 | same |
| Total scope | Rs 42,54,004 | same |
| Paid to 27-01-2026 | Rs 28,38,497 | same |
| Meter advance | Rs 2,00,000 (14-10-2026) | Rs 2,36,000 in his xlsx (17-10) |
| Net payable | Rs 12,15,507 | Rs 12,22,800 |

Where they disagree, **flag it to NDR in the chat** and keep the mail's own arithmetic internally
consistent — do not average them or quietly adopt the recipient's figure.

## Name and term normalisation (dictation → on record)

NDR speaks these; the mail must use the form on record:

- **Nagesh / Nandish → Nanjesh** (Shankar Electricals)
- **Buscom / Bescom → BESCOM**
- Bhavik → `bhavik@draas.com`; Anbu → sends as `anbarasan@draas.com`, receives at
  `pm2.blr@draas.com` (from Gmail headers, not the contacts sheet)

## Do not present dictation-only facts as if they were on record

The counterparty's stance often arrives verbally ("Bhavik had a long conversation with Nanjesh…
his stance is that this is how all these government cabling jobs are executed"). No such mail
exists in the mailbox — so the draft says *"Noted on your call with Nanjesh"* and does **not**
quote or date-stamp anything he did not write. Tell NDR in the summary that the stance is
dictation-only, so he can adjust if Bhavik's actual words differ.

## Chat summary for this task class

Terse, factual, in his order: draft id, To line, thread subject; then the argument as it was
written (objection 1, objection 2, sequencing, mechanics); then the **flags** — dictation
ambiguities, name normalisations, anything not evidenced in the mailbox, and any draft of his
that has since disappeared (check `in:sent` before calling it lost).
