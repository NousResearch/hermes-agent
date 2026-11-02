# Drafting from NDR's voice dictations — garbled stretches

NDR dictates almost everything by voice. Long notes degrade: somewhere in the
middle or tail there is a run of phonetically plausible but meaningless words —
a fabric of names-sounds and words that decode to nothing. Real examples:

- "T a Ravaanah, Sanchudura, Chima, Kastava, Cahari, Osumari, Riosalai..."
- "Saly, Gumbhane, Seshwaran, Udaya, Vasisth, Surpresorand, NK Gowri..."

This is normal input, not a failure to report.

## Rule: read for INTENT, not for literal tokens

1. **The coherent majority governs.** In the example above, ~90 words of
   gibberish sat in the middle of a long message whose opening and closing
   sentences stated the ask completely (thank the person, correct the scope
   misunderstanding, ask for a time-logged rundown). The draft was fully
   producible without decoding a single token of the noise.
2. **Do not block, and do not ask the user to repeat** when the meaning is
   recoverable from the coherent parts. Re-asking a user who just spoke for
   three minutes to repeat the whole thing is the worst available outcome.
3. **Do not fabricate into the gap.** Never infer a name, amount, date,
   deadline or phone number from garbled audio, and never smooth it over with
   plausible filler. Skip it and build from what is clear.

## When you MUST stop and ask

Ask — but ask about **one precise token**, not the whole message — when the
garbled stretch contains something decision-relevant:

- a person's name ("send this to <name>") — wrong recipient is unrecoverable
- an amount, a rate, a percentage
- a date or deadline ("by tomorrow" → resolve which date)
- a phone number, a unit number, a survey number
- a which-one-of-two choice (this plot or that one, Structure A or B)

Cross-check names with `contact_resolver` / `entity_resolver` /
`wa_search_messages` before guessing a spelling. Phonetic traps already known:
"Pebli"/"Ruan" = Ruhaan; "Balu" = Anbu; "giraffe" = Jiraaf.

## Ground the interpretation in context, then show your work

Before drafting, spend one read on the surrounding record — the live chat
thread, the prior email on the same subject, the group thread. NDR's dictation
assumes that context and compresses it ("the email I have sent him in this
regard"); the draft is wrong if it is written context-free.

Then, in the approval message that accompanies the draft, include a short
**"Context I built this on"** block: 2–4 bullets saying what you read and what
you inferred. This is what lets NDR catch a misread *at the same moment* he
approves the wording, without a second round trip. It also demonstrates the
draft is grounded rather than guessed.

## Decode a garbled name/number by SEARCHING for a fragment, not by guessing

A half-heard identifier is usually recoverable from the mailbox if you search on the part that
did survive. Verified 2026-10-23 (Anbu on O.S. 05/2026): the advocate came through as
"Shivshankar", and the phone as "944327 245" (nine digits, unusable). Searching Gmail for the
**leading digits as a bare string** — `q='94432'` — surfaced
`Shivashankar Muniswamy <dmsadvhsr8@yahoo.in>`, i.e. the correct full name *and* the mailbox he
actually writes from. (`dms adv hsr` = DMS Advocate, Hosur — the local-practice convention that
made the address self-confirming.)

Rules:
- Search on the **surviving fragment**: leading digits of a phone, a distinctive syllable of the
  name, or both. `q='94432'`, `q='Venkatamma'`, `q='Shiv Shankar' OR 'Sivashankar'` (spellings
  drift: Shiv-/Siva-/Shiva-).
- Then **confirm the identifier from a document, not from the voice** — his own letterhead in an
  attachment gave `D.M. SHIVA SHANKAR … CELL: 94432 70245`, which is the version that went into
  the mail. Say in the approval message that the phone is now document-verified, so the user knows
  you did not just transcribe a guess.
- Watch for the letterhead/mailbox domain mismatch (letterhead `yahoo.co.in`, mail arrives from
  `yahoo.in`): use the address the mail actually comes from, and mention the other one.

## When a decision-critical figure STILL won't decode — ship a marked blank

Sometimes the draft must exist before the number can be recovered (the counterparty is waiting, or
the user has moved on to the next item). Do not stall, and do not invent. Verified: a dictated
reconciliation read *"15.4 on 5 odd 2 lakhs is due out of which we have paid 2 lakhs"* — three
plausible readings, none assertable.

- Write the sentence with an obvious, unmissable blank: `the amount due to Shankar is Rs [___]
  lakh, against which Rs 2,00,000 has already been paid as an advance`.
- Put the *verified* figures around it in full (WO value, GST, the advance) so only the unknown
  stays blank — the draft is then 95% usable and the blank is one keystroke.
- In the approval message, list the blank as a numbered action item with the readings you heard:
  *"your voice gave 15.4 / 5-odd / 2 lakhs — I won't guess a figure NDR will hold you to"*.
- Never round, average, or pick the most likely reading to make the mail look finished.

## Corrections arrive as MORE voice notes — apply each to the artefact, then echo it back

A dictated draft is usually corrected in instalments minutes apart (here: *"the date is 9 December,
not 12"*, then *"the phone number is 944-327-0245"*). Each correction must land in the **actual
draft/sheet**, not just in the chat reply.

- Edit the live artefact (full raw-MIME `drafts().update` keeps the draft id and thread; or delete +
  recreate when the change adds an attachment), then read it back and assert the corrected string is
  present.
- Echo the correction in one line: *"Phone corrected in the same draft — 94432 70245."* One line,
  not a re-summary of the whole mail; he is tracking a delta, not a document.
- If a correction contradicts a document you already relied on, say which source won and why.

## A one-word tail like "Pag." is an artefact, not content — do not invent a matter from it

NDR's dictations frequently end on a single stray token the recogniser produced from a breath,
a click or a trailing "…and that's it". Observed repeatedly: **"Pag."** (and "Pay." mid-sentence),
which appears both as the tail of a long dictation and, once, as a voice note consisting of
nothing else. Verified 2026-10-25: "Pag" trailed a list of meeting topics
("…Nelamangala Property Matter, Pag.") and the obvious — wrong — move was to treat it as a
missing agenda item.

Rules:

- **Search the record before believing the token.** `wa_search_messages`, Gmail search on the
  bare string, and `session_search` for the same word. If it matches nothing but prior
  dictation tails, it is noise.
- **Say so plainly and move on**: *"'Pag' isn't a matter — it's the tail of your dictation; the
  same token shows up at the end of your <date> note. Nothing is missing from the list."*
  Never paper over it by adding a vague agenda line ("Pag — to be confirmed"), and never guess
  which matter it might be.
- **A bare artefact message is still a turn worth using.** Answer the artefact in one line, then
  deliver the thing that actually moved since the last turn (in that case: a counterparty had
  replied to the thread at 00:01 adding two agenda items). Do not spend the whole reply on the
  token.
- The same rule covers other known tails: "Thank you.", "…that's it, over", a repeated last name,
  and phonetic debris like "Saly / Gumbhane" (see above).

## After a gateway shutdown or a compaction, re-read the thread before resuming

A drafted-but-unsent email does not necessarily still be a draft: the user may have sent it
himself while you were down. Verified 2026-10-25 — a resume-after-shutdown turn found
`in:drafts` empty because all three drafts had already gone out, and the counterparty had
replied at 00:01 with new material.

- Before claiming "the draft is still in your Drafts folder", check both states:
  `gws_gmail_search(q='in:drafts')` **and** a `subject:"<thread>"` search on the account, and
  look at the `SENT`-labelled messages and any new inbound replies.
- Surface the new inbound mail in the resumed reply. The most useful thing in that turn was not
  the restored task — it was the 00:01 reply with two fresh agenda items, which went straight
  onto the tracker sheet.
- Update the live artefact (agenda sheet, tracker, doc), not just the chat: append the new item
  with its source attribution ("(Manish)") so the record shows where it came from.

## Pitfalls

- Treating "tomorrow"/"today" literally off the raw transcript. Resolve against
  the session date before writing it into the draft (server clock is UTC;
  convert to IST first).
- Rounding a dictated number up or down ("close to 12 or 13,000" → do not
  write "₹12,500"). Keep NDR's own hedge: "closer to ₹12,000–13,000".
- Silently dropping a dictated point that was hard to phrase. If he said it, it
  goes in — reorganised for readability, never omitted.
- Asking a clarifying question whose answer is sitting in the WhatsApp thread
  or the email thread you were told to look at.
