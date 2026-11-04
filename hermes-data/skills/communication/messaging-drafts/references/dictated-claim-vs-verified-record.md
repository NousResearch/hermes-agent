# When the dictation overstates the record

Verified 2026-10-24 (VGRO Finserv lender-DD status mail, Sarthak / admin3.blr@draas.com).

## The pattern

The user dictates the *headline* he wants the counterparty to receive, and it is usually the
commercial line, not a factual inventory:

> *"I will try to mail to Sudhir. As per our conversation, you have received all the documents and
> you are reviewing it. And you will try to arrange the meeting with the bankers next week."*

The verified record contradicted it: six items were still open (March-2026 provisional balance sheet,
Ranka Udaya collection-account statement, Amber and Oasis NOCs, executed Oasis sharing agreement, and
the credit answer itself). Sending "you have received all the documents" would have been a false
statement to a lender in writing.

## Rules

1. **Never transcribe a dictated claim that the record contradicts.** Draft the mail so the commercial
   message lands, but with an accurate qualifier — here a short block: *"Four items remain, and we will
   share them by Wednesday, 29-Oct-2026: … Kindly note these do not affect the financial and project data
   already in your hands."* Dated, itemised, and framed as additive rather than as a gap in what was sent.
2. **Tell the user you did it, in one line, and offer the delete.** *"I added this so we are not claiming a
   complete set when four items are still open; delete the block if you want it out."* He decides — but
   the default draft is honest. Do not quietly soften it, and do not quietly keep it either.
3. **Echo back any actor/tense flip instead of resolving it silently.** *"you will try to arrange the
   meeting with the bankers"* is ambiguous between the counterparty arranging and us arranging; the draft
   used "you will endeavour to arrange … kindly let us know the date and time", plus an explicit ask:
   *"Tell me if it's actually we who are arranging, and I'll flip it."*
4. **Restate the stale item the counterparty actually asked for.** Sudheer's item 1 was a *provisional
   BS as at March 2026*; what sat in the folder was the FY 2024-25 audited set. Say which of the two is
   in the folder rather than letting "balance sheet shared" stand.
5. **Dates carry the credibility.** A status mail with no dates reads as an excuse. Lead each submitted
   item with the send date pulled from `in:sent` (8-Oct folder, 15-Oct expense summary, 16-Oct mortgage
   deed, 23-Oct sales MIS + plot schedule), then name what each mail carried.
6. **Name the account + draft id + thread id when reporting the draft** (see
   `gmail-draft-threading-traps.md` §6). Here: `admin3.blr@draas.com → Drafts`, draft
   `r1670590798161556761`, thread `1a10b06e4405268e`. The user runs several Google accounts and Drafts is
   per-account.

## Reusable body skeleton for a lender/investor status reply

```
1. Reference to the discussion (one line).
2. Documents — received and under your review
   a) <date> - <what was sent>   [one bullet per outbound mail, newest last]
   ...one paragraph summarising what the shared folder covers project-wise
3. Items still being compiled  [dated commitment + "does not affect the data already with you"]
4. <The commercial ask> — firm position, clearance status, indicative timeline
5. <Next step/m meeting> — who acts, request the date
6. Offer of clarification + signature block of the drafting person
```

Until the count of open items is zero, block 3 stays in the draft — it is what makes block 4
answerable rather than ignorable.
