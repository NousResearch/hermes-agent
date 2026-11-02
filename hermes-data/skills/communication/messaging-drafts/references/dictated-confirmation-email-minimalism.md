# Dictated short-confirmation email — build only the points dictated

Trigger: NDR (or Sarthak) dictates a two- or three-point confirmation to a counterparty after a
call — "tell him he has confirmed receipt of all the documents and will get back to us, and about
the bankers' meeting next week". Worked example: VGRO Finserv / Sudheer Desu thread
`1a10b06e4405268e`, 2026-10-24.

## Rule

Build **exactly** the points dictated, in that order, and nothing else. A dictated confirmation is
two or three sentences of body text. Target: under ~10 lines.

## Pitfalls — both fired in the same session

1. **Do not inflate it into a status report.** The first draft added a dated inventory of every
   document shared since 8-Oct, a summary of the shared folder's contents, and a "four items still
   being compiled, to follow by 29-Oct" block. The correction was blunt: *"nothing is there like
   documents… we have shared all the documents with them"*. He wanted the confirmation only. If he
   wants the document history, he asks for it — or it belongs in a status mail he requests.
   Corollary: do not insert a "pending items" section to protect accuracy when he has decided the
   posture of the mail. Flag the inaccuracy to HIM in the chat, not to the counterparty in the mail.
2. **Do not upgrade a promise into a confirmation.** Dictation said *"he will confirm the meeting
   with the bankers next week and try to close it"*; the draft said the meeting "has been confirmed
   for next week". Correct wording: *"The meeting with the bankers is yet to be confirmed. You will
   confirm the same next week and endeavour to close the matter at the earliest."* Match the
   counterparty's actual commitment — "will confirm" ≠ "confirmed" ≠ "fixed".

## Shape that worked

> Dear <Name>,
>
> Greetings!
>
> Further to our discussion, we note the following:
>
> 1. You have confirmed receipt of all the documents and information shared by us for your review,
>    and you will revert with your observations in due course.
>
> 2. <Second dictated point, with its true status verb.>
>
> <One closing line carrying the ask he explicitly wants — e.g. "We would be grateful for your
> position on Ranka Amber and Ranka Oasis along with your observations.">
>
> Warm regards,
> <Dictator's standard signature block>

## Mechanics

- Reply on the counterparty's most recent message (not the user's own latest send) so `To` resolves
  to them and the `threadId` stays the master thread — see `gmail-draft-threading-traps.md` §3b/§2g.
- Cc: the crew the counterparty had on their own mail (here NDR + the five VGRO colleagues).
- **On every revision, delete the superseded draft and rebuild.** Leaving the long version and the
  short version side by side in the thread makes him guess which to send. Report the new draft id
  and say plainly that the old one was deleted.
- Read the new draft back before reporting: `To`, `Cc`, `threadId`, `labelIds == ['DRAFT']`.
