# Terse dictation → recover the counterparty, then verify the draft

Worked case: 05-Oct-2026, Bharat (sales1.blr@draas.com).

## The failure mode this prevents

A session expires; the user's next message is a voice note with all context stripped:

> *"So here I just want him to send an email only for delay. I don't want to put anyone in CC. So we
> have done the payment as per whatever the payment link was being shared. And I'm just waiting for
> the invoice to be shared. So in case if you can send it quickly, that would be great."*

No recipient, no vendor, no project, no amount. Guessing a counterparty on a mail about **money** is
the worst possible guess — so recover the referent from evidence first, and ask only once.

## Recovery ladder

1. **`session_search` first** — cheapest. Query the topic words (`payment link invoice delay`,
   sorted newest). If the expired session is indexed, this resolves it in one call.
2. **When sessions don't resolve it, search the user's OWN mailbox.** This is what actually worked
   and it is the step that is easy to skip:
   - `gws_gmail_search(service_name=<resolved via gws_resolve_account>, query='invoice OR "payment link" newer_than:10d')`
   - then a wide sweep: `newer_than:3d -from:<the automated sender you want to exclude>` — the inbox
     **plus SENT** is the picture (default search returns both).
   - The answer here was a **SENT forward to the accounts team that same morning**, containing the
     vendor's invoice email with a "PAY NOW" link. That identified the vendor and the exact thread.
   - Use the session user's own token (`service_name='google-draas'` for Bharat). Never DWD for this.
3. **Ask ONCE, with concrete candidates drawn from what you found** — `clarify` with up to 4 options
   naming the actual parties and amounts ("Vendor A — their@email · Vendor B — ₹X plan · our own
   accounts team · someone else"). The user replied with a single character ("2"). A generic
   "who should this go to?" would have cost a whole round trip and a re-explanation.
4. **Draft only after the party is named.**

## Verify the draft before reporting it ready

The draft tools return ids only. **Re-fetch and assert the envelope** — the recipient set is the thing
most likely to be wrong and it is invisible in a chat summary.

```
gws_gmail_reply_draft(message_id=<counterparty's OWN message id>, body=..., service_name=...)
# -> {"draft_id": ..., "message_id": ..., "thread_id": ...}
gws_gmail_get(message_id=<returned message_id>)   # assert To == counterparty, Cc empty
```

**Pitfall — reply to the counterparty's own message id, not the newest message in the thread.** In
the worked case the newest message in the thread was *the user's own forward to a colleague*; replying
to that would have addressed the colleague instead of the vendor. Pick the id of the message **from
the person you are answering**.

## Honour "no CC" as an envelope instruction

"I don't want to put anyone in CC" is an instruction about the *envelope*, not a style note. Assert
`Cc` is empty before reporting. A reply draft auto-fills `To` from the sender only, which usually
satisfies it — verify rather than assume.

## Report shape that works

State plainly that the draft is in Drafts and **not sent**; give To + subject + thread; quote the body;
then **surface every assumption you had to make** and offer the correction in one line — e.g. *"I read
it as we paid and they owe us the invoice. If you meant the opposite, say so and I'll rework it."*
Bharat corrects fastest when the assumption is stated rather than buried. Also offer the optional
trim ("want the amount removed?").

## Related

- Long-form source/disclosure equivalents live in `vendor-sourcing-outreach`.
- Voice-name and vocabulary corrections: `references/voice-name-corrections.md` (this skill).
