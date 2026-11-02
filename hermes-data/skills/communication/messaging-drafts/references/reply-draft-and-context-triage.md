# Reply-draft recipient verification + triaging a context-free voice ask

Two lessons from 05-Oct-2026 (Bharat, invoice chase for a NoBroker plan). NOTE: this pointer could
not be added to SKILL.md because that file is at its 100 KB limit — this reference is listed under
`linked_files`, so it is still discoverable via `skill_view`.

## 1. VERIFY the recipient of every reply-draft before reporting it ready

A Gmail thread's **last message may be your own forward to a colleague**, not the inbound message you
intend to reply to. `gws_gmail_reply_draft(message_id=...)` addresses the draft from the message you
pass in — so passing the wrong id silently addresses the wrong person, and the draft still comes back
with a `draft_id` and a `thread_id` that look perfectly healthy.

Concretely: the NoBroker thread was

1. inbound — pitch from `vallamkondu.dileep@nobroker.in`
2. **the user's own forward of that pitch to `rnr@draas.com`**

Replying to the *thread* would have addressed Roshini Ranka. The fix is to pass the **inbound
message's id** (`1a10a65b63a15127`), then prove the result by reading the draft message back:

```
gws_gmail_reply_draft(message_id="<inbound msg id>", service_name="google-draas", body=...)
  -> {"draft_id": "r329...", "message_id": "1a10abe...", "thread_id": "1a10a65b..."}

gws_gmail_get(message_id="1a10abe...", service_name="google-draas")
  -> "to": "Vallamkondu Dileep <vallamkondu.dileep@nobroker.in>"   # <-- the proof
```

Report the To (and explicitly "no CC" when the user has asked for no CC) in the reply. Never report a
draft as ready off the `draft_id` alone.

## 2. Triaging a voice ask whose context is gone (session expired / no prior turn)

The message arrived as: *"here I just want him to send an email only for delay. I don't want to put
anyone in CC. So we have done the payment as per whatever the payment link was being shared. And I'm
just waiting for the invoice to be shared. So in case if you can send it quickly, that would be
great."* — no recipient, no counterparty, and the prior session had expired.

What worked, in this order:

1. **Mine the mailbox before asking** — `gws_gmail_search` on the semantic content
   (`invoice OR "payment link" newer_than:10d`, then the vendor name). This is what identified the
   real candidates: a Zoho-Billing invoice with a PAY-NOW secure link (Tailortalk), a reseller
   tax-invoice with a payment link (Brio), and a paid portal plan (NoBroker). Search *before*
   `session_search` archaeology — the mailbox is the authoritative record of the transaction.
2. **Ask ONE question with numbered candidates**, via `clarify`, including the evidence next to each
   option (vendor, amount, what was found) and a final "someone else" escape. He replied with a single
   digit — "2" — which is the entire cost of the clarification.
3. **State the interpretation you will draft from, in the same question** — *"Your line I'll use:
   'Payment has been made as per the payment link shared. We are awaiting the invoice…' Correct?"* — so
   a wrong reading is corrected in the same tap rather than after the draft is built.
4. **Never guess the recipient to avoid asking.** The wrong-party risk is far worse than one question.

## 3. The "payment done, invoice awaited" chase — shape that works

- To the vendor's own contact only; honour "I don't want to put anyone in CC" literally (verify no CC).
- Open by naming the thing precisely (plan name + amount) so it cannot be confused with another order.
- One fact, one line each: *payment has been made as per the payment link shared* → *we are awaiting
  the invoice* → *request you to kindly share it at the earliest*.
- Subject: reply back onto the vendor's own thread (`Re: <their subject>`) so the context travels.
- Then offer the alternative reading in chat — here: if he actually meant the invoice is delayed and he
  wants the delay flagged, say so and it will be reworded. Cheap insurance against a mis-read dictation.
