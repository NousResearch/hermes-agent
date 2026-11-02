# Unanswered email → WhatsApp nudge (NDR workflow)

Recurring NDR ask, verbatim shape: *"Check if he has responded to that email. If not, the
same contents, referring to it — tell him I have sent an email but maybe you are not
checking it, so sending you a message as well. A WhatsApp message for him please."*

## Steps

1. **Resolve accounts first.** `gws_resolve_account` with no args lists every account +
   auth status. Search **both** `google-draas` and `google-ahfl` (and `google-gmail` when
   the counterparty writes to the personal address) — a reply may land in the other inbox,
   and reporting "no reply" off a single-account search is a wrong answer.
2. **Find the thread by topic, then read it.** `gws_gmail_search(query="<topic>")` →
   `gws_gmail_thread_get(thread_id=...)`. The message list is the answer: `labelIds`
   contains `SENT` for outbound and `INBOX` for inbound, so the **last message + its date**
   is definitive. Read the outbound bodies — the WhatsApp nudge must reuse its asks.
3. **Rule out a reply on another thread.** `gws_gmail_search(query="from:<their addr>")`
   newest-first. This also lets you report the honest line
   *"his last mail of any kind: <date> — <subject>"*, which is what makes the "no reply"
   claim credible instead of merely asserted.
4. **Report the evidence, not a verdict.** Give: the date + subject of the last outbound
   follow-up, the last inbound from them (date + subject), and which accounts were checked.
5. **Compose the nudge.** Shape NDR accepts:
   - Name the email **by date**, ideally both the original and the follow-up:
     *"I sent you a follow-up email on 15 October (following up my 10 October mail) on …"*
   - One line of the "maybe you aren't seeing the mail" framing he dictates.
   - Re-state the **same asks, same numbering**, as in the email. Do not soften, merge, or
     invent a new ask — NDR cross-checks the WhatsApp against what he already wrote.
   - Close with what happens once they confirm (the next step he is waiting to schedule).
   - Sign off as the user (first name only). For family contacts, single-name address,
     no "Dear", no honorifics.
6. **Deliver** via `whatsapp_link(text=..., phone=..., platform='telegram')` and hand over
   `display_link` verbatim. Say it is a **generated link that opens WhatsApp pre-filled** —
   never "sent" / "delivered".

## Pitfalls

- Do not equate "no reply in this thread" with "no reply" — step 3 exists for that.
- Do not escalate tone because the original email was firm; the WhatsApp carries NDR's
  content, not a new edge.
- If the counterparty owes the deliverable, keep the ask factual and dated — NDR's own
  phrasing is the source of truth for both content and register.
- Long nudge text may exceed Telegram's single-message limit → `whatsapp_link` returns
  `split: true` + `parts`; deliver **each part as its own message**.

## Worked case — Gunjur / Dharmesh Ranka (2026-10-16)

Thread `Re: Gunjur — Next Steps: Legal, Documentation, CLU & Resources`: outbound 22 Sep,
inbound 23 Sep, outbound 10 Oct, outbound follow-up 15 Oct — last message SENT, so no
reply. Cross-check `from:dharmesh.ranka1@gmail.com` showed his last mail of any kind was
8 Oct on an unrelated thread. Nudge reused the 15 Oct three asks verbatim (photocopy set
date / Rajan on the job / are the keys with him) and closed on planning the review + fixing
the Kusuma Muniraj handover date.
