# External counterparty mail — keep it short

Applies to anyone outside DRAAS who is a counterparty rather than a colleague:
Godrej Venture / GVI Hudson (Harsimran Singh, Satish Jadhav, Amit Saraf,
Viraj Majithia), buyer-side counsel, brokers' counsel, vendors.

## The correction

An Oct 2026 draft to Harsimran Singh shared two documents and then reproduced
the whole internal reading — chain analysis, every red flag ranked, the
lawyer's five conditions, the full requisition list, a point-by-point answer to
his seven questions. NDR's verdict:

> "the email to Harisimran is too extensive. You don't need to give that many
> details. Maybe just share the documents with him and give him a short
> summary of what is clean about this and what are the other documents we have
> requested and why. A very short summary."

The internal analysis is NDR's own work product for his decision. It goes to
**him**, in the report — not to the counterparty.

## The shape that works

1. One line of context ("As discussed. Both documents below, with view access
   granted to your email").
2. The links, numbered, each with a one-line description.
3. **What is clean** — one short paragraph, max two. "The last 30 years of
   title is continuous and fully registered, from <date> to <date>. The one
   litigation event inside that period was set aside on the impostors' own
   admission and the revenue record was restored in seven months. The extents
   reconcile exactly."
4. **What we have asked the seller for, and why** — bullets, one line each,
   only the items that affect *their* diligence. Say plainly when something
   does not touch the title being relied on ("the last two are outside the
   30-year window and do not affect the title we are relying on, but they are
   on the list").
5. The one question that is theirs to worry about commercially (for a DRC:
   GBA confirmation on transfer mechanics, validity and utilisation).
6. A closing ask — "please go ahead and start your team's due diligence now; I
   will share the balance as it comes in".

Nothing else. No comfort percentages, no negotiation read, no internal
ranking of red flags, no reproduction of the vendor lawyer's conditions in
full.

## Still true from the main skill

- Draft only. Never send. The draft goes in Drafts for NDR to send himself.
- Reply-all on a live thread: pull the Cc list from the **thread's earlier
  messages**, not just the message being replied to — the latest inbound mail
  often drops everyone who was cc'd on the original.
- Attachments: `EmailMessage.add_attachment(...)` on the base64url raw insert;
  the native `gws_gmail_reply_draft` tool has no attachment parameter. Set
  `threadId`, `In-Reply-To` and `References` from the replied message's
  metadata headers, then read the draft back and confirm the attachment size
  matches the local file.
- Grant Drive viewer access to everyone you cc, or the links in the mail
  bounce off a permission wall.
