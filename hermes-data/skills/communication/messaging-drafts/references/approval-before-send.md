# Approval before send — Telegram/reportee messages (preference, verified 2026-09-28)

NDR's standing behaviour when he dictates an outbound follow-up to a reportee or third
party via a real send path (Telegram `send_message`, not just a WhatsApp deep link): he
wants to **read the message before it goes**.

Worked case: *"Based on that, I want a WhatsApp message sent to both of them… Also I need
follow-up messages. **And show me the message before sending it to him on telegram.**"* —
the deliverable was the drafted text in the reply, un-sent, with *"say the word and it
goes"*.

Rules:
1. **Default = draft + present.** Show the full text in-channel (the reply itself), clearly
   marked as a draft, and state what is NOT yet sent.
2. **Send only on explicit go-ahead** ("send it", "go", an edit + "send"). After sending,
   report delivery rather than silently assuming it (paired Telegram gateways can fail;
   verify what the platform actually accepted if there is any doubt).
3. This sits alongside the email hard-rule (drafts only, never auto-send) and the
   WhatsApp-link rule (links are tapped by NDR, never "sent"). The Telegram path was the
   gap: it *can* be sent programmatically, which is exactly why the visible draft matters.
4. When a message is multi-part (tracker updates + several threads), put the pending
   message at the end of the reply, labelled, so the approval decision is one tap away.