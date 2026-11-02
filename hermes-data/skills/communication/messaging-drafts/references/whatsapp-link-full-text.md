# WhatsApp links: always carry the FULL message text

## The failure (Oct-2026, NDR → Vivek G of CMS IndusLaw, Ranka Oasis follow-up)

NDR asked for a WhatsApp message to Vivek. What he received opened WhatsApp with only **"Dear Vivek,"**
pre-filled — the entire body was missing. His report, verbatim:

> *"The whatsapp message for Vivek is not correctly made, it just cuts off after Dear Vivek and that's it,
> comma and that's it. Can you please check and redo it."*

It passed review a second time only after the whole message was pushed through `whatsapp_link` in one go and
the returned URL was handed over exactly as returned.

## Rules

1. **Compose the message first, in full, exactly as it must appear in WhatsApp** — greeting, every sentence,
   sign-off. Do the thinking at this step, never at the URL step.
2. **Call `whatsapp_link(phone=…, text=<FULL TEXT>)`.** Pass the text untouched — no pre-stripping, no
   pre-escaping, no shortening of words in brackets.
3. **Hand over the URL exactly as the tool returns it.** Never elide it with `...`, never print a shortened
   variant for tidiness, never rebuild it by hand (`api.whatsapp.com` by hand is forbidden anyway, and
   `wa.me` is a stale path that corrupts `%` on mobile).
4. **Also paste the readable message text next to the link** so NDR can proof it without leaving the chat.
   If the link is wrong, this is how he catches it — and how you catch it before he does.
5. **If the tool returns `split: true` and a `parts[]` array**, deliver each part as its own separate
   Telegram message. Never concatenate parts into one message — Telegram's own splitter will cut a link in
   half.

## Self-check before delivering

- Does the returned URL visibly contain the encoded opening **and** the encoded closing of the message
  (a rough length check: the URL should be materially longer than the message text)?
- Does a link that terminates immediately after the first comma, or after the greeting, mean truncated
  content? Yes — regenerate it; do not send it.
- Quote at least the first and last few words of the message in the reply, so the delivery itself shows the
  full span reached the link.

## Related

The `whatsapp_link` tool is the ONLY sanctioned producer of WhatsApp deep links (see the messaging-links
doctrine). If it is not in the tool list, say so — do not fall back to manual URL construction via
`execute_code`, `urllib.parse`, or string formatting.
