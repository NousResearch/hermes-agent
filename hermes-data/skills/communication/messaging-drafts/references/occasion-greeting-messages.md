# Occasion & greeting messages (birthdays, Diwali, congratulations)

NDR regularly dictates a short personal message for someone in his network and asks for a
**WhatsApp link** so he can send it himself. Recurring occasions: birthdays / milestones
(50th), Diwali, weddings, new ventures, condolences, festival greetings.

## The core rule — POLISH, never transcribe

His dictation is thinking-out-loud: it contains filler, false starts and placeholders.
Example (2026-10-08, Sharab Reddy's 50th):

> *"I just want to send him and I have gone to a couple of his Diwali parties. So I just want
> to send him a message. He recently had his 50th birthday. Hey chief, came to know of your
> 50th from Salim Bhai. I was hoping to catch you at the Brigade function today… just wanted
> to, you know, wish you a happy, whatever 50th… Basically somehow convey to him that you
> know. Definitely don't look any that close to 50…"*

The deliverable is a **clean, warm, sendable message** — not his words verbatim. Strip
`I just want to`, `you know`, `whatever 50th`, `basically somehow convey`. Keep every
substantive fact he supplied (who told him, which function, the compliment). NDR is
explicit that he wants "a good greeting".

## Composition shape

- **Open with the occasion + the name.** "Happy 50th, Sharab! 🎂"
- **Ground it in the real detail he gave** — how he heard (`Came to know from Salim Bhai`),
  where he hoped to meet (`at the Brigade function today but didn't see you there`). This is
  what makes it read as personal rather than boilerplate.
- **Land the compliment** he asked for, phrased so it flatters rather than teases:
  "you don't look anywhere close to 50 — whatever you're doing is clearly working, keep it up".
- **Warm close.** "Here's to good health and many more great years. Cheers!"
- Length: 3–5 sentences. One emoji at most, at the occasion point. Light `!` is fine; no
  exclamation-stacking.

## Delivery

1. Resolve the contact with `contact_resolver` — it returns the phone. **If several matches
   come back (common for names like "Reddy"), pick the one whose score/context matches and
   name the person you used in the reply** so NDR can correct.
2. Call `whatsapp_link` with `platform='telegram'` — **never hand-build the URL**.
3. Report it as a **generated link he can tap**, never as a message that was sent.

## Pitfalls

- **Don't invent detail he didn't give.** No fabricated shared memories, no guessed
  relationship ("friend" vs "acquaintance"). If the dictation says "acquaintance plus we did
  some transaction", the message stays warm but not intimate.
- **Don't correct his premise.** If he says "50th" for someone and the record says otherwise,
  still write the 50th greeting and flag the discrepancy separately in your reply.
- **Multiple near-name contacts is the norm** — `Sharab Reddy` returned 5 candidates. Never
  silently pick; state which number you used.
- If he asks for the same greeting on **two channels** (Telegram send + WhatsApp link), compose
  once and emit twice — the Telegram copy can keep bold, the WhatsApp copy should read clean.