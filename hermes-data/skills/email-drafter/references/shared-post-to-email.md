# Forwarding a shared post/article as an email (verified 2026-10-17)

Trigger: NDR says some version of *"I am sharing the x.com article with you … convert this into an
email sending it to <person> with all the links in it"*. He is forwarding an external piece to
someone (often his sons, a colleague, or a partner) and wants the links carried across intact, plus
his own framing of what the recipient should do with it.

## 1. Get the post text

`web_extract` on the post URL works for X/Twitter, e.g.
`https://x.com/<handle>/status/<id>`. The returned `title` carries the full post text even when the
body renders as a login wall, so read the title, not just the content block.

## 2. Resolve EVERY shortened link before drafting — this is the crux

X rewrites every outbound URL to a `t.co` shortener. A link list made of `https://t.co/…` is
useless to a recipient. Resolve each one and capture the real target:

```bash
for u in <code1> <code2> <code3> <code4>; do
  printf "%s -> " "$u"
  curl -s -o /dev/null -w '%{url_effective}\n' -L --max-time 25 "https://t.co/$u"
done
```

Copy the codes character-for-character out of the extracted text — a mistyped code resolves to
nothing and you will not notice until the recipient clicks it.

Worked example (Samnang Aing's "3D floor plan" post) — four `t.co` codes resolved to a live demo,
a starter-prompt page, a GitHub repo, and a YouTube video. All four came back in one loop.

## 3. Read the resolved targets before writing the body

Do not ship a bare link dump. `web_extract` the repo and the demo page so the email can say what the
thing actually IS (stack, how it was built, what it does) in one or two sentences. That is what makes
the email useful rather than a forwarded URL list. Cite the real stack (e.g. Three.js + GSAP
ScrollTrigger + Lenis, plain files, no build step) — not a guess.

## 4. Draft in HTML, draft only

`gws_gmail_draft_create` with `html=true`, `service_name="google-draas"`. Structure NDR dictated:
a short intro paragraph, the link list, then his two-part framing (what to try first, what to do
with it next), then who else is cc'd. Follow the house rules in the parent skill — 2–3 paragraphs,
no forced line breaks, plain URLs rather than code blocks.

**Report the resolved links back in the Telegram reply**, not just the draft id. He cannot see the
draft's contents without opening Gmail, and the whole point of the ask was that the links survive.

## 5. Pitfalls

- **Don't silently "correct" a dictated product/technology name — correct it and SAY so.** He
  dictated *"his Deep Seat and 3.js"*; the email went out as "DeepSeek and 3.js". That is the right
  reading, but a correction of a dictated proper noun belongs in the Telegram reply
  (*"you said 'Deep Seat' — I've written DeepSeek; say the word if you meant something else"*), the
  same way the `whatsapp-reply` skill handles a substituted surname. Silently fixing it is the
  failure mode; so is leaving an obviously garbled name in front of a recipient.
- **The recipient/cc set may be family.** Ruhaan `pebblyshark69@gmail.com`, Rivaan
  `rankarivaan@gmail.com` — resolve via `contact_resolver`, and note that the dictated name may be a
  misspelling of the saved one (he said "Rivan"; the contact is **Rivaan**).
- **He may ask for "both of them can experiment with it tomorrow"** — that is the cc instruction,
  and the closing line of the email. Keep it as his phrasing, not a summary of it.
- The draft lives in Drafts; he sends it. Never report it as sent.
