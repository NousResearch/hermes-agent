# Telegram Delivery & Recipient Resolution

How to actually get a message into a named colleague's Telegram chat — and the
two ways this goes wrong.

## 1. `send_message(action='list')` is NOT the full directory

`send_message(action='list')` returns the **known target list**, but that list is
not exhaustive. A person can be fully reachable by name while being **absent from
the list output**.

Verified 2026-09-26: `action='list'` returned only
`Nishant Ranka, Bharat Hawaldar, Prakash Singh, Anbarasan M, Sarthak Sharma,
Roshini Ranka, Ruhaan Ranka` — no VK Das. Yet
`send_message(platform='telegram', recipient_name='Vinod Kumar Das', ...)`
delivered successfully (resolved to a real chat_id, message_id returned).

**Rule:** never tell the user *"X isn't on Telegram"* on the strength of the list
alone. The list tells you what the tool will *offer*; it does not tell you what
the identity registry can *resolve*. Attempt the named send and report the
actual result.

## 2. ⚠️ NEVER probe reachability with a real message

Because the list is not authoritative, the temptation is to "test" the
resolution by sending something. **Do not.**

Sending `"Test — please ignore."` to check whether a recipient resolves puts a
stray, confusing message into a colleague's actual chat — it is a real message
to a real person, indistinguishable from a mistake. Verified as a genuine error
2026-09-26 (a probe to VK Das, then the two real messages).

**Correct handling when you are unsure whether a recipient resolves:** just send
the real message. If it fails, you learn the same thing and nobody received
noise. If you need certainty *before* composing, resolve the name with
`contact_resolver` instead — that is a lookup, not a send.

## 3. Resolve the name via the contact sheet first

Use `contact_resolver` for the canonical form and the right channels. People in
NDR's orbit are called several different things, and the sheet holds all the
variants:

- **Vinod Kumar Das (Rahul)** — NDR says "VK Das", "Vinod Kumar Das", and
  "Rahul" for the same person. `vkdas@draas.com`. (NDR delegates statutory
  notices to "Rahul" = VK Das.)
- Always check the **phone label** before using a number (work/DRA only; see the
  contact-phone rule) — but for Telegram delivery you don't need the number at
  all, just the name.

## 4. Delivery conventions for NDR's colleagues

- **Caption colleagues** with *Project — Subject* as the first line (e.g.
  `*Ranka 99 — Chronology, Case & Event Note*`); third parties/vendors go
  uncaptioned.
- **Plain links, not code blocks** — NDR reads these on mobile and a fenced
  block truncates the tap target.
- **One message per document.** When sending several artifacts to the same
  person, send each as its **own** Telegram message. NDR asks for this
  explicitly ("send him all of that *separate* telegram message"); merging two
  documents into one message makes the links ambiguous.
- **Give a direct Google link**, not a description of where to look:
  `https://docs.google.com/document/d/<id>/edit` and
  `https://drive.google.com/file/d/<id>/view`.

## 5. Pairing with a Drive share

When the message shares a file link, grant the access first and **report the
share state in the completion message to NDR** — recipients cannot see their own
permission, and NDR cannot see it either from the chat. State the role and, if
you set one, the **actual ISO expiry timestamp** (see
`google-workspace/references/drive-share-pitfalls.md` §1).

## 6. Explain the document, don't just link it

When the user says *"explain what each of the parts are about"*, walk the
document's own section list (Part A/B/C…, or its headings) with one line each,
lift the single most important takeaway as a headline, and **flag any
unverified/caveated content the document itself discloses** (e.g. a "NOT
INDEPENDENTLY VERIFIED" section). Read the document before writing this — never
paraphrase from the filename.
