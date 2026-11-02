# WhatsApp Deep-Link DELIVERY — routing, encoding, and the "the encoding is not working" fix

Class of task (learned 2026-10-13, *Palaksha* judgment → Ranka NorthStar TDR messages).
The `whatsapp_link` tool GENERATES a link; it never sends. Everything below is about
**delivering that link to NDR** so it actually works when he taps it.

> NDR's exact pushback, verbatim: *"Can you check the number again because the encoding
> is not thing. Cross check the number and cross check the number you used in the
> WhatsApp encoding please. Give me your new WhatsApp link."*

---

## 1. Deliver a long link as its OWN message

A deep link carrying a full dictated message runs **2,500–3,000 characters**. Telegram
caps a single message at 4096. If you paste that link into a reply that also carries the
summary of what you did, the combined message exceeds the cap, Telegram splits it, and
**the split lands in the middle of the URL** — the link arrives dead and NDR reads that
as an encoding problem.

Rule: when the link is long, send it via `send_message` **as its own message**
(`target='telegram'` for the home channel, or `recipient_name=...` for a person), then
write the summary as a separate turn. Short links (a few hundred chars) can stay inline.

## 2. Raw `url`, not escaped `display_link`, for long payloads

`whatsapp_link` returns three things:

- `url` — the raw `api.whatsapp.com/send?...` link
- `display_text` — MarkdownV2-escaped copy of the message
- `display_link` — `[<display_text>](<url>)`, i.e. the escaped text wrapped as an inline link

`display_link` is correct for SHORT links shown inline in a Telegram reply — it stops
`—`, `.`, `-`, `|`, `(` `)` breaking the parser. For a long dictated message the escaped
form is thousands of extra backslashes and a second way to break. **Use the raw `url`**
for anything long. If the tool returns `split: true` + a `parts` array, send each part as
its own message — never combine.

## 3. Write "and", not "&", in case names

The tool encodes `&` as the fullwidth ampersand `%EF%BC%86` — deliberate, because `%26`
is corrupted to `�` by WhatsApp's server-side redirect on mobile. It is legitimate, but
it is the first thing NDR spots and questions.

Where you control the text, avoid the `&` entirely: *"Dr G.V. Palaksha **and** Anr. v.
State of Karnataka **and** Ors."* — identical meaning, no exotic encoding. Keep `&` only
when quoting a document verbatim.

Never pre-process or strip characters before handing text to the tool, and never
hand-build or "clean" the URL afterwards.

## 4. Verify the number against the contacts sheet BEFORE generating

Resolve, then read the real row — do not trust memory or an older session note (a stale
session note had `98868 54455` for Mohan Sir where the sheet says `98868 85455`).

```python
# sheet: "NDR DRAAS Google contacts"  1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g
# tab:   "NDR DRAAS Google contacts.csv"
# read header row 1, then the person's row, cols A..AC
#   AB = "Phone 1 - Label"   AC = "Phone 1 - Value"
```

Then **state the encoded digits back to him** — `phone=919886885455` — so he can eyeball
them against what he has in his phone.

**Labelled-number rule.** NDR's hard rule is to use a number labelled Primary / DRA /
Work and never an unlabelled one; a number whose only label is `Mobile` should be
flagged for confirmation before it is used for anything consequential. Mohan Sir's entry
(row 4242, "Mohan Sir | JDTP GBA | East", org "JDTP GBA – East Zone") carries exactly one
number, labelled `Mobile` — flag it, give the link anyway, and ask him to confirm the
digits. Do not stall the deliverable over it.

## 5. The fix sequence when he says the encoding is broken

1. Re-read the contacts-sheet row and the `Phone 1 - Value`.
2. Regenerate with `whatsapp_link` (fresh link — do not edit the old one).
3. Send the raw `url` **alone**, as its own message.
4. Quote the encoded number (`phone=91…`) and the source of the number in the reply.
5. If a digit conflict exists between sources, name both variants and ask which is right.

## 6. Recipient routing — Telegram vs WhatsApp-only

| Recipient | Channel |
|---|---|
| Colleague/advocate **in the identity registry** (e.g. Rahul = **Vinod Kumar Das**, vkdas@draas.com — he IS reachable, as "Vinod Kumar Das") | `send_message(recipient_name="Vinod Kumar Das", platform="telegram")` — a real send; still get NDR's go first per `confirm-before-actions` |
| Colleague **not** on Telegram (e.g. Mohan Sir / Mohan JDTP, JDTP GBA East, +91 98868 85455 — WhatsApp only, **no email on record**) | `whatsapp_link` deep link — NDR taps it himself |
| Government officials / external parties | WhatsApp deep link, captioned `*<Project> — <subject>*` |

When NDR says *"send him on telegram, you don't need to send him on WhatsApp"*, that is
the registry check resolving to a Telegram identity — send it, don't generate a link.

## 7. Worked example (2026-10-13)

- Recipient: **Mohan Sir / Mohan JDTP**, JDTP GBA – East Zone, +91 98868 85455.
- Message: `*Ranka NorthStar — TDR for the southern CDP road | HC Karnataka judgment*`
  + his dictated body + the citation + the Drive link + "Just wanted your opinion on the
  same."
- First attempt: the deep link was pasted **inside a long summary reply** → NDR reported
  the encoding was broken.
- Fix: regenerated, replaced `&` with "and", sent the raw URL **as its own message**,
  and quoted `phone=919886885455` with its contacts-sheet source. Accepted.
