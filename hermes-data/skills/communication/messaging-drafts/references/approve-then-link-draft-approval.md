# Approve-then-link: draft approval BEFORE generating WhatsApp links

NDR-corrected workflow, 2026-09-19 (Ranka Northstar landowner broadcasts).

## Two standing format rules — apply by default, don't wait to be asked

Confirmed again 2026-09-21 (Serenity Hillview → MP → DC). These are now defaults for any
substantive WhatsApp draft, and **especially** for anything that will be forwarded:

1. **WhatsApp bold is `*single asterisks*`** — not Markdown `**`. It survives
   `api.whatsapp.com` encoding and renders bold in the app.
2. **Lay identifiers out as `*Label:*` + value, one per line**, in this order: application/file
   number, survey number, status — then the narrative. He asks for this explicitly:
   *"highlight the relevant portions in bold, create sections that makes it easy to read …
   highlighting relevant portion, application number, survey number, status, stuff like that."*

```text
Respected Sir,

*Subject:* Cancellation of land conversion order - Sy. No. 93/2, Huralagurki Village, ...

*Land:* 6 acres 26 guntas.
*Conversion Order:* Official Memorandum No. 239145 dated 28-07-2021 (residential).
*Application pending:* File No. BRDC-DVH0ALN1/1/2026, Comp. No. 1054046.
```

**Bold the label, not the value** — the labels are what the reader scans for. One identifier per
line; do not run file number, comp. no. and survey number into prose.

## The rule

For any substantive message to a third party or group, **present the full text for
approval first, and generate the links only after he approves.** Verbatim:

> "Can you redo draft two and again, present it to me as text before I, after which,
> once I approve it, give me the WhatsApp links for it."

The anti-pattern is producing the draft already wrapped in links: he then has to ask
for the text separately, and a link containing unapproved wording has already been
created. Text first. Links on approval.

## Show only the revised section when he revises a section

When he corrects one part of a long message, re-present **only that part**:

> "After making these changes, just show me the two objective portion of the draft.
> Once I approve it, we can go ahead and make WhatsApp link for the entire message."

So: show the revised section → on approval, regenerate links for the **whole**
message (not just the changed section). Do not re-dump the entire draft on every
iteration — it is long and he is reading on a phone.

## Expect several rounds — apply, don't defend

High-stakes broadcasts went through 3–4 rounds, each one a factual or tonal
correction that was authoritative:

- a word confusion corrected ("ears" → "heirs" — a legal term, not a typo: watch for
  near-homophones with legal meaning in voice dictation);
- scope narrowed ("there is only refundable… there is no non-refundable at all" —
  remove the category entirely, do not merely de-emphasise it);
- a framing change (developer's own position stated as settled, then the *open* matter
  attributed to the landowner group);
- tone softened ("let me give you the final take" → "final take… hoping this date
  works for everyone").

Apply each correction in full and re-present. Do not argue the earlier wording.

## Offer two variants when a message is high-stakes

He asks for "one or two drafts if required". Produce a **full** version and a
**concise** version, label them Draft 1 / Draft 2, state the trade-off in one line,
and let him choose. Do not merge them into one compromise.

## If he asks for the draft "in HTML"

He sometimes wants to read and compare drafts as rendered HTML rather than a wall of
chat text. When he asks, write it as a styled HTML file (WhatsApp-chat look) and give
him the path — see the `visual-rendering` skill for rendering text content to a clean
page. This is for *reading/choosing*, not delivery.

## Link delivery mechanics (unchanged, but easy to get wrong)

- Generate via the sanctioned tool. Never hand-encode a URL.
- If the result has `split: true`, deliver **each part as its own separate bubble** —
  one short label + the raw tool URL per part. Do not combine parts into one reply;
  he has reported merged/garbled parts when they are combined.
- Tell him each part is a separate WhatsApp message to send in order.
- End-to-end for a group: links carry no `phone=` parameter; he picks the group.
- One matter per recipient per message — if two unrelated items are owed to the same
  person, that is two links, not one long message.

**See also:** `references/landowner-group-milestone-update.md` (the six-block
broadcast structure) and `references/ndr-followup-patterns.md`.
