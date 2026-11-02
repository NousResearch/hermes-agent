# WhatsApp GROUP messages — generating the deep link

Verified 2026-10-16 (NDR → a WhatsApp group with Chanakya Semlani and his son Jay).

## The rule

`whatsapp_link` is the **only** sanctioned way to produce a WhatsApp deep link (see SOUL.md). For a
**group** message, call it with `text` and **no `phone`**:

```
whatsapp_link(
    text="Sorry guys, apologies for the delay. ...",
    platform="telegram"       # when the link is being delivered into a Telegram chat
)
```

Omitting `phone` produces a link that opens WhatsApp with the message pre-filled and WhatsApp's own
**recipient picker** on top — the user then taps the group. You cannot address a group by phone
number, and you must not try to guess or synthesise a group ID.

State this in the reply so the user isn't puzzled by the missing recipient: "tap the link, WhatsApp
opens with the text pre-filled, and you just pick the group."

For a **1:1** message always pass `phone` (E.164 or local; 10-digit numbers get ISD 91 prepended).
Example from the same thread: `whatsapp_link(text=..., phone="+1 212 600 4900")`.

## Checklist

1. **Never hand-build the URL.** No `api.whatsapp.com/send?text=...` typed by hand, no `urllib.parse`
   inside `execute_code`, and never `wa.me` (its server-side redirect corrupts `%` into `�` on
   mobile). If `whatsapp_link` isn't in the tool list, say you can't generate the link.
2. **Pass the text through untouched.** `&`, `#`, `(`, `)`, `'`, `+`, `-`, `=`, `.`, `!`, emoji and
   newlines are all handled. Pre-stripping or escaping characters is what breaks the link.
3. **Long messages auto-split.** The tool returns `split: true` and a `parts` array. Deliver **each
   part as its own separate Telegram message** — never concatenate them, or Telegram's own splitter
   cuts a link in half.
4. **Show the message text in the reply before the link.** The user reads and approves the wording
   before tapping; they often want to trim it.
5. **Don't assume group membership.** Take the participants from what the user says ("I have a group
   with Chanakya and his son Jay"). Never add a name to the group framing that the user didn't name.

## Captioning — colleague vs vendor vs family

NDR's standing caption rule (see memory / the contact sheet conventions):

- **Colleagues** → ALWAYS caption: `*Project — Subject*`.
- **Vendors / third parties** → NEVER caption unless explicitly asked.
- **Family and personal** → no caption. A message to a cousin or family friend opens straight into
  the content. In this session the group message opened directly with *"Sorry guys, apologies for
  the delay…"* — adding a project header would have made a family favour read like project
  correspondence.

## Include working links in the message body

When the message asks someone to review documents, put the actual Google Docs URLs in the message
text rather than a vague reference — and use `/edit` links (not `usp=drivesdk`), since the
recipient has to open and edit them:

```
Chanakya, if you can review these two documents and let me know if these are in order:

1. Pour-card digitisation report, Project Ranka Ankura:
https://docs.google.com/document/d/<id>/edit

2. Director's covering letter:
https://docs.google.com/document/d/<id>/edit
```

Confirm the links actually resolve before sending — a draft-stage 404 in a WhatsApp message is not
recoverable.
