# WhatsApp link with NO recipient preset (phone omitted)

Verified 2026-10-13 (Yerdhikanali property brief dictated for Aamir).

## Trigger

Either of these:

- NDR says *"create whatsapp link w/o phone number"* / *"just give me the link, I'll pick the
  contact"*.
- The recipient's number cannot be confirmed — the contacts sheet holds a single **unlabelled**
  phone (the `contact-phone-lookup` LABEL-WINS rule says never use an unlabelled / `Mobile` number),
  so you asked, and he answered with the phone-less link.

## The call

```python
whatsapp_link(text="<full message, verbatim>", platform="telegram")   # phone OMITTED
```

- **Omit `phone` entirely** — do not pass an empty string, and never hand-build the URL.
- The tool is the ONLY sanctioned producer of a WhatsApp deep link; manual
  `api.whatsapp.com/send` construction (or `wa.me`, which corrupts `%` → `�` on mobile) is banned.
- Pass the message text **as-is** — do not pre-strip or escape `&`, `#`, `(`, `)`, `'`, `+`, `—`
  or emoji; the tool percent-encodes correctly.

## Result + delivery

- Returns `api.whatsapp.com/send?text=…` with **no recipient**: opening it launches WhatsApp with
  the text pre-filled and lets NDR choose the contact from his own list. That is the whole point —
  the task completes without a number we cannot vouch for.
- Deliver the **`display_link`** (MarkdownV2-safe inline link), never the bare `url` — Telegram
  re-wraps a bare URL and can cut it.
- If the response carries `split: true`, deliver **each `parts[]` entry as its own separate
  Telegram message**; combining them breaks the link at Telegram's 4096-char cap.
- Never report it as "sent" / "delivered" — a deep link is not a delivery. Say "generated a link
  you can tap; nothing has been sent."

## Pair it with the caveats you already owed

A phone-less link does not retire the open questions. Still state, in the same reply:

- the number you could **not** confirm (and that the link therefore has no recipient);
- any proper noun you could not verify against Drive / mail / Kelsa / the calendar (here:
  "Yerdhikanali" and "Bayappa" appeared nowhere) — ask for the correct spelling.

## Do NOT

- Do not stall the task waiting for a number when a phone-less link answers the request.
- Do not use an unlabelled number just to make the link "complete".
- Do not caption the message with `*Project — Subject*` for an external/vendor recipient — that
  caption convention is for DRA colleagues; vendor messages go uncaptioned.
