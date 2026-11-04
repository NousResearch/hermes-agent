# WhatsApp link generation — sanctioned tool module fallback

When the `whatsapp_link` tool is NOT exposed in the current toolset (Telegram
sessions sometimes lack it), do NOT refuse and do NOT hand-encode a URL. The
sanctioned implementation is a Python module on the host and can be called
directly:

```python
import sys; sys.path.insert(0, '/opt/hermes')
from tools.whatsapp_link_tool import whatsapp_link_tool

result = whatsapp_link_tool({'phone': '+91XXXXXXXXXX', 'text': "<full message>", 'platform': 'telegram'})
# result is a JSON *string* with fields: phone, text, url, display_link, mode, delivered, note
```

Worked 2026-09-23 (Ankush Musaddi clarification message, phone +91 98208 93856).

## Return shape (parse the string with json.loads)
- `url` — the deep link: `https://api.whatsapp.com/send?phone=...&text=<url-encoded>`. Use THIS for delivery.
- `display_link` — markdown `[text](url)` form; use when presenting in-chat.
- `note` — always says "deep link ONLY, nothing sent". Report to the user as a
  generated link they tap, never as a message that was sent/delivered.
- `delivered: false`, `mode: "deep_link"`.

## Rules (same hard rules as the tool itself)
- **Never build the URL manually** (no f-strings, no urllib.urlencode by hand,
  no wa.me: its server-side redirect corrupts `%` on mobile). Always call the
  module. The tool handles `&`, `#`, `(`, `)`, `'`, `+`, `-`, `=`, `.`, `!`,
  spaces, emoji, newlines correctly — stripping/escaping them breaks mobile.
- `.dwg`-style file/preview text or long messages: if the tool returns a
  `parts` array (split:true), deliver each part as its own Telegram message.
- Long messages auto-split; always deliver the `url` (or `display_link`)
  exactly as returned, never "cleaned".
- Phone number: use the work/DRA number per the contact-phone rule unless the
  user explicitly says otherwise; for 3rd-party brokers, use the number from
  their own email signature (verified source) when the contact sheet has none.

## Truncated message BODY — the link is fine, the payload was short (NDR, 2026-10-08)

Symptom: NDR reports the link *"just cuts off after Dear Vivek and that's it, comma and that's it."*
This is **not** the URL-length failure in `whatsapp-deep-link-length.md` (that one corrupts the URL
itself). Here the `text` handed to the tool was a **stub** — the body was truncated before encoding,
so the link faithfully opens a one-line message.

Rules:
- Pass the **COMPLETE** message text in one call — never a stub, never hand-abbreviated, never
  "rest as drafted". Whatever string you pass is exactly what the recipient sees.
- **Paste the FULL message text in the reply next to the link.** A bare link gives NDR nothing to
  verify before tapping; he cannot tell a truncated payload from a good one until it opens.
- When he reports truncation: regenerate through the tool with the full text (never patch, re-type or
  salvage the URL) and re-paste the complete text. One clean regeneration — don't narrate the cause
  at length.
- Sanity check before sending: the rendered text length should look like the draft you just wrote.
  A link whose visible text ends at the greeting is the tell.