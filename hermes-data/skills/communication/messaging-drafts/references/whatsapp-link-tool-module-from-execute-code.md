# Calling the sanctioned whatsapp_link tool when it's not in the live tool list

Verified 2026-09-23 (Ankush Musaddi / Infimax land-clarification message, +91 98208 93856).

When the `whatsapp_link` tool is absent from the session's tool list, its module is still
importable from `execute_code` — this is the SAME sanctioned tool, NOT a manual URL
construction fallback:

```python
import sys; sys.path.insert(0, '/opt/hermes')
from tools.whatsapp_link_tool import whatsapp_link_tool
result = whatsapp_link_tool({'phone': '+919820893856',
                             'text': "Full message text with & # ( ) ' + = ! em-dash — all fine",
                             'platform': 'telegram'})
```

Key facts:
- `result` is a JSON **string** with fields: `phone`, `text`, `url`, `display_text`,
  `display_link`, `mode`, `note`. Use `json.loads` or `str()` and extract.
- The **`url` field is the real deep link** (`https://api.whatsapp.com/send?phone=...&text=...`)
  — hand THIS to the user. `display_text`/`display_link` are Telegram-markdown-escaped
  variants (backslash-escaped periods etc.), NOT the plain URL.
- Pass the phone WITH `+`; the tool normalizes it. Pass the FULL RAW message — never
  pre-escape `&`, `#`, `(`, `)`, `—`, apostrophes, newlines; the tool handles encoding.
- Long messages auto-split into a `parts` array; when `split: true`, deliver each part
  as its own Telegram message.
- Note field reminds: `mode: deep_link`, `delivered: false` — report the link as
  tap-to-send, never as "message sent".
- The SOUL hard rule (never hand-encode `api.whatsapp.com/send`, never `wa.me` — its
  server-side redirect corrupts `%` on mobile) is satisfied ONLY by calling this module
  or the native tool. Do not build the URL by string formatting under any circumstances.
- If the module import fails entirely, tell the user you cannot generate the link rather
  than fabricating one.