# WhatsApp URL Encoding — Research Notes

**Date:** 2026-05-03 (updated 2026-05-07, 2026-07-11)
**Status:** VERIFIED WORKING

## The Rule

**`api.whatsapp.com` is the default.** `wa.me` can be used with fullwidth character replacement (see below).

## Why Both `wa.me` AND `api.whatsapp.com` Break With `%26`

**⚠️ VERIFIED Jun 2026:** Mobile WhatsApp WebViews (Android/iOS) incorrectly parse `%26` (URL-encoded `&`) as a URL query separator on **BOTH** endpoints — not just `wa.me`. On Android, the native URL handler strips `%26` even from `api.whatsapp.com/send` URLs, truncating the message at the `&` character.

This applies regardless of endpoint:
- `%26` inside the `text=` parameter value → Android WhatsApp WebView treats it as a parameter separator → message truncated
- `&` between URL parameters (`phone=...&text=...`) stays unencoded — these are real parameter separators, not part of the message

## Correct Approach — Fullwidth Character Substitution for ALL WhatsApp Links

**⚠️ Both `%26` (ampersand) AND `%25` (percent) break on mobile WhatsApp WebViews.** Replace EVERY problematic character with its fullwidth Unicode equivalent BEFORE URL-encoding:

| Character | Standard | Fullwidth | Unicode | URL Encoding | Why |
|-----------|----------|-----------|---------|--------------|-----|
| `&` (Ampersand) | `&` | `＆` | U+FF06 | `%EF%BC%86` | `%26` interpreted as URL param separator → message truncated |
| `%` (Percent) | `%` | `％` | U+FF05 | `%EF%BC%85` | `%25` triggers double-decode → weird chars in message |
| `=` (Equals) | `=` | `＝` | U+FF1D | `%EF%BC%9D` | `%3D` can break query string parsing |
| `+` (Plus) | `+` | `＋` | U+FF0B | `%EF%BC%8B` | `+` interpreted as space by URL parsers |

**Universal recipe:**
```python
import urllib.parse

phone = '919900093816'  # strip + and spaces
# Replace ALL problematic characters BEFORE encoding
message = message.replace('&', '\uFF06')   # fullwidth ampersand
message = message.replace('%', '\uFF05')   # fullwidth percent
message = message.replace('=', '\uFF1D')   # fullwidth equals
message = message.replace('+', '\uFF0B')   # fullwidth plus
encoded = urllib.parse.quote(message, safe='')
link = f"https://api.whatsapp.com/send?phone={phone}&text={encoded}"
```

**Verify the correct hex:**
```python
import urllib.parse
print(urllib.parse.quote('\uFF06'))  # MUST print %EF%BC%86
print(urllib.parse.quote('\uFF05'))  # MUST print %EF%BC%85
print(urllib.parse.quote('\uFF0B'))  # MUST print %EF%BC%8B
```

## `whatsapp_link` TOOL STATUS — NOW HANDLES FULLWIDTH CORRECTLY

**Historical bug (fixed after 11 Jul 2026):** The built-in `whatsapp_link` tool used to emit raw `%26` for `&` characters which broke on Android WhatsApp WebViews. **This has been fixed.** The tool now performs fullwidth character substitution in `_encode_wa_text()` — after URL encoding, it replaces `%26` → `%EF%BC%86` (fullwidth ＆) and `%23` → `%EF%BC%83` (fullwidth ＃).

**Current behaviour (confirmed Jul 2026):** The `whatsapp_link` tool is SAFE to use for messages containing `&`, `%`, `=`, or `+`. It handles the encoding correctly.

**⚠️ Tool-access caveat:** The `whatsapp_link` tool is registered under the "messaging" toolset and may NOT appear in your direct tool list depending on your session configuration. In that case you do **not** need to hand-encode anything — **call the tool's own handler**, which is the same sanctioned encoder the tool wraps:

```python
import sys, json
sys.path.insert(0, '/opt/hermes')                      # so `tools.` resolves
from tools.whatsapp_link_tool import whatsapp_link_tool
result = json.loads(whatsapp_link_tool(
    {"phone": "+919845890316", "text": message, "platform": "telegram"}))
print(result["url"])            # stdlib-quoted, fullwidth-safe
print(result.get("split"))      # True => deliver result["parts"] as separate bubbles
print(result.get("display_link"))  # MarkdownV2-safe inline link for Telegram
```

- **Import path — use `from tools.whatsapp_link_tool import ...` with `sys.path.insert(0, '/opt/hermes')`.** The shorter `sys.path.insert(0, '/opt/hermes/tools')` + `from whatsapp_link_tool import ...` is the stale form: it works only from a CWD where the module also resolves as a top-level import, and fails with `ModuleNotFoundError` in a plain session CWD. Verified working 2026-09-21.
- `platform="telegram"` returns `display_text` (MarkdownV2-escaped) and `display_link` alongside the raw `url`.
- Reading the handler's source first (`/opt/hermes/tools/whatsapp_link_tool.py`) is a legitimate step when the phone/`text` semantics matter — it documents the encoding contract (fullwidth `&`/`#`, single-encode `%`, ISD-91 auto-prepend for bare 10-digit numbers, and the 4096-unit split budget).
- **If the module genuinely is not present on disk**, only then fall back to the manual fullwidth recipe in "Correct Approach" — and say so, rather than silently substituting a hand-built URL.

Use `platform="telegram"` to get `display_link` (MarkdownV2-escaped inline link for Telegram delivery).

**Fallback (manual encoding):** When the tool is genuinely unavailable (not just missing from tool list), use the fullwidth substitution recipe in the "Correct Approach" section above.

## When to apply fullwidth vs when standard encoding is fine

**Safe to use basic URL encoding (e.g. via `urllib.parse.quote` directly):** the message has none of `&`, `%`, `=`, `+` (most short personal messages). For these, the basic approach is fine and saves a few lines of Python.

**Must use fullwidth substitution:** any message containing `&`, `%`, `=`, or `+`. Default to always-on — the cost is ~5 lines of Python either way.

**Diagnostic if the user reports a truncated message:**
1. Inspect the URL — does the body contain `%26`? If yes, the link was built without the fullwidth workaround. Rebuild it.
2. Does the body contain `％` (`%EF%BC%85`)? That's the fullwidth percent — means the workaround was applied correctly. The user is seeing a different bug.

## Fullwidth Ampersand Sanity Test

```python
def safe_whatsapp_link(phone, message):
    # Apply fullwidth substitution FIRST
    message = message.replace('&', '\uFF06')
    message = message.replace('%', '\uFF05')
    message = message.replace('=', '\uFF1D')
    message = message.replace('+', '\uFF0B')
    # Sanity checks
    assert '\uFF06' in message or '&' not in message, "fullwidth & substitution failed"
    assert '\uFF05' in message or '%' not in message, "fullwidth % substitution failed"
    encoded = urllib.parse.quote(message, safe='')
    base = "https://api.whatsapp.com/send"
    if phone:
        return f"{base}?phone={phone}&text={encoded}"
    return f"{base}?text={encoded}"
```

## Verification (last good link, Jul 2026)

`https://api.whatsapp.com/send?text=Hi%20Eshwari%2C...` — body has no `&`, so direct encoding is safe. Worked end-to-end.

When the user message had no `&` but used punctuation like "—" (em dash) or "₹" (rupee), direct encoding worked fine — these are safe characters.

## Common Pitfalls

1. **Forgetting to check for `&` in proper names** — e.g. "Dr X & Dr Y", "Eshwari & Roshni", "M&A". The user often doesn't notice the `&` in the draft until the link truncates.
2. **Percent signs in prices** — "₹25,000" is fine, but "25% discount" is not. Use fullwidth.
3. **Plus signs in amounts or phone refs** — "Class 10+1", "+91 98800 55634" in body text.
4. **Equals signs in "Dr=DW" type shorthand** — uncommon but possible.

## When in Doubt, Always Apply the Fullwidth Recipe

The cost of always applying it is ~5 lines of Python. The cost of forgetting it is a broken link and a frustrated user. Default to always-on.

---

## Telegram Renders wa.me Links as Markdown — Hidden Pitfall

**Different bug, same symptom (Nishant, 12 Jul 2026):** Standard URL encoding produces a perfectly valid wa.me URL with correct URL-encoding (`%28` for `(`, `%27` for `'`, `%29` for `)`, `%E2%80%94` for em-dash). The URL itself is fine. But **Telegram's Markdown parser** sees the link text inside a markdown message and **mismatches the URL boundaries** — it stops at the first `)` it finds, treats everything after as garbage, and the visible "link" is truncated.

**Root cause:** Telegram treats the wa.me link text as inline Markdown. The text inside the wa.me URL often contains punctuation that Telegram interprets as Markdown link syntax: `[text](url)` ends at the first `)`. When the message body has `(...)`, Telegram thinks the URL ends at the closing paren.

**Three characters that break Telegram rendering** (URL is correct, but Telegram truncates the visible link):

| Character | Example in message | Telegram effect |
|-----------|-------------------|-----------------|
| `(` and `)` | `(#9, #13, #15, #16, #25)` | URL parsing stops at first `)`, rest is junk |
| `'` (apostrophe) | `I've sent you` | Sometimes breaks the link rendering |
| Unbalanced brackets in `[...]` | rare, but possible | Same truncation |

**Rule — when delivering a wa.me link as Telegram chat text:**

1. **Sanitize the message body before generating the link.** Replace `(...)` with ` — ` (em-dash) or ` - ` (hyphen), and `'X` → `X`. Examples:
   - `(#9, #13, #15, #16, #25)` → `items 9, 13, 15, 16, 25`
   - `I've sent you` → `I have sent you`
   - `Viewer (read-only)` → `Viewer / read-only`
2. **Generate the link with the SANITIZED text** using the fullwidth recipe via `execute_code` (see the "Correct Approach" section above). The receiver will see the sanitized version in WhatsApp — if that loses meaning, consider a non-wa.me link (e.g. `gmail-compose` URL, Drive link, or plain text).
3. **Display the link in a code block in Telegram** if you must keep the original punctuation. Code blocks suppress Markdown processing:
   ```
   https://wa.me/9844123300?text=...
   ```
   The link won't be clickable in a code block, but at least the full URL is visible.
4. **Do NOT** edit the URL itself to escape these characters — wa.me and WhatsApp expect literal `(`, `)`, `'` in the decoded text. The fix is in the **text payload**, not the URL.

**Diagnostic if the user says "the link is broken / truncated":**
- Inspect the Telegram message text — does it contain `(`, `)`, `'`?
- If yes: rebuild the message with the sanitized version.
- The wa.me URL itself was probably always correct.

**Independent of the `&` / `%` / `=` / `+` fullwidth issue** — those break on mobile WhatsApp, this one breaks on Telegram. Both can affect the same link. Sanitize for BOTH channels:

```python
# Build a Telegram-safe + WhatsApp-safe message
text = text.replace("(", "").replace(")", "")  # remove parens
text = text.replace("'", "")                    # remove apostrophes
# Then apply the fullwidth substitution for &, %, =, +
text = text.replace('&', '\uFF06').replace('%', '\uFF05').replace('=', '\uFF1D').replace('+', '\uFF0B')
import urllib.parse
encoded = urllib.parse.quote(text, safe='')
link = f"https://wa.me/919844123300?text={encoded}"
```
