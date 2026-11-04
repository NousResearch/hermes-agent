# WhatsApp Link Generation Patterns

## Tool: `whatsapp_link`
The ONLY sanctioned way to produce a WhatsApp deep link. Never hand-encode `api.whatsapp.com/send` or `wa.me` URLs.

### Single-recipient links (phone + text)
```python
whatsapp_link(phone="+919845026390", text="Message here", platform="telegram")
```
Returns: `display_link` (MarkdownV2-safe inline link) and `url` (raw). On Telegram, always use `display_link`.

### Group / forward links (no phone)
```python
whatsapp_link(text="Message for a group or to paste somewhere")
```
Omit `phone` entirely. URL has no `phone` parameter — WhatsApp opens asking where to send. Perfect for:
- Group chats (no single number to target)
- When user wants to copy text and forward manually
- Sharing with a person whose number you don't have but who's on the user's WhatsApp

### Long messages (auto-split)
The tool auto-splits long messages into a `parts` array (one deep link per part) when a single link would exceed 4 KB. Each part is a complete `api.whatsapp.com/send` URL. When `split: true`, deliver **each part as its own Telegram message** — never combine parts into one message (Telegram caps at 4096 chars and its own splitter would cut the URL in half).

### Message formatting
WhatsApp supports:
- `*bold*` for bold
- `_italic_` for italic  
- `~strikethrough~` for strikethrough
- ```` `code` ```` for monospace
- Emojis work fine

NDR preference: colleagues get captioned links (`*Project — Subject*`), vendors get uncaptioned.

### Telegram display
When `platform="telegram"`, the response includes `display_text` (MarkdownV2-escaped) and `display_link` (MarkdownV2-safe inline link). Always call with `platform="telegram"` when delivering the link to a Telegram chat.

**RAW URL delivery** (NDR preference for some links): use `url` from the response, not `display_link`. NDR uses raw URLs for WhatsApp links shared in his own chats so he can see the full URL before tapping.