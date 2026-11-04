# WhatsApp Chunked-Message HTML — Click-to-Send wa.me Pages

**Class-level pattern for: "break a long message into multiple WhatsApp-sized chunks, give me wa.me links so I can click each one and WhatsApp opens with the text pre-typed."**

The user wants to send a long pre-composed message (often a status update, report handoff, or escalation) to a single contact on WhatsApp. WhatsApp itself has no "send multi-part message as one" feature, and pasting a 4,000-character block into the compose box feels risky. The cleanest UX is a single HTML page with one "Open WhatsApp" button per chunk — the user opens the page in their phone browser, clicks each button in order, WhatsApp launches with the chunk pre-filled, they hit Send, swipe back, click the next.

## When to use this pattern

Trigger phrases from the user:
- "create a html page with a link because this is a very long message"
- "break the message up into multiple messages with multiple links"
- "restrict the number of characters to that [WhatsApp limit]"
- "create the encoding and give me the multiple messages accordingly in one html page"
- "so I can click one by one and send everything to [name]"

If the user just wants one message: skip the HTML and use `whatsapp_link` directly.

## Hard limits to respect

- **WhatsApp message body**: 65,536 chars max per single message (very loose).
- **wa.me URL**: 2,048 chars max (URL spec, browsers/WhatsApp clients also enforce this). **This is the binding limit.** A 1,500-char message URL-encodes to ~2,000 chars — use 1,200 chars as the safe per-chunk ceiling.
- **The tool now emits `api.whatsapp.com/send?phone=...&text=...` links (changed 2026-08-07); `wa.me` is stale.** The 2,048 cap applies to these too. The tool's auto-split can still emit over-cap parts (observed 2,166-char part on 2026-09-22) — verify `len(url) ≤ ~1,900` for every part returned and re-chunk at sentence boundaries if a part is over.
- **Per-message practical limit**: keep each chunk ≤ 1,200 chars so the URL stays well under 2,048 even with full URL-encoding of bullets, em-dashes, and special characters.

## Number-format pitfall (CRITICAL)

`wa.me` accepts ONLY digits, no `+`, no spaces, no dashes. For an Indian mobile `+91 94497 84569`:

- ✅ Correct: `wa.me/919449784569` (12 digits: 91 + 10-digit mobile)
- ❌ Common bug: `wa.me/9194497845699` (13 digits — extra trailing `9` from a doubled digit)
- ❌ Other common bugs: leading `0` kept after country code, double country code, leading `+`

**Cross-check the number from TWO independent sources before generating any wa.me URL** (see `contact-phone-lookup` for the lookup order):

1. Google Contacts (People API) — `searchContacts(query=NAME, readMask='names,phoneNumbers,organizations')` — verify org/title matches the role the user is messaging about
2. The source document where the number is written (a letterhead, prescription, business card) — re-OCR it at high DPI if it was hand-circled or annotated, because **circled digits get absorbed into the adjacent digit in OCR output** (e.g. `9449784569` with a circle around the `9` comes out as `944978456` and the `9` is lost)

**Bug the user caught in practice (Jul 2026, KDR pre-op):** I encoded `9194497845699` (trailing `99`) when the correct segment was `919449784569`. The HTML page worked mechanically but every WhatsApp button failed with "phone number not on WhatsApp". The user had to ask me to re-verify against both the Haldipur letterhead and the Google Contacts entry to find the real number.

**Verification step before delivering:** Open the generated HTML in a browser, find the first `wa.me/...` URL, copy just the digit segment, and confirm it's 12 digits (for IN) or 11 digits (for US) — if it has 13, you doubled a digit; if it has 11, you dropped a digit.

## HTML page structure (template)

The page is dark-themed, mobile-first, with one card per chunk:

```
[1/N] Chunk title
  [pre block: monospace, scrollable, max-height 280px]
  [▶ Open WhatsApp with chunk N pre-typed]  ← wa.me button (green)
  [🔗 Open wa.me link]                       ← secondary, lets user inspect the raw URL
  [📋 Copy text]                              ← fallback if WhatsApp link fails
  [✅ marked sent]                            ← appears after click (localStorage)
```

Key design points:

- **Per-chunk `[N/M]` prefix in the message body** so the receiver sees the order, and so accidental mid-thread sends still read sensibly
- **Per-chunk copy button** as a fallback for the rare case where wa.me is down
- **localStorage sent-tracker** so the user can pause, close the tab, come back, and see what's left
- **Telegram file delivery** (the HTML, not a link): saves to `/opt/data/` AND uploads to Drive TMP, then attaches via `MEDIA:/path/to/file.html` in the Telegram response — the user opens the attachment in their phone browser, no auth needed

## Code skeleton

```python
import sys, json
sys.path.insert(0, '/opt/hermes')
from tools.whatsapp_link_tool import whatsapp_link_tool

# NO manual urllib.parse.quote — use the whatsapp_link tool for encoding.
# It handles fullwidth ampersand/hash substitution (%EF%BC%86 / %EF%BC%83)
# which prevents wa.me URLs from breaking on mobile WhatsApp clients.

CORRECT_PHONE = '919449784569'  # NO '+', NO spaces — verify length!
# Omit phone entirely for group messages (wa.me/?text=...) by not passing it.
chunks = [ ... ]  # each ≤ 1200 chars so URL stays under 2048 limit

def wa_url(text, phone=CORRECT_PHONE):
    """Encode one chunk into a wa.me URL using the sanctioned tool."""
    params = {"text": text}
    if phone:
        params["phone"] = phone
    result = json.loads(whatsapp_link_tool(params))
    return result['url']

# Build HTML template, then save local + upload to Drive TMP + MEDIA: deliver
```

For the upload step use `googleapiclient.http.MediaFileUpload(path, mimetype='text/html')` or `MediaInMemoryUpload(html_bytes, mimetype='text/html')` — `media_body=raw_string` alone fails with `UnknownFileType` because googleapiclient can't sniff the type from a string.

## Delivery pattern

1. Build the HTML in `/opt/data/kdr_shridhar_chunks_20260711_v2.html` (date-suffixed, version-suffixed if iterating)
2. Upload to Drive `TMP` folder, **delete any prior `_v1` / `_v2` of the same file first** to avoid stale links
3. Return both `MEDIA:/opt/data/...html` (for phone browser, no auth) AND the Drive link (for desktop preview)
4. Include a sample wa.me URL in the response so the user can sanity-check the phone segment before clicking

## What NOT to do

- Don't put the full message in one chunk because the user might paste it manually — they explicitly said "click one by one and send everything", so chunks are required
- Don't use a single `whatsapp_link` call for the entire multi-chunk message — that generates one URL that blows past the 2,048 limit. Use it PER CHUNK instead (each ≤ 1,200 chars).
- Don't use a different phone number for each chunk — single contact, single conversation
- Don't trust OCR'd numbers from a circled/annotated document without re-rendering at high DPI and visually verifying (see `ocr-and-documents` references)
- Don't deliver the HTML as a Drive-only link — the user opens it on their phone, where Drive opens in the Gmail app, which often blocks wa.me redirects. Local file via Telegram attachment is the reliable path.
