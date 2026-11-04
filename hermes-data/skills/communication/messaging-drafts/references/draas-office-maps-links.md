# DRA Office Google Maps Pins & Group WhatsApp Delivery (verified 2026-10-11)

Use when the user asks for a WhatsApp/email message that must include the DRA
office address **plus a Google Maps link**, or any generic "share the office
location" request.

## The DRA office (Prism Greystone)

- Operational / corporate office (all DRA group entities):
  **204-206, Prism Greystone, Cunningham Road, Bengaluru - 560052**
- Registered office is DIFFERENT: Ranka Chambers, 31 Cunningham Rd (see below).

## Finding the correct Google Maps pin — decoy trap

- **"Prism Greystone" the building has NO Google Maps pin.** Searching that
  string returns unrelated listings:
  - **"Prizm Developers" (70, Cunningham Rd, Vasanth Nagar 560001)** — a
    DIFFERENT construction company, not DRA. Its name is likely derived from
    the building. Do not use this pin.
  - "Prism Inc" (lighting store, Richmond Town), "Prism Innovations"
    (architect, Vijayanagar), "Prizm Azalea" (apartments, Benson Town) — all
    unrelated.
- **The correct pin is the company listing "DRA Realty Pvt Ltd"** — website
  drahomes.in, plus code XHQV+P5R, Vasanth Nagar, Bengaluru 560001,
  place_id `ChIJe86JqmYXrjsRZTs_gHiijF0`.
- Developer/search-for-the-COMPANY, never search-for-the-BUILDING.

**Verified office link (browser-confirmed 2026-10-11):**
```
https://www.google.com/maps/search/?api=1&query=DRA%20Realty%20Pvt%20Ltd&query_place_id=ChIJe86JqmYXrjsRZTs_gHiijF0
```

Related DRA listings (do not mix up):
- "D.R.A. Projects Private Limited" → 4, Ranka Chambers, 31 Cunningham Rd,
  Vasanth Nagar 560001 (registered-office pin, place_id ChIJV2lioGgWrjsR0-DpFzI8gDs).
- "DRA Homes" (Chennai, Royapettah, Lloyds Rd) — separate city.

## Re-verifying a Maps pin (Apify google-places actor)

Working input schema (two failed attempts before this worked):
- Actor: `compass/crawler-google-places` (also exposed as preset `google-places`).
- Input keys: **`searchStringsArray`** (NOT `queries` — the preset rejects
  `queries` with "INVALID INPUT: ... searchStringsArray ..."), plus
  `maxCrawledPlacesPerSearch` (small, 3-5), `language: "en"`,
  **`countryCode` must be LOWERCASE** (`"in"`); uppercase `"IN"` → HTTP 400
  invalid-input.
- Returns per listing: `placeId`, `cid`, `url` (query_place_id form), lat/lng,
  `website`, `phone`, `address`.
- Browser-verify the returned `url` before quoting it in a message — the
  listing should resolve to the expected company name + website.

## WhatsApp GROUP messages (no phone)

- Groups have no phone number: call `whatsapp_link` with **text only** (omit
  `phone`). It produces a deep link that opens WhatsApp with the text
  pre-filled; the user picks the group and sends. It does NOT deliver.
- On Telegram, deliver the `display_link` (MarkdownV2-safe) — never the bare
  URL, never a wa.me link.
- Long messages come back as a `parts` array — send EACH part as its own
  Telegram message (Telegram caps at 4096 chars and its splitter breaks links).
- A Google Maps URL inside the message text gets its `&` encoded as the
  fullwidth `%EF%BC%86` — that is CORRECT and expected. Do not "clean" it.
- Address line to use verbatim in messages:
  `DRA Office — 204-206, Prism Greystone, Cunningham Road, Bengaluru - 560052`