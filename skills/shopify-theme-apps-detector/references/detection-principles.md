# Detection Principles & Evidence Reference

## Shopify Detection Signals (Priority Order)

| Signal | Source | Reliability |
|--------|--------|-------------|
| `cdn.shopify.com` asset references | HTML body | Highest — every Shopify store serves assets via Shopify CDN |
| `myshopify.com` domain reference (e.g. `Shopify.shop`) | HTML body | Highest — explicit store identifier |
| `powered-by: Shopify` response header | HTTP HEAD | Highest — set by Shopify infrastructure |
| `window.Shopify` JS object in HTML | HTML body | Highest — injected by Shopify platform |

**Non-Shopify indicators**: absence of all `cdn.shopify.com`, `myshopify.com`, `powered-by: Shopify`, and `window.Shopify` signals.

## Theme Detection

### Primary: `window.Shopify.theme` object

Extracted from inline script in HTML. Fields:
```json
{
  "name": "Dawn",
  "id": 166598836522,
  "schema_name": "dawn",
  "schema_version": "15.0.0",
  "theme_store_id": 887,
  "role": "main"
}
```

- `theme_store_id` is a 4-5 digit number → official Theme Store theme → link to `https://themes.shopify.com/themes/<theme_store_id>`
- `theme_store_id: null` → custom or third-party theme → web-search `schema_name` to find official page
- `name` is merchant-customizable and unreliable for identification; use `schema_name` instead

### Secondary: `server-timing` header

`theme;desc="166598836522"` — this is the store-specific entity ID (9+ digits), NOT the `theme_store_id`. Use only to cross-reference with `Shopify.theme.id`.

### Tertiary: CSS class namespaces

Unique class prefixes that appear consistently across pages indicate the theme family:
- `product__title`, `product__info-container` → Dawn / OS 2.0 pattern
- `product-single__title` → Debut / OS 1.0 pattern
- Custom prefixes (e.g. `lumia-`, `prd--`) → third-party theme, web-search to identify

## App Detection

### Tier 1: External Script URLs (Highest Confidence)

Scripts loaded from non-Shopify domains are the strongest app signal:

```
https://static.klaviyo.com/...          → Klaviyo (email marketing)
https://config.gorgias.chat/...         → Gorgias (customer support)
https://cdn.judge.me/...                → Judge.me (reviews)
https://cdn.loox.io/...                 → Loox (reviews)
https://a.klaviyo.com/...               → Klaviyo
https://code.tidio.co/...               → Tidio (chat)
https://widget.intercom.io/...          → Intercom (chat)
https://js.hs-scripts.com/...           → HubSpot
https://www.googletagmanager.com/...    → Google Tag Manager (not an app, but confirms GTM)
https://connect.facebook.net/...        → Meta Pixel (not an app)
```

### Tier 2: App Embed Blocks (High Confidence)

`cdn.shopify.com/extensions/<uuid>/<app-slug>/assets/` URLs are Shopify app embed blocks. Extract the `<app-slug>` segment:

```
cdn.shopify.com/extensions/019e3fc0.../js-client-286/assets/pushowl-shopify.js
                                        ↑ app slug: "js-client" → PushOwl
cdn.shopify.com/extensions/019d9c68.../axon-shop-integration-83/assets/app-embed.js
                                        ↑ app slug: "axon-shop-integration" → Axon
```

Web-search the slug: `"<slug>" Shopify app`

### Tier 3: Window Globals (Medium Confidence)

```
_learnq, klaviyo          → Klaviyo
GorgiasChat               → Gorgias
jdgm, judgeme             → Judge.me
Loox                      → Loox
Yotpo                     → Yotpo
tidioChatApi              → Tidio
Intercom                  → Intercom
TriplePixelData           → Triple Whale (analytics)
gladlyConfig              → Gladly (customer service)
```

### Tier 4: CSS Class Namespaces (Lower Confidence)

```
jdgm-*      → Judge.me
loox-*      → Loox
yotpo-*     → Yotpo
stamped-*   → Stamped.io
okendo-*    → Okendo
```

### Tier 5: data-* Attributes (Supplementary)

```
data-yotpo-product-id     → Yotpo
data-raters               → Ryviu or similar
data-judgeme-*            → Judge.me
```

## HTML Report Design Spec

### Required Sections

1. **Header bar** — store URL, scan date, Shopify badge, summary counts
2. **Theme card** — name, schema_name, version, confidence, link, evidence
3. **Apps grid** — card per app: logo, name, category, confidence, App Store link, evidence
4. **Unconfirmed clues** — table of ambiguous signals
5. **Technical appendix** — all scripts, globals, pages crawled (collapsible)

### App Store Links

- Known app: `https://apps.shopify.com/<app-handle>` (find via web search)
- Unknown app: `https://apps.shopify.com/search?q=<app-name>`

### Theme Store Links

- Theme Store theme: `https://themes.shopify.com/themes/<theme_store_id>`
- Custom theme: link to official vendor page if found via web search

### Logo Sources

1. `https://logo.clearbit.com/<vendor-domain>` (e.g. `logo.clearbit.com/klaviyo.com`)
2. Fallback: inline SVG generic icon

### Confidence Color Coding

- HIGH: `#22c55e` (green)
- MEDIUM: `#f59e0b` (amber)
- LOW: `#94a3b8` (slate)

## Common False Positives to Avoid

| Signal | Looks Like | Actually |
|--------|-----------|---------|
| `www.googletagmanager.com/gtag/js` | App | Google Analytics / GTM — not a Shopify app |
| `connect.facebook.net/en_US/fbevents.js` | App | Meta Pixel — not a Shopify app |
| `cdn.shopify.com/s/files/...` | App | Shopify core CDN — not an app |
| `shop.app/checkouts/...` | App | Shopify Shop Pay — native feature |
| `cdn.shopify.com/shopifycloud/...` | App | Shopify platform scripts — not an app |

Always web-search before marking as confirmed. GTM and Meta Pixel should be noted as "tracking/analytics" but not listed as Shopify apps.
