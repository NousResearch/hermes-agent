# Public Data Extraction — No-API Mode

Use this reference when the merchant chooses read-only (public storefront) mode and provides only a store domain or product URL. This guide explains how to extract every available product field using Shopify's built-in public JSON endpoints and HTML scraping, so the full audit report can still be generated despite having zero API permissions.

## Overview

When the merchant gives a product URL instead of API credentials:

```text
Shopify Admin API fields available:   ~18 fields
Public JSON + HTML fields available:  ~16 fields (89%)
All fields can be produced from public data.

## Step-by-Step: Extract Data from a Product URL

### 1. Parse the URL

Given a URL like `https://www.your-store.com/collections/dog-harnesses/products/flora-dog-walking-set`:

```javascript
const url = new URL(userUrl);
const storeDomain = url.hostname;                              // "www.your-store.com"
const pathParts = url.pathname.split("/").filter(Boolean);
const productsIndex = pathParts.indexOf("products");
const handle = productsIndex !== -1 ? pathParts[productsIndex + 1] : null;  // "flora-dog-walking-set"
```

Then resolve the store domain to `.myshopify.com` (same method as the connected mode):
- If `storeDomain` ends with `.myshopify.com`, use directly
- If it is `admin.shopify.com/store/<name>`, extract to `<name>.myshopify.com`
- Otherwise, fetch the storefront HTML and extract `Shopify.shop = "..."`

### 2. Fetch Product JSON (Primary Source)

```
GET https://{store-domain}/products/{handle}.json
```

No headers needed. Returns:

| JSON Key | Maps To | Used For |
|----------|---------|----------|
| `title` | Product title | Asset title score, recommended title |
| `body_html` | Product description (full HTML) | Asset description score, recommended description |
| `vendor` | Brand | Evidence context |
| `product_type` | Category | Intent classification |
| `tags` | Comma-separated tags | Evidence, context |
| `handle` | URL slug | Constructing product URL |
| `variants[].title` | Variant name (Size/Color) | Pricing context |
| `variants[].price` | Variant price | Pricing context |
| `variants[].sku` | SKU | Inventory context |
| `variants[].inventory_quantity` | Stock count | Opportunity signal |
| `variants[].available_for_sale` | Availability | Exclusion signal |
| `images[].src` | Full CDN image URL | Download for vision analysis |
| `images[].alt` | Existing alt text | Alt text audit, score |
| `images[].width` | Image width | Image metadata |
| `images[].height` | Image height | Image metadata |
| `options[].name` | Option name (Size, Color) | Variant structure |
| `options[].values` | Option values | Variant structure |
| `published_at` | Publication timestamp | Default: assume active |

**Example response:**

```json
{
  "product": {
    "id": 1234567890,
    "title": "Vetreska Flora Dog Harness with backpack & Collar & Lead 3-in-One Set",
    "body_html": "<p>Take your pet's style to the next level...</p>",
    "vendor": "Vetreska",
    "product_type": "Dog Walking Sets",
    "handle": "flora-dog-walking-set",
    "tags": "",
    "published_at": "2024-01-15T10:00:00-05:00",
    "variants": [
      {
        "id": 9876543210,
        "title": "XS",
        "price": "59.99",
        "sku": "FLORA-XS",
        "inventory_quantity": 15,
        "available_for_sale": true
      }
    ],
    "images": [
      {
        "id": 1111111111,
        "alt": "Vetreska Flora 3-in-1 Dog Walking Set in Pink",
        "src": "https://cdn.shopify.com/s/files/1/.../image.jpg",
        "width": 2048,
        "height": 2048
      }
    ],
    "options": [
      { "name": "Size", "values": ["XS", "S", "M", "L"] }
    ]
  }
}
```

### 3. Fetch Product Page HTML (SEO Fields)

```
GET https://{store-domain}/products/{handle}
```

Parse the HTML `<head>` for SEO fields:

```javascript
// Extract from raw HTML
const titleMatch = html.match(/<title>([^<]*)<\/title>/i);
const metaDescMatch = html.match(/<meta\s+name="description"\s+content="([^"]*)"/i);
const canonicalMatch = html.match(/<link\s+rel="canonical"\s+href="([^"]*)"/i);
const ogTitleMatch = html.match(/<meta\s+property="og:title"\s+content="([^"]*)"/i);
const ogDescMatch = html.match(/<meta\s+property="og:description"\s+content="([^"]*)"/i);
const ogImageMatch = html.match(/<meta\s+property="og:image"\s+content="([^"]*)"/i);

const effectiveSeoTitle = titleMatch ? titleMatch[1].trim() : "";
const effectiveMetaDescription = metaDescMatch ? metaDescMatch[1].trim() : "";
const onlineStoreUrl = canonicalMatch ? canonicalMatch[1] : `${domain}/products/${handle}`;
```

**Important: Shopify SEO Fallback Rule**

The `<title>` and `<meta name="description">` tags always show the **effective** value after Shopify's fallback rules are applied:

| Shopify Admin Field | If Empty | Effective HTML Shows | Our Detection |
|-------------------|----------|---------------------|---------------|
| `seo.title` | Falls back to `product.title` | Product title as `<title>` | Cannot distinguish; use `source: "public_html"` |
| `seo.description` | Falls back to first 155 chars of `product.description` | Truncated description as `<meta>` | Cannot distinguish; use `source: "public_html"` |

**Do NOT claim the merchant has custom SEO fields or is using fallbacks** — the HTML alone cannot tell us which case applies. Mark all SEO field sources as `"public_html_effective"`.

### 4. Scan All Products (for batch planning)

```
GET https://{store-domain}/products.json?limit=250&page=1
```

Pagination: increment `page=N` until the response `products` array is empty.

Note: The list endpoint returns product objects **without** `body_html`. You need the individual product JSON for full description data. Use the list endpoint only for scanning and opportunity scoring.

```javascript
async function scanAllProducts(domain) {
  const products = [];
  let page = 1;
  while (true) {
    const response = await fetch(`https://${domain}/products.json?limit=250&page=${page}`);
    const data = await response.json();
    if (!data.products || data.products.length === 0) break;
    products.push(...data.products);
    page++;
  }
  return products;
}
```

### 5. Fetch Collection Data (for narrowing)

```
GET https://{store-domain}/collections.json
```

Returns list of collections with `handle`, `title`, `products_count`.

```
GET https://{store-domain}/collections/{handle}/products.json?limit=250
```

Returns products within that collection (same format as `/products.json`).

## Field Availability Matrix

| Report Field | Admin API | Public JSON | HTML Scrape | Availability in read-only mode |
|-------------|-----------|-------------|-------------|----------------------|
| product title | ✅ | ✅ | ✅ | ✅ Full |
| product description (HTML) | ✅ | ✅ `body_html` | ✅ Parse `<div>` | ✅ Full (prefer JSON) |
| product vendor | ✅ | ✅ | ✅ | ✅ Full |
| product type | ✅ | ✅ | ✅ | ✅ Full |
| product tags | ✅ | ✅ | ✅ | ✅ Full |
| product handle | ✅ | ✅ | ✅ | ✅ Full |
| product URL (onlineStoreUrl) | ✅ | ❌ | ✅ `<link rel="canonical">` | ✅ Constructed |
| product status | ✅ | ❌ | ❌ | ⚠️ Inferred from `published_at` |
| variants (all fields) | ✅ | ✅ | ❌ | ✅ Full from JSON |
| prices | ✅ | ✅ | ✅ | ✅ Full from JSON |
| SKUs | ✅ | ✅ | ❌ | ✅ Full from JSON |
| inventory | ✅ | ✅ | ❌ | ✅ Full from JSON |
| images (all fields) | ✅ | ✅ (CDN) | ✅ (CDN) | ✅ Full, CDN URLs are public |
| image alt text | ✅ | ✅ | ✅ | ✅ Full |
| options | ✅ | ✅ | ✅ | ✅ Full from JSON |
| SEO title (effective) | ✅ | ❌ | ✅ `<title>` | ✅ Effective value only |
| SEO title (custom vs fallback) | ✅ | ❌ | ❌ | ❌ Not available |
| Meta description (effective) | ✅ | ❌ | ✅ `<meta>` | ✅ Effective value only |
| Meta description (custom vs fallback) | ✅ | ❌ | ❌ | ❌ Not available |
| Collection memberships | ✅ | ❌ | ❌ | ❌ Not available |
| Published timestamps | ❌ | ✅ | ❌ | ✅ Full from JSON |

## Public Mode Limitations in the Report

When generating the audit report in read-only mode, apply these adjustments:

### SEO Title / Meta Description Source Labels
- Use `"public_html_effective"` as the source label for both fields.
- Show this note in the report:
  ```text
  "SEO title and meta description values were extracted from the page HTML. The skill cannot determine whether these are custom SEO fields or Shopify default fallbacks — they are shown as-is."
  ```

### Opportunity Scoring Adjustments
- Remove points for "custom SEO field missing" (we cannot detect this).
- Keep points for weak effective SEO title / meta description (we can still score the content).
- Skip `onlineStoreUrl` check — construct URL from domain + handle instead.
- Skip `status` check — assume active if `published_at` is set and the JSON returned successfully.

### Product Status
- If `published_at` is present and non-null, assume product is active.
- If `published_at` is null or missing, warn: "Product status unknown (public mode); treat as draft/unpublished."

## Image Download for Vision Analysis

All image URLs from the public JSON are Shopify CDN URLs that require no authentication:

```
https://cdn.shopify.com/s/files/1/{shop_id}/files/{filename}
```

You can:
1. Download each image to a temp directory using `fetch` or `curl`
2. Open the image through the host environment's image input pathway
3. Generate alt text from actual pixel evidence (mark source as `"vision"`)
4. Fall back to context-only alt text if vision is unavailable

No API scope needed — the CDN is public.

## Public Mode Validation Checklist

Before declaring data extraction complete:

- [ ] Store domain resolved to `.myshopify.com`
- [ ] Product handle extracted from URL
- [ ] `/products/{handle}.json` returned 200 with valid product object
- [ ] `body_html` present and non-empty (descriptions may be HTML)
- [ ] Product page HTML fetched and parsed for `<title>` and `<meta name="description">`
- [ ] CDN image URLs extracted for any vision-based alt text analysis
- [ ] If scanning: `/products.json` pagination works (no blocking)
- [ ] If the store blocks public JSON (403/Cloudflare/etc.), inform the user and suggest connected mode

## Fallback if Public Access Is Blocked

Some Shopify stores block `/products.json` and/or `/products/{handle}.json` via:

- Cloudflare bot protection
- Akamai / edge security
- Custom reverse proxy

If the JSON endpoint returns 403, 404, or a non-JSON response after 3 retries:

1. Inform the user: "Your store's public endpoints appear to be blocked by security measures. The skill needs either API access (connected mode) or storefront JSON access."
2. Suggest the user try a product URL instead of store domain (some stores block JSON on store root but allow it on product pages).
3. If still blocked, recommend switching to connected mode.
4. Do not fall back to full HTML scraping — raw HTML parsing is unreliable for product description, variants, and images.
