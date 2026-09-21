---
name: rightmove-property-search
description: >
  Search UK property listings on Rightmove by location, price, bedrooms, and
  property type. Returns structured listings for property track (sale + family
  home search). No login required for basic searches.
category: browser
version: 1.0.0
---

# Rightmove Property Search

## Target
- URL: https://www.rightmove.co.uk/property-for-sale/search.html?<params>
- Method: Browser (browser_navigate + browser_snapshot)
- Bot detection: Medium (Cloudflare)

## Capabilities
- Search properties for sale across UK by location, price, beds, property type
- Structured results: price, address, beds, agent, image, url
- Pagination support
- No side effects: read-only

## Anti-Detection
- Bot detection level: Medium (Cloudflare)
- CAPTCHA risk: Possible (Cloudflare challenge)
- Rate limit: ~30 requests/hour
- Strategy: 5-10s delays between requests, full browser session, realistic User-Agent
- On Cloudflare challenge: retry once after 15s delay

## Input Parameters
- `locationIdentifier`: e.g., `REGION^94058` (Nottingham), `OUTCODE^231` (NG postcode)
- `minPrice`: integer (£)
- `maxPrice`: integer (£)
- `minBedrooms`: integer
- `maxBedrooms`: integer
- `propertyTypes`: comma-separated (detached, semi-detached, terrace, flat, bungalow)
- `radius`: 0.0, 0.25, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 40.0 (miles)
- `sortType`: 6 (newest first), 1 (highest price), 2 (lowest price)
- `index`: pagination offset (multiples of 24)

## Workflow

### Step 1: Build Search URL
```
https://www.rightmove.co.uk/property-for-sale/search.html
  ?locationIdentifier=REGION^94058
  &minPrice=250000
  &maxPrice=500000
  &minBedrooms=3
  &maxBedrooms=4
  &propertyTypes=detached,semi-detached
  &radius=10.0
  &sortType=6
  &index=0
```

### Step 2: Navigate and Extract
```
browser_navigate("<built_url>")
browser_snapshot(full=true)
```
Check for Cloudflare challenge:
- If challenged: wait 15s, retry once
- If still blocked: report error

### Step 3: Parse Results
Extract from each property card:
- `price` — displayed price (e.g., "£350,000")
- `address` — property address
- `bedrooms` — number of bedrooms
- `propertyType` — detached/semi/terrace/flat
- `agent` — estate agent name
- `image` — thumbnail URL
- `url` — property detail page
- `id` — Rightmove property ID
- `added` — "Added today", "Reduced", etc.
- `sold` — STC (Sold Subject to Contract) indicator

### Step 4: Return
```json
{
  "success": true,
  "query": {
    "location": "Nottingham",
    "min_price": 250000,
    "max_price": 500000,
    "bedrooms": [3, 4],
    "type": ["detached", "semi-detached"]
  },
  "total_results": 45,
  "results": [
    {
      "id": 123456789,
      "price": 350000,
      "address": "123 High Street, Nottingham, NG1 1AA",
      "bedrooms": 3,
      "property_type": "Semi-Detached",
      "agent": "Purple Bricks",
      "added": "Added today",
      "status": "available",
      "image": "https://media.rightmove.co.uk/...",
      "url": "https://www.rightmove.co.uk/properties/123456789"
    }
  ],
  "pagination": {
    "page": 1,
    "total_pages": 2,
    "next_index": 24
  }
}
```

## Integration Points

- **Property track (P3)**: Daily cron 07:00 for new listings → Telegram/Discord alert
- **Kanban**: Price drop >5% → task for review
- **Comparisons**: Track multiple locations, monitor market trends

## Pitfalls

- Cloudflare may block headless browser sessions
- `locationIdentifier` format varies by region/postcode — may need lookup
- Some listings are "Sold Subject to Contract" — flag in output
- Price displayed may be "Offers in Excess of" or "Guide Price" — note format
- Image URLs are temporary — link may expire
- Rightmove changes HTML structure periodically — inspect before extracting
- UK-specific: prices in GBP, distances in miles, addresses follow UK format

## Verification

1. Search for Nottingham properties (£250k-£500k, 3+ beds)
2. Verify results contain expected property data
3. Test pagination (page 2)
4. Test with no results (should handle gracefully)
5. Test price format parsing (£ signs, commas, ranges)
