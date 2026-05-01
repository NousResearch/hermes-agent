---
name: find-deals
description: Use when the user wants to search for cannabis product deals at their preferred Arizona dispensaries. Accepts free-form product queries (e.g., "Khalifa Kush 3.5g flower under $40") and optional filters (brand, type, weight, THC/CBD, price range, deal type). Searches Zen Leaf Chandler, The Flower Shop Ahwatukee, Trulieve Tempe, and Story Cannabis North Chandler.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [shopping, deals, local-search, pricing, retail, dispensaries, cannabis]
    related_skills: [web-search, maps]
---

# Find Deals

## Overview
This skill searches for current cannabis product deals, menu pricing, and promotions at the user's four preferred Arizona recreational dispensaries:

1. **Zen Leaf Chandler** — zenleafdispensaries.com/locations/chandler/menu/recreational
2. **The Flower Shop Ahwatukee** — theflowershopusa.com/ahwatukee/
3. **Trulieve Tempe** — trulieve.com/dispensaries/arizona/tempe
4. **Story Cannabis North Chandler** — storycannabis.com/dispensary-locations/arizona/north-chandler-dispensary/

The user types queries the same way they would into a dispensary website search bar (e.g., "Khalifa Kush 3.5g flower under $40"). The skill parses the query, maps intent to filters, then queries each dispensary's public menu or uses web search to surface matching products, deals, and pricing.

## When to Use
- User asks "find deals for X" where X is a product, strain, brand, or category
- Comparing prices across the user's preferred dispensaries
- Searching for daily specials, first-time patient discounts, or loyalty rewards at these locations
- Filtering menus by weight, potency (THC/CBD), price range, or product type

## Query Format
The user provides a **free-form search string** similar to a dispensary site search bar:

```
"Khalifa Kush 3.5g flower under $40"
"Trulieve live resin 1g vape"
"edibles under $25"
"daily special flower"
"high THC sativa 1/8"
```

Hermes parses the query into:
- **Product / Strain / Brand** (e.g., Khalifa Kush, Trulieve, Wyld)
- **Product Type** (flower, pre-roll, edible, vape, concentrate, tincture, topical, accessory)
- **Weight** (1g, 3.5g, 1/8, 1/4, 1/2, 1oz, 0.5g, etc.)
- **Price Range** (under $X, between $X and $Y)
- **Potency** (THC > X%, CBD > Y%, high CBD, etc.)
- **Deal Type** (daily special, first-time, loyalty, bulk, clearance)

## Filters
| Filter | Examples | Notes |
|--------|----------|-------|
| **brand** | Trulieve, Story, Connected, Alien Labs | Exact or partial match |
| **type** | flower, pre-roll, edible, vape, concentrate, tincture, topical | Standardized categories |
| **weight** | 1g, 3.5g, 1/8, 1/4, 1oz, .5g | Convert fractions to grams |
| **price_max** | under $40, <$25, max 50 | Upper bound only |
| **price_range** | $30-$50, 40 to 60 | Both bounds |
| **thc_min** | >20%, over 25% THCa, high THC | Minimum THC/THCa percentage |
| **cbd_min** | >5% CBD, high CBD | Minimum CBD percentage |
| **deal_type** | daily-special, first-time, loyalty, bulk, clearance | Promotional filter |

## Workflow
1. Parse the user's free-form query into structured filters.
2. For each of the 4 preferred dispensaries:
   a. Construct a site-specific search query or attempt to fetch the public menu page.
   b. If the menu is API-driven (e.g., Dutchie, Jane, Leafly embed), query the endpoint if accessible.
   c. Fallback to general web search scoped to the dispensary domain.
3. Extract matching products: name, brand, type, weight, THC/CBD %, price, deal label.
4. Present results in a table grouped by dispensary, sorted by price (lowest first) or deal priority.
5. Flag any first-time patient or loyalty-restricted deals and note expiration when visible.

## Parsing Rules
- **"under $X"** or **"<$X"** → `price_max`
- **"$X-$Y"** or **"$X to $Y"** → `price_range`
- **"3.5g"**, **"1/8"**, **"eighth"** → `weight: 3.5g`
- **"1g"**, **"1 gram"** → `weight: 1g`
- **"high THC"**, **">20%"**, **"over 25% THCa"** → `thc_min`
- **"high CBD"**, **">5% CBD"** → `cbd_min`
- **"flower"**, **"bud"** → `type: flower`
- **"vape"**, **"cart"**, **"cartridge"** → `type: vape`
- **"edible"**, **"gummy"**, **"chocolate"** → `type: edible`
- **"live resin"**, **"rosin"**, **"badder"**, **"shatter"** → `type: concentrate`
- **"daily special"**, **"deal"**, **"sale"** → `deal_type: daily-special`
- **"first time"**, **"new patient"** → `deal_type: first-time`

## Response Format
```
Results for: "Khalifa Kush 3.5g flower under $40"

Zen Leaf Chandler
-----------------
Khalifa Kush (Story) — Flower, 3.5g, 28% THCa | $38.00 | Daily Special: $32.30

The Flower Shop Ahwatukee
-------------------------
Khalifa Kush x The Menthol (Connected) — Flower, 3.5g, 26% THCa | $45.00
No deal match under $40

Trulieve Tempe
--------------
Khalifa Kush (Trulieve) — Flower, 3.5g, 24% THCa | $35.00 | Loyalty: $31.50

Story Cannabis North Chandler
-----------------------------
Khalifa Kush (Story) — Flower, 3.5g, 29% THCa | $40.00 | First-Time: $32.00

Total matches: 4 across 4 dispensaries
```

## Common Pitfalls
1. **Menu APIs are often blocked.** Many dispensaries use Dutchie, Jane, or Leafly menus that require auth or are CORS-restricted. Always have a web-search fallback.
2. **Prices change fast.** Menu data may be stale; advise the user to confirm with the dispensary before purchasing.
3. **Deal restrictions.** First-time and loyalty discounts often require membership or a new-patient check-in. Always note restrictions.
4. **Medical vs. recreational.** These 4 locations are recreational; medical-only deals won't apply.
5. **Weight notation.** Users may say "eighth" or "1/8" — normalize to 3.5g for consistent matching.
6. **Brand ambiguity.** "Story" could be the dispensary or the brand; disambiguate from context.

## Verification Checklist
- [ ] User query parsed into at least product name + one filter.
- [ ] All 4 preferred dispensaries queried or attempted.
- [ ] Results filtered and sorted by price.
- [ ] Deal restrictions (first-time, loyalty) noted explicitly.
- [ ] User reminded to verify pricing and availability in-store.

## Preferred Dispensaries (Hardcoded)
| # | Name | Location | Menu URL |
|---|------|----------|----------|
| 1 | Zen Leaf Chandler | Chandler, AZ | https://zenleafdispensaries.com/locations/chandler/menu/recreational |
| 2 | The Flower Shop Ahwatukee | Ahwatukee, AZ | https://theflowershopusa.com/ahwatukee/ |
| 3 | Trulieve Tempe | Tempe, AZ | https://www.trulieve.com/dispensaries/arizona/tempe |
| 4 | Story Cannabis North Chandler | North Chandler, AZ | https://storycannabis.com/dispensary-locations/arizona/north-chandler-dispensary/ |