---
name: find-deals
description: Use when the user wants to search for cannabis product deals at their preferred Arizona dispensaries. Accepts free-form product queries (e.g., "Khalifa Kush 3.5g flower under $40") and optional filters (brand, type, weight, THC/CBD, price range, deal type). Searches Zen Leaf Chandler, The Flower Shop Ahwatukee, Trulieve Tempe, and Story Cannabis North Chandler.
version: 1.2.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [shopping, deals, local-search, pricing, retail, dispensaries, cannabis]
    related_skills: [web-search, maps]
---

# Find Deals

## Overview
This skill searches for current cannabis product deals, menu pricing, and promotions at the user's four preferred Arizona recreational dispensaries. Due to the dynamic nature of cannabis menu platforms (Dutchie and Jane), the skill works best when a web search backend is configured (FIRECRAWL_API_KEY, TAVILY_API_KEY, or EXA_API_KEY).

## Preferred Dispensaries
| # | Name | Location | Platform | Menu URL |
|---|------|----------|----------|----------|
| 1 | Zen Leaf Chandler | Chandler, AZ | Jane/IHeartJane | https://zenleafdispensaries.com/locations/chandler/menu/recreational |
| 2 | The Flower Shop Ahwatukee | Ahwatukee, AZ | Jane/IHeartJane | https://theflowershopusa.com/ahwatukee/ |
| 3 | Trulieve Tempe | Tempe, AZ | Dutchie | https://www.trulieve.com/dispensaries/arizona/tempe |
| 4 | Story Cannabis North Chandler | North Chandler, AZ | Dutchie | https://storycannabis.com/dispensary-locations/arizona/north-chandler-dispensary/ |

## When to Use
- User asks "find deals for X" where X is a product, strain, brand, or category
- Comparing prices across the user's preferred dispensaries
- Searching for specific vape brands (Boutiq, Abstrakt, Clean Concentrates, Dime, etc.)
- Filtering menus by weight, potency (THC/CBD), price range, or product type

## Query Format
The user provides a **free-form search string** similar to a dispensary site search bar:

```
"Khalifa Kush 3.5g flower under $40"
"Trulieve live resin 1g vape"
"edibles under $25"
"daily special flower"
"high THC sativa 1/8"
"find deals for boutiq, abstrakt, clean concentrates, dime"
"show me all vape carts under $35"
```

## Parsed Filters
| Filter | Examples |
|--------|----------|
| **brand** | Trulieve, Story, Connected, Alien Labs, Boutiq, Abstrakt, Clean Concentrates, Dime |
| **type** | flower, pre-roll, edible, vape, concentrate, tincture, topical |
| **weight** | 1g, 3.5g, 1/8, 1/4, 1oz, .5g |
| **price** | under $40, $30-$50, max 50 |
| **potency** | >20% THC, high CBD |
| **deal_type** | daily-special, first-time, loyalty, bulk |

## Execution Strategy
1. **Parse query** into structured filters from free-form text.
2. **Check web tools availability.** If `FIRECRAWL_API_KEY`, `TAVILY_API_KEY`, or `EXA_API_KEY` is configured, use `web_search_tool` and `web_extract_tool` to fetch current menus.
3. **For each dispensary**, construct a site-specific search query:
   - `"Boutiq vape site:zenleafdispensaries.com/chandler"`
   - `"Abstrakt site:theflowershopusa.com"`
   - `"Dime vape site:trulieve.com/tempe"`
4. **Extract prices and availability** from search results or page extracts.
5. **Present results** grouped by dispensary in a price-sorted table.

## Known Limitations
| Issue | Cause | Workaround |
|-------|-------|------------|
| Dynamic menus (JS-rendered) | Dutchie/Jane platforms | Requires web_crawl or browser automation |
| API auth blocks | Dutchie/Jane GraphQL endpoints are CORS/auth-protected | Use search/extract tools instead of direct API |
| Real-time stock | Inventory updates frequently | Always verify in-store before visiting |

## Web Tool Configuration
The skill depends on a configured web backend. Add to `~/.hermes/.env`:
```bash
export FIRECRAWL_API_KEY="your_key"       # Or TAVILY_API_KEY, EXA_API_KEY
export FIRECRAWL_API_URL=""                # Optional: self-hosted Firecrawl
```

Or select via `hermes tools` interactive setup.

## Example User Queries
- `"find deals for boutiq, abstrakt, clean concentrates, dime"`
- `"Khalifa Kush 3.5g flower under $40"`
- `"show me all vape carts under $35"`
- `"find high THC sativa 1/8 at my spots"`

## Response Format
```
Results for: "Boutiq, Abstrakt, Clean Concentrates, Dime"

Zen Leaf Chandler
-----------------
Boutiq Live Resin Cart 1g — Vape, 87% THC | $42.00 | Daily Special: $35.70
(No Abstrakt, Clean Concentrates, or Dime found)

The Flower Shop Ahwatukee
-------------------------
Abstrakt Vape Cart .5g — Vape, 82% THC | $28.00
Boutiq Live Rosin 1g — Concentrate, 78% THC | $55.00 | First-Time: $44.00

Trulieve Tempe
--------------
Dime .5g Disposables — Vape, 85% THC | $30.00 | Loyalty: $27.00
Clean Concentrates Live Resin 1g — Concentrate, 80% THC | $45.00

Story Cannabis North Chandler
-----------------------------
Boutiq Liquid Diamonds 1g — Vape, 90% THC | $48.00
Clean Concentrates Premium Badder 1g — Concentrate, 75% THC | $50.00 | Daily Special: $42.50

Total matches: 7 across 4 dispensaries
```

## Common Pitfalls
1. **No web tools configured.** If none of FIRECRAWL_API_KEY, TAVILY_API_KEY, or EXA_API_KEY is set, the skill falls back to direct page fetching which often fails on JS-rendered menus.
2. **Brand spelling matters.** "Boutiq" is spelled with a 'Q' (not 'K'), "Abstrakt" with a 'K' (not 'C').
3. **Dispensary-specific brands.** Some brands are exclusive to certain chains (e.g., certain house brands). Cross-location availability varies.
4. **Deal restrictions.** First-time and loyalty discounts require in-store verification.

## Verification Checklist
- [ ] Web search backend configured
- [ ] Query parsed into at least one filter
- [ ] All 4 preferred dispensaries queried
- [ ] Results filtered and sorted by price
- [ ] Deal restrictions noted
- [ ] User reminded to verify in-store