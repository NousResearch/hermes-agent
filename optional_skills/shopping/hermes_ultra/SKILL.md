---
name: hermes-ultra
description: Multi-store price tracker and shopping intelligence skill. Tracks prices across Amazon, eBay, Best Buy, Newegg, Walmart, and comparison sites. Provides deal scoring, scalper detection, trend prediction, and smart buy/wait/avoid recommendations.
version: 2.0.0
author: Dusk1E
license: MIT
platforms: [macos, linux, windows]
metadata:
  hermes:
    tags: [Shopping, Price-Tracker, E-Commerce, Deal-Finder, Scalper-Detection]
    related_skills: []
---

# Hermes Ultra — Shopping Intelligence

Multi-store price tracking and deal analysis skill for Hermes Agent.
Searches and monitors product prices across major retailers, scores deals,
detects scalper pricing, predicts price trends, and provides
BUY / WAIT / AVOID recommendations with detailed reasoning.

## When to Use

- User asks to check, track, or compare prices for a product
- User wants to find the cheapest price across multiple stores
- User asks "Is this a good deal?" or "Should I buy this now?"
- User wants to monitor a product for price drops
- User suspects scalper/overpriced listings
- User asks about price trends or price history
- Keywords: "price check", "track price", "compare prices", "deal score",
  "scalper", "price history", "buy now or wait"

## Supported Stores

| Store | Coverage | Parser Type |
|-------|----------|-------------|
| Amazon (US, UK, DE, FR, IT, ES, TR) | Product pages + search | Site-specific |
| eBay (US, UK, DE, FR) | Listings + search | Site-specific |
| Best Buy | Product pages + search | Site-specific |
| Newegg | Product pages + search | Site-specific |
| Walmart | Product pages + search | Site-specific |
| Idealo (DE, UK, FR, IT, ES) | Price comparison | Site-specific |
| PriceSpy / PriceRunner | Price comparison | Site-specific |
| CamelCamelCamel | Amazon price history | Site-specific |
| Any e-commerce site | JSON-LD / OpenGraph | Generic fallback |

## Prerequisites

### Required Python packages

```bash
pip install httpx pyyaml rich
```

### Optional (for JS-heavy sites)

```bash
pip install playwright
playwright install chromium
```

Without Playwright, the skill uses httpx (no JavaScript rendering).
Some Amazon/Walmart pages may require Playwright for accurate prices.

## Quick Reference

| Action | How to invoke |
|--------|--------------|
| Search & compare prices | "Find the best price for RTX 4090" |
| Check specific URL | "Check the price of https://amazon.com/dp/B0..." |
| Track a product | "Track this product and alert me if it drops below $500" |
| View tracked products | "Show my tracked products" |
| Price history | "Show price history for product #3" |
| Deal score | Automatically included in price checks |
| Scalper detection | Automatically included when deviation > 15% |
| Trend prediction | Automatically included when 3+ data points exist |

## Procedure

### Price Check (URL provided)

1. Use the `web_extract` or `terminal` tool to fetch the product page HTML
2. Run the appropriate parser from `scripts/parsers/` based on the URL domain
3. If the parser returns no price, trigger LLM fallback (`scripts/llm_fallback.py`)
4. Calculate deal score using `scripts/scoring.py`
5. Run scalper detection using `scripts/scalper_detector.py`
6. Generate price reasoning using `scripts/reasoning.py`
7. Format the full intelligence report using `scripts/alerts.py`

### Product Search (name/query provided)

1. Run `scripts/searcher.py` to search across all 5 major stores concurrently
2. For each store result, fetch and parse the product page
3. Build a market comparison table (cheapest → most expensive)
4. Score the best deal and detect cross-site scalping
5. Present the full report with BUY/WAIT/AVOID recommendation

### Price Tracking

1. Store the product in the SQLite database (`scripts/database.py`)
2. Record the initial price as the first history entry
3. On subsequent checks, compare with previous prices
4. Trigger alerts when: price drops below target, deal score hits 80+

## Architecture

```
selectors.yaml          ← All CSS/regex patterns (edit this when sites change)
scripts/
├── parsers/
│   ├── selector_loader.py   ← YAML → regex engine (hot-reload)
│   ├── amazon_global.py     ← Amazon parser (reads from YAML)
│   ├── ebay_global.py       ← eBay parser
│   ├── bestbuy.py           ← Best Buy parser
│   ├── newegg.py            ← Newegg parser
│   ├── walmart.py           ← Walmart parser
│   ├── generic.py           ← JSON-LD/OG fallback parser
│   └── price_comparison.py  ← Idealo, PriceSpy, Camel
├── llm_fallback.py     ← LLM extraction when selectors break
├── reasoning.py        ← MSRP vs price analysis + BUY/WAIT/AVOID
├── scoring.py          ← 0-100 deal score engine
├── scalper_detector.py ← Price manipulation detection
├── trend_predictor.py  ← Linear regression trend analysis
├── searcher.py         ← Multi-store concurrent search
├── scraper.py          ← Stealth web scraper (Playwright/httpx)
├── database.py         ← SQLite product + price history store
└── alerts.py           ← Rich terminal formatting + notifications
```

## Selector Maintenance

When a retailer changes their HTML structure, update `selectors.yaml`:

```yaml
# Example: Amazon changed their price class
amazon:
  price:
    - 'class="new-price-class"[^>]*>([\\d.,]+)'   # Add new pattern
    - 'class="a-price-whole"[^>]*>([\\d.,]+)'      # Keep old as fallback
```

The selector loader hot-reloads the YAML file — no restart needed.

If even updated selectors fail, the LLM fallback will attempt to extract
product data from the raw page text automatically.

## Pitfalls

1. **Anti-bot protection**: Amazon, Walmart, and eBay have aggressive
   bot detection. Playwright mode helps but isn't foolproof. If blocked,
   results from other stores are still shown.

2. **Rate limiting**: The scraper adds random delays (1.5–5s) between
   requests. Rapid successive queries may trigger blocks.

3. **JavaScript-rendered prices**: Some sites (Walmart, certain Amazon
   pages) render prices via JavaScript. Without Playwright, these may
   show as "price unavailable."

4. **Dynamic selectors**: If all store selectors break simultaneously,
   check for site-wide HTML restructuring and update `selectors.yaml`.

5. **Database location**: Data is stored at
   `~/.hermes/skills/shopping/hermes_ultra.db`. Use `HERMES_HOME` env
   var to override the base path.

## Data Storage

- **Product database**: `~/.hermes/skills/shopping/hermes_ultra.db` (SQLite)
- **Selectors**: `selectors.yaml` in the skill directory
- **No cloud dependencies**: All processing is local
