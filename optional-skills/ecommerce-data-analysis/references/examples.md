# Usage Examples

## Example 1: Search by Keyword

Find e-commerce sites related to "pizza oven":

```bash
python3 scripts/query.py search "pizza oven" --size 5
```

Expected output:

```
Search: keyword='pizza oven' | 42 results | page 1/9

#    Domain                         Platform     Country  GMV(12m)     Growth
------------------------------------------------------------------------------
1    ooni.com                       shopify      GB       $56.0M       +13.3%
     Ooni Pizza Ovens
     Make great pizza at home with Ooni portable pizza ovens...

2    pizzaovens.com                 woocommerce  US       $2.1M        +8.5%
     Pizza Ovens - Wood Fired & Gas
     Shop wood fired and gas pizza ovens for home and commercial use...
```

## Example 2: Search with Filters

Find Shopify stores selling coffee in China:

```bash
python3 scripts/query.py search "coffee" --country CN --platform shopify
```

Expected output:

```
Search: keyword='coffee' | country=CN | platform=shopify | 8 results | page 1/1

#    Domain                         Platform     Country  GMV(12m)     Growth
------------------------------------------------------------------------------
1    example-coffee.cn              shopify      CN       $5.6M        +11.5%
     Example Coffee Shop
     Premium coffee beans...
```

## Example 3: Filter-Only Search (No Keyword)

Find all US Shopify stores with GMV > $1M:

```bash
python3 scripts/query.py search --country US --platform shopify --min-gmv 1000000 --sort gmvLast12month
```

## Example 4: Range Filters

Find stores with significant social media presence and high growth:

```bash
python3 scripts/query.py search "fashion" --min-instagram 10000 --min-growth 20 --sort growth
```

## Example 5: Get Domain Analytics

Look up full analytics for a specific domain:

```bash
python3 scripts/query.py domain ooni.com
```

Expected output:

```
================================================================
  Domain:    ooni.com
  Brand:     Ooni
  Platform:  shopify  |  Plan: Shopify Plus
  Country:   GB  |  City: Edinburgh  |  State: Scotland
  Status:    active  |  Created: 2018-03-15
  Employees: 250
================================================================

  Revenue (GMV)
  2023:      $45.0M
  2024:      $52.0M
  2025:      $58.0M
  2026:      N/A
  Last 12m:  $56.0M
  Growth:    +13.3%
  Monthly:   5000000

  Products
  Count: 85  |  Avg Price: $349.00

  Traffic
  Monthly Visits: 2.5M  |  Page Views: 8.1M
  Alexa Rank: #45230  |  Platform Rank: #312

  Social Media
  Instagram    520K  (30d: +3,200)
  TikTok       180K  (30d: +8,500)
  Facebook     290K  (30d: +1,100)
  YouTube      45K

  Description
  Make great pizza at home with Ooni portable outdoor pizza ovens...

  Tags: pizza oven, outdoor cooking, wood fired, gas oven
  Categories: Food & Drink > Kitchen Appliances
```

## Example 6: Export Raw JSON

Get machine-readable output for further processing:

```bash
python3 scripts/query.py domain ooni.com --json
```

Returns the full API response as formatted JSON — useful for piping into `jq` or other tools.

## Example 7: Paginated Search with Filters

```bash
# Page 1
python3 scripts/query.py search "coffee" --country US --page 1 --size 10

# Page 2
python3 scripts/query.py search "coffee" --country US --page 2 --size 10
```

## Example 8: Competitor Analysis Workflow

Step 1 — Discover competitors by keyword + country:

```bash
python3 scripts/query.py search "sustainable fashion" --country US --min-gmv 500000 --size 10
```

Step 2 — Get detailed analytics for each competitor:

```bash
python3 scripts/query.py domain everlane.com
python3 scripts/query.py domain reformation.com
python3 scripts/query.py domain patagonia.com
```

Step 3 — Export JSON for comparison:

```bash
python3 scripts/query.py domain everlane.com --json > everlane.json
python3 scripts/query.py domain reformation.com --json > reformation.json
```

## Example 9: Direct API Call (curl)

```bash
curl -X POST https://api.eccompass.ai/public/api/v1/search \
  -H "APEX_TOKEN: $APEX_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "coffee",
    "filters": {"countryCode4": "CN", "platform": "shopify"},
    "ranges": {"gmvLast12month": {"min": 100000}},
    "sortField": "gmvLast12month",
    "sortOrder": "desc",
    "page": 1,
    "size": 20
  }'
```
