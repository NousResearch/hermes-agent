---
name: ecommerce-website-data
version: 1.2.18
description: >
  Free, instant access to ecommerce data on 10M+ ecommerce stores and
  ecommerce websites. Search Shopify stores, WooCommerce sites, and more.
  Capabilities: (1) Search ecommerce stores by keyword & filters
  (2) Full store analytics with 100+ data fields (3) Historical GMV &
  traffic trends (4) Installed apps & tech stack (5) LinkedIn contacts.
  Use when the user wants ecommerce website data, Shopify store analytics,
  competitor analysis, revenue trends, or decision-maker contacts.
  Triggers: "search ecommerce stores selling pet food",
  "find Shopify stores in US", "ecommerce website data for ooni.com",
  "what apps does X use", "revenue trend for X", "competitor analysis".
author: eccompass.ai
website: https://eccompass.ai
license: Proprietary
metadata:
  hermes:
    tags: [ecommerce, analytics, competitor-analysis, lead-generation, shopify]
    config:
      - key: APEX_TOKEN
        description: "ECcompass API access token"
        prompt: "Enter your APEX_TOKEN"
        help: "Get your free token at https://eccompass.ai → Dashboard → API Access → Create Token"
        required_for: "API access to fetch ecommerce data"
---

## **Ecommerce Website Data**

The All-in-One Skill for ecommerce intelligence. 

Powered by [EcCompass AI](https://eccompass.ai) — one of the world's largest DTC databases — this skill delivers *free, live data* on 10M+ stores with 100+ analytics fields. 

## **What You Can Do**

Search Stores — "Find Shopify stores selling pet food with 10k+ Instagram followers" 

Domain Analytics — "Show me ooni.com's GMV trend and tech stack" 

Lead Contacts — "Get decision-maker emails for this brand" 


## **Data Coverage**

| Metric | Value |
| :--- | :--- |
| Total domains | 10,000,000+ |
| Countries | 200+ |
| Platforms | Shopify, WooCommerce, Wix, Squarespace, BigCommerce and more |
| GMV data | Monthly GMV from 2023 to Date |
| Social media | Instagram, TikTok, Twitter/X, YouTube, Facebook, Pinterest |
| Lead Contacts | Verified LinkedIn profiles and business emails |
| Update frequency &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | Monthly |

## **Usage Notes**

Data freshness — Store data is live and updated monthly, not a static snapshot.

GMV accuracy — Revenue figures are estimates based on traffic and industry benchmarks, not actual financial records.

Contact verification — All emails and LinkedIn profiles have been verified.

## **Scope**

Can do — Store discovery, single-store analytics, competitor lookup, tech stack detection, contact extraction

Cannot do — Access store backend, guarantee exact GMV, provide real-time inventory or pricing, execute outreach campaigns.

## **Setup**

**100% Free. One-minute setup.**

### Install via Hermes Agent

You can install this skill directly from the Hermes Skills Hub or GitHub.

**Option 1: Install from Skills Hub**
```bash
/skills install ecommerce-website-data
```
*(or run `hermes skills install ecommerce-website-data` in your CLI)*

**Option 2: Install directly from GitHub**
If the skill is hosted on a GitHub repository, you can install it directly by pointing to the repository:
```bash
/skills install github:roger52027/ecommerce-website-data
```
*(or run `hermes skills install github:roger52027/ecommerce-website-data` in your CLI)*

When you first use the skill, Hermes will automatically prompt you to enter your `APEX_TOKEN`. 
Get your free token at [eccompass.ai](https://eccompass.ai) → Dashboard → API Access → Create Token.

## **Quick Start**

**IMPORTANT**: Always use the Python script for API calls. It has the correct base URL and authentication built in.

```bash
# Search by keyword
python3 {baseDir}/scripts/query.py search "pet food"

# Search with filters
python3 {baseDir}/scripts/query.py search "coffee" --country CN --platform shopify

# Filter only (no keyword)
python3 {baseDir}/scripts/query.py search --country US --platform shopify --min-gmv 1000000

# Search with range filters and sorting
python3 {baseDir}/scripts/query.py search "fashion" --min-instagram 10000 --sort gmvLast12month

# Find stores that have a TikTok page
python3 {baseDir}/scripts/query.py search --platform shopline --exists tiktokUrl

# Find stores with both email and TikTok presence
python3 {baseDir}/scripts/query.py search --country US --exists tiktokUrl,emails

# Get full analytics for a domain
python3 {baseDir}/scripts/query.py domain ooni.com

# Export as JSON
python3 {baseDir}/scripts/query.py domain ooni.com --json

# Get historical GMV and traffic data
python3 {baseDir}/scripts/query.py historical ooni.com

# Get installed apps/plugins
python3 {baseDir}/scripts/query.py apps ooni.com

# Get LinkedIn contacts
python3 {baseDir}/scripts/query.py contacts ooni.com
```

## **API Base URL**

```
https://api.eccompass.ai
```

**CRITICAL**: All API paths start with `/public/api/v1/`. The `/public` prefix is mandatory — without it, you will get an authentication error. Never omit `/public` from the path.

## **API Endpoints**

### **1. Search —** `POST https://api.eccompass.ai/public/api/v1/search`

Full URL: `https://api.eccompass.ai/public/api/v1/search` (POST, Content-Type: application/json)

Authentication: `APEX_TOKEN` header (NOT `Authorization` header).

```bash
curl -X POST https://api.eccompass.ai/public/api/v1/search \
  -H "APEX_TOKEN: $APEX_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "coffee",
    "filters": { "countryCode4": "CN", "platform": "shopify" },
    "ranges": { "gmvLast12month": { "min": 100000 } },
    "sortField": "gmvLast12month",
    "sortOrder": "desc",
    "page": 1,
    "size": 20
  }'
```

All body fields are optional.

Exists filter: Use `"exists": ["tiktokUrl", "emails"]` to require that specific fields are present and not empty. This is the correct way to express "has a TikTok page" or "has contact email". Common exists fields: `tiktokUrl`, `instagramUrl`, `facebookUrl`, `youtubeUrl`, `twitterUrl`, `linkedinUrl`, `emails`, `phones`.

Keyword search covers: domain, title, description, tags, categories, **platform name**, merchant name, technologies, and **installed apps**. So "klaviyo" in keyword will match stores using the Klaviyo app. You do NOT need to know exact values — just put the search term in the keyword field.

Filters are case-insensitive and support both single values and arrays (OR logic). Any field listed below can be used as a filter, range, or exists condition.

```json
// Single value
{"filters": {"platform": "shopify"}}

// Array = OR logic: matches Europe OR Africa
{"filters": {"region": ["Europe", "Africa"]}}

// Combine keyword + multi-value filter
{"keyword": "energy drink", "filters": {"region": ["Europe", "Africa"]}}

// App/plugin filter
{"filters": {"countryCode4": "US", "installedApps": "klaviyo"}}
```

### **Complete Field Reference**

Geography (Keyword, for filters):
`countryCode4`, `countryCode`, `countryCode2`, `countryCode3`, `city`, `state`, `region`, `subregion`, `zip`, `streetAddress`, `latitude`, `longitude`, `companyLocation`(Text)

Platform & Store (Keyword, for filters):
`platform`, `plan`, `status`, `platformDomain`, `lastPlatform`, `lastPlan`, `lastPlatformChanged`, `lastPlanChanged`, `created`, `languageCode`, `currency`, `salesChannels`

Content (Text, for filters — partial match):
`description`, `title`, `merchantName`, `categories`, `tags`, `tagsV5`, `metaDescription`, `metaKeywords`, `features`

Products (Keyword/Text for filters):
`averageProductPrice`(Keyword), `avgPriceFormatted`(Keyword), `categoriesV1`(Keyword), `tagsFirst`(Keyword)

Tech Stack (Text, for filters — partial match):
`technologies`, `installedApps`, `theme`, `themeVendor`, `themeSpend`, `themeVersion`(Keyword)

Contact (Text, best with exists):
`emails`, `phones`, `contactPageUrl`

Social URLs (for exists — check presence):
`tiktokUrl`, `instagramUrl`, `facebookUrl`, `youtubeUrl`, `twitterUrl`, `linkedinUrl`, `pinterestUrl`, `linkedinAccount`(Keyword)

Revenue (Long, for ranges — all in USD):
`gmvLast12month`, `gmv2023`, `gmv2024`, `gmv2025`, `gmv2026`, `estimatedMonthlySales`, `estimatedSalesYearly`

Traffic (Long/Integer, for ranges):
`estimatedVisits`, `estimatedPageViews`, `alexaRank`, `rank`, `platformRank`, `rankPercentile`(Float), `platformRankPercentile`(Float)

Products numeric (Integer/Long, for ranges):
`productCount`, `avgPriceUsd`, `maxPrice`, `minPrice`, `variantCount`, `productImages`, `productsSold`, `vendorCount`

Employees (Integer, for ranges):
`employeeCount`, `monthlyAppSpend`

Social Followers (Integer, for ranges):
`instagramFollowers`, `instagramFollowers30d`, `instagramFollowers90d`, `tiktokFollowers`, `tiktokFollowers30d`, `tiktokFollowers90d`, `twitterFollowers`, `twitterFollowers30d`, `twitterFollowers90d`, `youtubeFollowers`, `youtubeFollowers30d`, `youtubeFollowers90d`, `facebookFollowers`, `facebookFollowers30d`, `facebookFollowers90d`, `facebookLikes`, `pinterestFollowers`(Long), `pinterestFollowers30d`(Long), `pinterestFollowers90d`(Long)

Growth (Double, for ranges):
`growth`

### **Tip for the agent**:
- Platform names (Shopify, Shopline, WooCommerce, etc.) → prefer `keyword` over `filters.platform` (fuzzy match)
- Plugins/apps (Klaviyo, Yotpo, etc.) → use `keyword` or `filters.installedApps` / `filters.technologies`
- Multiple regions/countries (e.g. "Europe or Africa") → array in filters: `{"region": ["Europe", "Africa"]}`
- "Has TikTok" / "Has Email Address" → use `exists`: `{"exists": ["tiktokUrl"]}` or `{"exists": ["emails"]}`
- Monthly GMV → use `estimatedMonthlySales` range directly, or `gmvLast12month` (annual)

### **2. Domain Analytics —** `GET https://api.eccompass.ai/public/api/v1/domain/{domain}`

Full URL: `https://api.eccompass.ai/public/api/v1/domain/{domain}` (GET)

```bash
curl -H "APEX_TOKEN: $APEX_TOKEN" https://api.eccompass.ai/public/api/v1/domain/ooni.com
```

Returns 100+ fields of complete analytics for a single domain.

### **3. Historical Data —** `GET https://api.eccompass.ai/public/api/v1/historical/{domain}`

```bash
curl -H "APEX_TOKEN: $APEX_TOKEN" https://api.eccompass.ai/public/api/v1/historical/ooni.com
```

Returns monthly GMV, UV, PV, and average price data from 2023 onwards. Use when the user asks for revenue trends, traffic history, or month-over-month performance.

### **4. Installed Apps —** `GET https://api.eccompass.ai/public/api/v1/installed-apps/{domain}`

```bash
curl -H "APEX_TOKEN: $APEX_TOKEN" https://api.eccompass.ai/public/api/v1/installed-apps/ooni.com
```

Returns the list of apps/plugins installed on a store with details: name, rating, install count, vendor info, pricing plans. Use when the user asks "what apps does X use" or "which plugins are installed".

### **5. LinkedIn Contacts —** `GET https://api.eccompass.ai/public/api/v1/contacts/{domain}`

```bash
curl -H "APEX_TOKEN: $APEX_TOKEN" https://api.eccompass.ai/public/api/v1/contacts/ooni.com
```

Returns verified LinkedIn contacts for a domain's company: name, position, email, LinkedIn profile URL. Use for lead generation, decision-maker lookup, or outreach.

## **Requirements**

- Python 3.6+
- Network access to `api.eccompass.ai`
- `APEX_TOKEN` environment variable (get yours at [eccompass.ai](https://eccompass.ai))

## **Documentation**

- [AI Instructions](SKILL.md) — How the agent uses this skill
- [API Schema](references/schema.md) — Full response format and field definitions
- [Usage Examples](references/examples.md) — Real-world scenarios with sample output

## **License**

Proprietary — [EcCompass AI](https://eccompass.ai)

## **Support**

For questions, issues, or feature requests, visit [https://eccompass.ai](https://eccompass.ai).
