# ECcompass API Response Schema

## API Base URL

```
https://api.eccompass.ai
```

**CRITICAL: All paths include `/public/api/v1/` prefix. The `/public` part is mandatory — omitting it will cause authentication errors.**

Authentication: Pass your API token via the `APEX_TOKEN` HTTP header (NOT the `Authorization` header).

## Common Response Envelope

All endpoints return:

```json
{
  "success": true,
  "message": "...",
  "data": { ... }
}
```

Error response:

```json
{
  "success": false,
  "code": 400,
  "message": "Domain not found"
}
```

---

## 1. Search — `POST https://api.eccompass.ai/public/api/v1/search`

### Request

```
POST https://api.eccompass.ai/public/api/v1/search
APEX_TOKEN: <your_token>
Content-Type: application/json
```

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `keyword` | string | No | Text search across domain, title, description, tags, categories, **platform**, merchant name, technologies, **installed apps** (case-insensitive, partial match) |
| `filters` | object | No | Match filters. Key = field name (camelCase), Value = string or array of strings (array = OR logic) |
| `ranges` | object | No | Numeric range filters. Key = field name, Value = `{ "min": N, "max": N }` |
| `exists` | array | No | Fields that must exist (not null/empty). Use camelCase names. E.g. `["tiktokUrl", "emails"]` |
| `sortField` | string | No | Field name to sort by (camelCase) |
| `sortOrder` | string | No | `asc` or `desc` (default: desc) |
| `page` | int | No | Page number, starts from 1 (default: 1) |
| `size` | int | No | Results per page, max 100 (default: 20) |

### All Available Fields

All fields below can be used in `filters`, `ranges` (numeric only), or `exists`. Filters are **case-insensitive**. Use an array for OR logic: `"region": ["Europe", "Africa"]`.

#### Geography (Keyword — exact match filters)

| Field | Description |
|-------|-------------|
| `countryCode4` | Primary country code (US, CN, GB, etc.) — **most commonly used** |
| `countryCode` | Country code (alternative) |
| `countryCode2` | 2-letter country code |
| `countryCode3` | 3-letter country code |
| `region` | Continent (Europe, Africa, Asia, Americas, Oceania) |
| `subregion` | Subregion (Northern Europe, Southeast Asia, etc.) |
| `city` | City name |
| `state` | State / province |
| `zip` | ZIP / postal code |
| `streetAddress` | Street address |
| `latitude` | Latitude |
| `longitude` | Longitude |

#### Platform & Store (Keyword — exact match filters)

| Field | Description |
|-------|-------------|
| `platform` | E-commerce platform (shopify, woocommerce, magento, shopline, etc.) |
| `plan` | Platform plan (e.g. Shopify Plus) |
| `status` | Site status (active, etc.) |
| `platformDomain` | Platform domain |
| `lastPlatform` | Last known platform |
| `lastPlan` | Last known plan |
| `lastPlatformChanged` | Date of last platform change |
| `lastPlanChanged` | Date of last plan change |
| `created` | Store creation date |
| `languageCode` | Primary language code |
| `currency` | Currency code |
| `salesChannels` | Sales channels |

#### Content (Text — partial match filters)

| Field | Description |
|-------|-------------|
| `title` | Website title |
| `description` | Website description |
| `merchantName` | Store / brand name |
| `categories` | Product categories |
| `categoriesV1` | Standardized categories (Keyword) |
| `tags` | Tags |
| `tagsV5` | AI-generated tags (best for search) |
| `tagsFirst` | Primary tag (Keyword) |
| `metaDescription` | HTML meta description |
| `metaKeywords` | HTML meta keywords |
| `features` | Store features |

#### Tech Stack (Text — partial match filters)

| Field | Description |
|-------|-------------|
| `technologies` | Detected technologies (JSON) |
| `installedApps` | Installed apps / plugins (comma-separated) — use for app queries like "klaviyo" |
| `theme` | Theme name |
| `themeVendor` | Theme vendor |
| `themeSpend` | Theme price |
| `themeVersion` | Theme version (Keyword) |

#### Contact (Text — best used with `exists`)

| Field | Description |
|-------|-------------|
| `emails` | Business email addresses |
| `phones` | Phone numbers |
| `contactPageUrl` | Contact page URL |

#### Social URLs (Text/Keyword — best used with `exists`)

| Field | Description |
|-------|-------------|
| `tiktokUrl` | TikTok profile URL |
| `instagramUrl` | Instagram profile URL (Keyword) |
| `facebookUrl` | Facebook page URL (Keyword) |
| `youtubeUrl` | YouTube channel URL |
| `twitterUrl` | Twitter/X profile URL |
| `linkedinUrl` | LinkedIn page URL (Keyword) |
| `pinterestUrl` | Pinterest profile URL |
| `linkedinAccount` | LinkedIn account (Keyword) |

#### Revenue (Long — for `ranges`, all in USD)

| Field | Description |
|-------|-------------|
| `gmvLast12month` | Last 12 months GMV — **most commonly used** |
| `gmv2023` | 2023 annual GMV |
| `gmv2024` | 2024 annual GMV |
| `gmv2025` | 2025 annual GMV |
| `gmv2026` | 2026 annual GMV |
| `estimatedMonthlySales` | Estimated monthly GMV |
| `estimatedSalesYearly` | Estimated yearly GMV |

#### Traffic & Ranking (Integer/Long — for `ranges`)

| Field | Description |
|-------|-------------|
| `estimatedVisits` | Estimated monthly visits (Long) |
| `estimatedPageViews` | Estimated monthly page views (Long) |
| `alexaRank` | Alexa global rank |
| `rank` | Overall rank |
| `platformRank` | Rank within platform |
| `rankPercentile` | Rank percentile (Float) |
| `platformRankPercentile` | Platform rank percentile (Float) |

#### Products (Integer/Long — for `ranges`)

| Field | Description |
|-------|-------------|
| `productCount` | Number of products (Long) |
| `avgPriceUsd` | Average price in USD |
| `maxPrice` | Highest product price |
| `minPrice` | Lowest product price |
| `variantCount` | Number of variants |
| `productImages` | Total product images (Long) |
| `productsSold` | Products sold |
| `vendorCount` | Number of vendors (Long) |
| `monthlyAppSpend` | Monthly app spend |

#### Employees (Integer — for `ranges`)

| Field | Description |
|-------|-------------|
| `employeeCount` | Number of employees |

#### Social Followers (Integer — for `ranges`)

| Platform | Followers | 30d Change | 90d Change |
|----------|-----------|------------|------------|
| Instagram | `instagramFollowers` | `instagramFollowers30d` | `instagramFollowers90d` |
| TikTok | `tiktokFollowers` | `tiktokFollowers30d` | `tiktokFollowers90d` |
| Twitter/X | `twitterFollowers` | `twitterFollowers30d` | `twitterFollowers90d` |
| YouTube | `youtubeFollowers` | `youtubeFollowers30d` | `youtubeFollowers90d` |
| Facebook | `facebookFollowers` | `facebookFollowers30d` | `facebookFollowers90d` |
| Pinterest | `pinterestFollowers`(Long) | `pinterestFollowers30d`(Long) | `pinterestFollowers90d`(Long) |

Additional: `facebookLikes`, `instagramPosts`, `twitterPosts`, `pinterestPosts`

#### Growth (Double — for `ranges`)

| Field | Description |
|-------|-------------|
| `growth` | Year-over-year growth rate (percentage, 11.5 = 11.5%) |

### Example Requests

Single-value filter:

```json
{
  "keyword": "coffee",
  "filters": {
    "countryCode4": "CN",
    "platform": "shopify"
  },
  "ranges": { "gmvLast12month": { "min": 100000 } },
  "sortField": "gmvLast12month",
  "sortOrder": "desc",
  "page": 1,
  "size": 20
}
```

Array filter (OR logic):

```json
{
  "keyword": "energy drink",
  "filters": {
    "region": ["Europe", "Africa"]
  }
}
```

Exists filter (field must be present):

```json
{
  "filters": {"platform": "shopline"},
  "exists": ["tiktokUrl"]
}
```

### `data` Structure

| Field | Type | Description |
|-------|------|-------------|
| `records` | array | List of matching domains |
| `total` | long | Total number of matches |
| `page` | int | Current page number |
| `size` | int | Results per page |
| `totalPages` | long | Total pages |

### Fields in Each Record

| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | Domain name |
| `merchantName` | string | Store / brand name |
| `title` | string | Website title |
| `description` | string | Website description |
| `categories` | string | Product categories |
| `countryCode4` | string | Country code (US / CN / GB, etc.) |
| `platform` | string | E-commerce platform (shopify / woocommerce, etc.) |
| `estimatedMonthlySales` | long | Estimated monthly GMV (USD) |
| `gmv2023` | long | 2023 GMV (USD) |
| `gmv2024` | long | 2024 GMV |
| `gmv2025` | long | 2025 GMV |
| `gmv2026` | long | 2026 GMV |
| `gmvLast12month` | long | Last 12 months GMV |
| `growth` | double | YoY growth rate (percentage, 11.5 = 11.5%) |

### Example Response

```json
{
  "success": true,
  "message": "Search completed successfully",
  "data": {
    "records": [
      {
        "domain": "example-coffee.cn",
        "merchantName": "Example Coffee",
        "title": "Example Coffee Shop",
        "description": "Premium coffee beans...",
        "categories": "Food & Drink",
        "countryCode4": "CN",
        "platform": "shopify",
        "estimatedMonthlySales": 500000,
        "gmv2023": 4500000,
        "gmv2024": 5200000,
        "gmv2025": 5800000,
        "gmv2026": null,
        "gmvLast12month": 5600000,
        "growth": 11.5
      }
    ],
    "total": 8,
    "page": 1,
    "size": 20,
    "totalPages": 1
  }
}
```

---

## 2. Domain Analytics — `GET https://api.eccompass.ai/public/api/v1/domain/{domain}`

### Request

```
GET https://api.eccompass.ai/public/api/v1/domain/ooni.com
APEX_TOKEN: <your_token>
```

### `data` Fields (PublicDomainDto, 100+ fields)

#### Basic Info

| Field | Type | Description |
|-------|------|-------------|
| `id` | long | Record ID |
| `domain` | string | Domain name |
| `merchantName` | string | Brand / store name |
| `title` | string | Website title |
| `description` | string | Website description |
| `platform` | string | E-commerce platform |
| `plan` | string | Platform plan (e.g. Shopify Plus) |
| `status` | string | Site status |
| `created` | string | Store creation date |
| `languageCode` | string | Primary language |

#### Geography

| Field | Type | Description |
|-------|------|-------------|
| `countryCode4` | string | Primary country code (US / CN / GB) |
| `countryCode` | string | Country code |
| `city` | string | City |
| `state` | string | State / province |
| `region` | string | Region |
| `subregion` | string | Subregion |
| `streetAddress` | string | Street address |
| `zip` | string | ZIP / postal code |
| `latitude` | string | Latitude |
| `longitude` | string | Longitude |
| `companyLocation` | string | Company location |

#### Revenue (GMV, USD)

| Field | Type | Description |
|-------|------|-------------|
| `gmv2023` | long | 2023 GMV |
| `gmv2024` | long | 2024 GMV |
| `gmv2025` | long | 2025 GMV |
| `gmv2026` | long | 2026 GMV |
| `gmvLast12month` | long | Last 12 months GMV (most commonly used) |
| `growth` | double | YoY growth rate (percentage, 11.5 = 11.5%) |
| `estimatedMonthlySales` | long | Estimated monthly GMV (USD) |
| `estimatedSalesYearly` | long | Estimated yearly GMV (USD) |

#### Products

| Field | Type | Description |
|-------|------|-------------|
| `productCount` | long | Number of products |
| `avgPriceUsd` | int | Average price (USD, integer) |
| `avgPriceFormatted` | string | Formatted average price (e.g. $58.30) |
| `averageProductPrice` | string | Average product price |
| `currency` | string | Currency |
| `maxPrice` | int | Highest price |
| `minPrice` | int | Lowest price |
| `variantCount` | int | Number of variants |
| `productImages` | long | Total product images |
| `productsSold` | int | Products sold |

#### Traffic & Ranking

| Field | Type | Description |
|-------|------|-------------|
| `estimatedVisits` | long | Estimated monthly visits |
| `estimatedPageViews` | long | Estimated monthly page views |
| `alexaRank` | int | Alexa global rank |
| `platformRank` | int | Rank within platform |
| `rank` | int | Overall rank |
| `employeeCount` | int | Number of employees |

#### Social Media

Each platform includes: followers / 30-day change / 90-day change / profile URL

| Platform | Followers | 30d Change | 90d Change |
|----------|-----------|------------|------------|
| Instagram | `instagramFollowers` | `instagramFollowers30d` | `instagramFollowers90d` |
| TikTok | `tiktokFollowers` | `tiktokFollowers30d` | `tiktokFollowers90d` |
| Twitter/X | `twitterFollowers` | `twitterFollowers30d` | `twitterFollowers90d` |
| YouTube | `youtubeFollowers` | `youtubeFollowers30d` | `youtubeFollowers90d` |
| Facebook | `facebookFollowers` | `facebookFollowers30d` | `facebookFollowers90d` |
| Pinterest | `pinterestFollowers` | `pinterestFollowers30d` | `pinterestFollowers90d` |

URL fields: `instagramUrl`, `tiktokUrl`, `twitterUrl`, `youtubeUrl`, `facebookUrl`, `pinterestUrl`, `linkedinUrl`

#### Tags & Categories

| Field | Type | Description |
|-------|------|-------------|
| `tagsV5` | string | AI-generated product tags (best for search) |
| `tagsFirst` | string | Primary category tag |
| `tags` | string | Tags |
| `categories` | string | Categories |
| `categoriesV1` | string | Standardized categories |

#### Technology Stack

| Field | Type | Description |
|-------|------|-------------|
| `technologies` | string | Detected technologies (JSON) |
| `installedApps` | string | Installed apps (comma-separated) |
| `theme` | string | Theme name |
| `themeVendor` | string | Theme vendor |
| `themeVersion` | string | Theme version |
| `monthlyAppSpend` | int | Monthly app spend |
| `features` | string | Features |

#### Contact

| Field | Type | Description |
|-------|------|-------------|
| `emails` | string | Email addresses |
| `phones` | string | Phone numbers |
| `contactPageUrl` | string | Contact page URL |

#### Reviews

| Field | Type | Description |
|-------|------|-------------|
| `trustpilot` | string | Trustpilot rating / info |
| `yotpo` | string | Yotpo rating / review count |

---

## 3. Historical Data — `GET https://api.eccompass.ai/public/api/v1/historical/{domain}`

### Request

```
GET https://api.eccompass.ai/public/api/v1/historical/ooni.com
APEX_TOKEN: <your_token>
```

### `data` Structure

| Field | Type | Description |
|-------|------|-------------|
| `records` | array | Monthly records |
| `total` | int | Total number of months |
| `months` | array | List of available months (YYYY-MM) |

### Fields in Each Monthly Record

| Field | Type | Description |
|-------|------|-------------|
| `dataMonth` | string | Month (YYYY-MM) |
| `domain` | string | Domain name |
| `category` | string | Product category |
| `monthlyGmv` | long | Monthly GMV (USD) |
| `yearlyGmv` | long | Yearly GMV (USD, available from 2025+) |
| `pv` | long | Page views |
| `uv` | long | Unique visitors |
| `avgPriceUsd` | int | Average product price (USD) |

---

## 4. Installed Apps — `GET https://api.eccompass.ai/public/api/v1/installed-apps/{domain}`

### Request

```
GET https://api.eccompass.ai/public/api/v1/installed-apps/ooni.com
APEX_TOKEN: <your_token>
```

### Response

`data` is an array of installed app objects. `total` is the count.

### Fields in Each App

| Field | Type | Description |
|-------|------|-------------|
| `token` | string | App unique identifier |
| `name` | string | App name |
| `description` | string | App description |
| `categories` | string | App categories (JSON array string) |
| `platform` | string | Platform (shopify, etc.) |
| `state` | string | Install state (Active, etc.) |
| `installedAt` | string | Installation date (ISO 8601) |
| `averageRating` | string | Average rating |
| `reviewCount` | int | Number of reviews |
| `installs` | int | Total installs across all stores |
| `installs30d` | int | Install count change (30 days) |
| `installs90d` | int | Install count change (90 days) |
| `appStoreUrl` | string | App store listing URL |
| `vendorName` | string | Vendor / developer name |
| `vendorEmail` | string | Vendor contact email |
| `vendorUrl` | string | Vendor profile URL |
| `plans` | string | Pricing plans (JSON array string) |
| `freeTrialDuration` | string | Free trial duration |

---

## 5. LinkedIn Contacts — `GET https://api.eccompass.ai/public/api/v1/contacts/{domain}`

### Request

```
GET https://api.eccompass.ai/public/api/v1/contacts/ooni.com
APEX_TOKEN: <your_token>
```

### `data` Structure

| Field | Type | Description |
|-------|------|-------------|
| `domain` | string | Domain name |
| `contacts` | array | List of LinkedIn contacts |
| `total` | int | Number of contacts |

### Fields in Each Contact

| Field | Type | Description |
|-------|------|-------------|
| `firstName` | string | First name |
| `lastName` | string | Last name |
| `position` | string | Job title / position |
| `email` | string | Verified email address |
| `companyName` | string | Company name |
| `status` | string | Email verification status |
| `sourcePage` | string | LinkedIn profile URL |

---

## Common Platform Values

`shopify` / `woocommerce` / `magento` / `bigcommerce` / `wix` / `squarespace` / `shopline` / `shoplazza` / `custom`

## Common Country Codes (countryCode4)

| Code | Country |
|------|---------|
| `US` | United States |
| `CN` | China |
| `GB` | United Kingdom |
| `CA` | Canada |
| `DE` | Germany |
| `AU` | Australia |
| `FR` | France |
| `JP` | Japan |
