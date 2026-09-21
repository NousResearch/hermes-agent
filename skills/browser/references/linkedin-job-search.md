---
name: linkedin-job-search
description: >
  Search LinkedIn jobs using LinkedIn's public unauthenticated guest API.
  Returns structured job data with title, company, location, posting date, and
  canonical URL. No cookies, no auth, no browser session required.
category: browser
version: 1.0.0
---

# LinkedIn Job Search

## Target
- URL: https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
- Method: Fetch (web_extract / curl to public guest API)
- Bot detection: Low (guest API is explicitly public)

## Capabilities
- Search LinkedIn jobs by keyword, location, recency, and sort order
- Structured results: jobId, title, company, location, posted date
- Pagination support (10 results per page)
- No side effects: never applies, saves, or messages
- No authentication required

## Anti-Detection
- Bot detection level: Low (guest API is designed for public access)
- CAPTCHA risk: None
- Rate limit: ~1 req/s sustained (conservative estimate)

## API Details

### Base Endpoint
```
GET https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
```

### Parameters
| Parameter | Required | Description |
|-----------|----------|-------------|
| `keywords` | Yes | URL-encoded role/skills string |
| `location` | Yes | Free-text location OR numeric geoId |
| `f_TPR` | No | Recency filter (see below) |
| `sortBy` | No | `DD` (most recent), `R` (relevance) |
| `start` | No | Pagination offset (0, 10, 20...) |

### Recency Filters (f_TPR)
| Value | Window |
|-------|--------|
| `r3600` | Last 1 hour |
| `r86400` | Last 24 hours |
| `r604800` | Last 7 days |
| `r2592000` | Last 30 days |
| (omitted) | All time |

### Location (geoId)
Common UK geoIds:
- `90402519` — United Kingdom
- `102890719` — London Area
- `104455041` — Midlands
- `90402520` — Nottingham Area
- `90000022` — Remote

Look up unknown geoIds via:
```
GET https://www.linkedin.com/jobs-guest/api/typeaheadHits?query=<text>&typeaheadType=GEO
```

## Workflow

### Step 1: Build Search URL
```
https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search
  ?keywords=product%20manager
  &location=Nottingham%20Area
  &f_TPR=r604800
  &sortBy=DD
  &start=0
```

### Step 2: Fetch Results
```bash
curl -s "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=product%20manager&location=Nottingham%20Area&f_TPR=r86400&sortBy=DD"
```

Returns HTML `<li>` fragments, NOT JSON. Parse using the extraction rules below.

### Step 3: Parse Job Cards
Each card is an `<li>` block. Extract:

| Field | How |
|-------|-----|
| `jobId` | `data-entity-urn="urn:li:jobPosting:(\d+)"` |
| `title` | Text content, collapse whitespace, trim |
| `company` | Text content, collapse whitespace, trim |
| `location` | Text content, collapse whitespace, trim |
| `posted_iso` | `<time datetime="YYYY-MM-DD">` |
| `posted_relative` | `<time>` text content (e.g., "6 hours ago") |
| `actively_hiring` | Boolean — true if "Actively Hiring" text present |
| `url` | Concatenate: `https://www.linkedin.com/jobs/view/{slug}-{jobId}` |

### Step 4: Paginate
- Page size: 10 results per response (hard-coded)
- Increment `start` by 10 per page
- Stop when: `< 10 cards returned` OR max pages reached
- No `totalResultCount` in response

### Step 5: Return
```json
{
  "success": true,
  "query": {
    "keywords": "product manager",
    "location": "Nottingham Area",
    "recency": "last_7_days",
    "sort": "most_recent"
  },
  "page": {
    "start": 0,
    "size": 10
  },
  "jobs": [
    {
      "job_id": "4304338796",
      "title": "Senior Product Manager",
      "company": "TechCo Ltd",
      "location": "Nottingham, England",
      "posted_iso": "2026-05-28",
      "posted_relative": "2 days ago",
      "actively_hiring": true,
      "url": "https://www.linkedin.com/jobs/view/senior-product-manager-at-techco-4304338796"
    }
  ]
}
```

### Outcome Shapes

| Condition | Response |
|-----------|----------|
| Results found | `success: true, jobs: [...]` |
| No results | `success: true, jobs: []` |
| Auth wall hit | `success: false, error: "auth_wall"` |
| Rate limited | `success: false, error: "rate_limited"` |

## Integration Points

- **Job hunt (P1)**: Daily cron for saved queries → Telegram/Discord digest
- **Kanban**: High-match roles → task for application review
- **job-prep skill**: Job data → CV tailoring per spec
- **Dedup with Indeed results**: Cross-reference job IDs

## Pitfalls

- Response is HTML, not JSON — must parse DOM via browser_snapshot or regex
- Page size is EXACTLY 10 — `&count=` params are silently ignored
- No total result count — must paginate until < 10 cards
- `datetime` is date-only (no time) — use relative text for sub-day precision
- Title/company/location have heavy whitespace — always `.replace(/\s+/g,' ').trim()`
- The `/jobs/collections/recommended/` path requires auth — DON'T use it
- geoId lookup endpoint may return different IDs for UK vs global
- Title slugs may contain percent-encoded UTF-8 — use jobId as dedup key
- Guest API is undocumented — may change without notice

## Verification

1. Search for a known role ("product manager", "Nottingham")
2. Test recency filter (last 24h, 7d, 30d)
3. Test pagination (page 2)
4. Test with no results (should return empty array)
5. Test geoId lookup for UK locations
6. Verify canonical URL format
