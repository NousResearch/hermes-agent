---
name: glassdoor-company-research
description: >
  Research company ratings, salary data, and interview difficulty from Glassdoor
  for job hunt preparation. Note: Glassdoor has aggressive anti-bot protections.
  Only aggregate data accessible without Browserbase + pre-authed session.
category: browser
version: 1.0.0
---

# Glassdoor Company Research

## Target
- URL: https://www.glassdoor.co.uk/Reviews/<company>-Reviews-E<id>.htm
- Method: Browser (browser_navigate + browser_snapshot)
- Bot detection: VERY HIGH (Cloudflare + give-to-get wall + fingerprinting)

## Capabilities
- Company aggregate data: overall rating, sub-ratings, rating distribution
- Salary range data (public pages)
- CEO approval percentage
- Company metadata (HQ, industry, size, founded)
- **Cannot** extract individual reviews (login-gated + give-to-get wall)

## Anti-Detection
- Bot detection level: VERY HIGH
- CAPTCHA risk: Nearly guaranteed on review pages
- Rate limit: ~10 requests before blocks
- Strategy: Browser session with delays, accept that review extraction is impossible

## Authentication Status (from browse.sh research)

| Path | Status | Reason |
|------|--------|--------|
| Partner API | 410 Gone | Retired 2021 |
| Internal GraphQL | 403 Forbidden | Requires anti-CSRF + session cookies |
| Autocomplete | 403 | Cloudflare challenge |
| Direct fetch | 403 | Cloudflare interstitial |
| **Browserbase + proxies + auth** | Works | Requires paid Browserbase + pre-authed session |
| **Our VPS** | Limited | Aggregate data only, no reviews |

## Limitations (Current Setup)

Due to no paid Browserbase subscription and no residential proxies:
- **Cannot** extract individual employee reviews
- **Cannot** bypass give-to-get wall
- **Cannot** maintain session persistence
- **Can** access some public aggregate data (company overview page)
- **Can** access salary data via public pages

## Workflow

### Step 1: Search for Company
```
https://www.glassdoor.co.uk/Reviews/<slug>-Reviews-E<id>.htm
```
Requires knowing the company's Glassdoor EmployerId upfront.
Alternative: search via Google/Bing for the company's Glassdoor page.

### Step 2: Extract Aggregate Data
From company overview page (public), extract:
- `overall_rating` — star rating (1-5)
- `sub_ratings` — culture, diversity, work-life, compensation, career
- `recommend_pct` — % would recommend to friend
- `ceo_approval_pct` — % approve of CEO
- `review_count` — total number of reviews
- `salary_count` — total salary reports
- `industry` — company industry
- `size` — employees (1-50, 51-200, 201-1000, 1001-5000, 5000+)
- `founded` — founding year
- `hq` — headquarters location

### Step 3: Return
```json
{
  "success": true,
  "company": "Company Name",
  "data_source": "aggregate_only",
  "limitation": "Individual reviews require paid Browserbase + pre-authed session",
  "overall_rating": 4.2,
  "sub_ratings": {
    "work_life_balance": 3.8,
    "culture_values": 4.0,
    "compensation": 4.1,
    "career_opportunities": 3.9,
    "diversity": 3.7
  },
  "recommend_pct": 78,
  "ceo_approval_pct": 85,
  "review_count": 1423,
  "salary_count": 567,
  "metadata": {
    "industry": "Software & Technology",
    "size": "5000+",
    "founded": 2012,
    "hq": "San Francisco, CA"
  },
  "url": "https://www.glassdoor.co.uk/Reviews/..."
}
```

## Alternative Approaches

Since Glassdoor individual reviews are inaccessible:

1. **Google reviews** — Search "Company Name reviews Google" → extract via web_search
2. **Indeed company pages** — Indeed has company reviews with lower bot detection
3. **LinkedIn company pages** — Public company data from LinkedIn
4. **Blind (teamblind)** — Anonymous tech company reviews (US-centric)
5. **Fishbowl** — Professional community discussions about companies

## Integration Points

- **Job hunt (P1)**: Aggregate company data for targeting decisions
- **job-prep skill**: Basic company research for interview prep
- **Kanban**: Company data → informed application decisions

## Pitfalls

- Glassdoor is the most aggressively protected site in this skill set
- Without Browserbase/residential proxies, only aggregate data is available
- Company overview page may also require login depending on recent changes
- UK and US Glassdoor have different content/moderating
- Individual salaries and review details are behind the give-to-get wall
- Cookie sessions expire after ~30 days

## Verification

1. Test with large well-known company (plausible aggregate data)
2. Test aggregate rating against Google search results
3. Document extraction status (verified working / partially working / blocked)
4. Try Indeed as fallback for company reviews
