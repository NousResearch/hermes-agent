---
name: indeed-job-search
description: >
  Search Indeed jobs with full filter surface (keyword, location, salary, job type,
  remote/hybrid, experience level, date posted). Returns structured job results
  including title, company, salary, location, description, and URL.
category: browser
version: 1.0.0
---

# Indeed Job Search

## Target
- URL: https://www.indeed.co.uk/jobs?q=<keyword>&l=<location>
- Method: Browser (browser_navigate + browser_snapshot)
- Bot detection: High (Cloudflare/Akamai fingerprinting + bot-detection redirects)

## Capabilities
- Full filter surface: keyword, location, radius, date posted, salary, job type, experience level, remote/hybrid
- Structured data from embedded page JSON (`window._initialData`)
- Pagination support (start parameter)
- 5 outcome branches: results, zero_results, location_unparseable, bot_block, posting_not_found

## Anti-Detection
- Bot detection level: HIGH
- CAPTCHA risk: Possible (Cloudflare challenge)
- Rate limit: ~20 requests/hour before blocks
- Strategy: 10-15s delays between requests, full browser session, avoid rapid pagination
- On block: retry once after 30s delay. If still blocked, return bot_block and stop.

## Input
- Keyword (q): URL-encoded, supports boolean operators (AND, OR, NOT), quoted phrases, field prefixes (title:, company:)
- Location (l): City/region, postcode, or blank for nationwide
- Filters (all optional):
  - radius: 0/5/10/15/25/35/50/100 (miles)
  - fromage: 1/3/7/14 (days since posted)
  - salary/pay range
  - jt: fulltime/parttime/contract/temporary/permanent/internship
  - explvl: entry_level/mid_level/senior_level
  - sc=0kf%3Aattr%28DSQF7%29%3B — remote filter
  - sc=0kf%3Aattr%28PAXZC%29%3B — hybrid filter
  - start: 0/10/20 (pagination offset)

## Workflow

### Step 1: Build URL
```
https://www.indeed.co.uk/jobs?q=<keyword>&l=<location>&fromage=7&jt=fulltime
```
Use indeed.co.uk, not indeed.com (UK market).

### Step 2: Navigate and Extract
```
browser_navigate("<built_url>")
browser_snapshot(full=true)
```

Check for bot detection (look for "verify", "robot", redirect to login):
- If bot_block detected: retry once with 30s delay
- If still blocked: return `outcome: "bot_block"`

Extract structured data from embedded JavaScript:
- Look for `window._initialData` in source
- Look for `window.mosaic.providerData["MosaicProviderRichSearchDaemon"]`
- These contain structured job listings as JSON

### Step 3: Parse Results
Extract from each job card:
- `title` — job title
- `company` — employer name
- `salary` — salary range (if listed)
- `location` — job location
- `jobType` — fulltime/parttime/etc.
- `postedDate` — relative date ("5 days ago")
- `description` — short snippet
- `url` — /viewjob?jk=xxx link
- `jk` — Indeed job key (for detail fetch)

### Step 4: Return Results
```json
{
  "outcome": "results",
  "query": {
    "keyword": "product manager",
    "location": "Nottingham",
    "filters": {
      "fromage": 7,
      "jt": "fulltime",
      "remote": true
    }
  },
  "total_results": 45,
  "results": [
    {
      "jk": "abc123def",
      "title": "Senior Product Manager",
      "company": "TechCo Ltd",
      "salary": "£60,000 - £80,000",
      "location": "Nottingham (Remote)",
      "posted_date": "2 days ago",
      "description": "We are looking for a Senior Product Manager...",
      "url": "https://www.indeed.co.uk/viewjob?jk=abc123def"
    }
  ],
  "pagination": {
    "page": 1,
    "total_pages": 5,
    "next_start": 10
  }
}
```

## Outcome Branches

| Condition | Behaviour |
|-----------|-----------|
| Results found | Return structured results with pagination |
| Zero results | `outcome: "zero_results"` — suggest broader query |
| Location unparseable | `outcome: "location_unparseable"` — suggest different format |
| Bot block | `outcome: "bot_block"` — wait, retry once, then stop |
| Job not found (single lookup) | `outcome: "posting_not_found"` |

## Integration Points

- **Job hunt (P1)**: Daily cron for saved queries, dedup with LinkedIn results
- **Kanban**: High-match roles → Kanban task for application
- **job-prep skill**: Pass job data for CV tailoring

## Pitfalls

- Indeed has aggressive bot detection; blocks are expected
- No viable API (Publisher API deprecated 2023)
- Embedded JSON structure may change without notice
- UK-specific: use indeed.co.uk, UK postcodes, GBP salary format
- Some listings are sponsored/recommended — flag in output
- Salary data is frequently missing from UK listings

## Verification

1. Test with broad keyword (e.g., "product manager")
2. Test with location filter (Nottingham)
3. Test with remote filter
4. Test with date filter (last 3 days)
5. Expect occasional bot blocks — document frequency
