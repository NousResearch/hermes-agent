---
name: firecrawl
description: "Use when you need to scrape a specific URL into clean markdown, discover all pages via sitemap, or crawl a site with subpage discovery. Requires firecrawl-py and FIRECRAWL_API_KEY. Free tier: 500 credits/mo at firecrawl.dev."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [web, scraping, crawl, sitemap, markdown, firecrawl]
    related_skills: [tavily, exa-search]
---

# Firecrawl

Website: https://firecrawl.dev | API: https://api.firecrawl.dev

## When to Use

* Scrape a single URL into clean markdown (better than raw curl)
* Discover all pages on a site via sitemap mapping
* Crawl an entire site with configurable depth and subpage following
* Bypass paywalls or JavaScript-rendered content that curl cannot handle

**Trigger examples:**
- "Get the content of this URL"
- "Scrape this article"
- "Find all pages on this website"
- "Crawl this site"

## Setup

```bash
pip install firecrawl-py
```

Add to `~/.hermes/.env`:
```
FIRECRAWL_API_KEY=fc-xxxxxxxxxxxxxxxx
```

Dashboard: https://firecrawl.dev/dashboard

## Core Usage

### Scrape a URL (most common)

```python
import os
from firecrawl import V1FirecrawlApp

fc = V1FirecrawlApp(api_key=os.environ.get('FIRECRAWL_API_KEY'))

r = fc.scrape_url('https://example.com/article')
print(r.markdown)                          # clean markdown content
print(r.metadata.get('title'))             # page title
print(r.metadata.get('description'))      # meta description
print(r.metadata.get('statusCode'))       # 200 = success
```

### Map a site (discover all pages via sitemap)

```python
map_result = fc.map_url('https://example.com/blog')
print(map_result.links)                   # list of discovered URLs
```

### Crawl a site with subpage discovery

```python
result = fc.crawl_url('https://example.com', params={
    'crawlSubpages': True,
    'maxDepth': 2,
})
for page in result.data:
    print(page['metadata']['title'], page['metadata']['sourceURL'])
```

## Common Pitfalls

1. **Using `Firecrawl()` (v2) instead of `V1FirecrawlApp()` (v1).** The v2 class has a known URL parsing bug in the current pip package version. Always use `V1FirecrawlApp` for reliable results.

2. **Missing metadata fields.** Always check `statusCode` in metadata — a 403 or 404 means the page is inaccessible and `markdown` may be empty.

3. **Rate limits.** Free tier has 500 credits/mo. A typical scrape costs ~3-10 credits depending on page size. Monitor at https://firecrawl.dev/dashboard.

4. **JavaScript-heavy pages.** Firecrawl handles basic JS rendering but is not a full browser. For complex SPAs, use `browser_navigate` + `browser_snapshot` instead.

## Verification Checklist

- [ ] `pip install firecrawl-py` completes without error
- [ ] `FIRECRAWL_API_KEY` set in `~/.hermes/.env`
- [ ] `python -c "from firecrawl import V1FirecrawlApp; print('OK')"` runs cleanly
- [ ] `fc.scrape_url('https://example.com')` returns markdown content