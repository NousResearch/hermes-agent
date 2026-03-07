---
name: duckduckgo-search
description: Free web search via DuckDuckGo — text, news, images, videos. No API key needed. Use the Python DDGS library or CLI to search, then web_extract for full content.
version: 1.2.0
author: gamedevCloudy
license: MIT
metadata:
  hermes:
    tags: [search, duckduckgo, web-search, free, fallback]
    related_skills: [arxiv]
---

# DuckDuckGo Search

Free web search using DuckDuckGo. **No API key required.**

Preferred when `web_search` tool is unavailable or unsuitable (no `FIRECRAWL_API_KEY` set). Can also be used as a standalone search tool.

## Setup

```bash
# Install the ddgs package (one-time)
pip install ddgs
```

## Python API (Primary)

Use the `DDGS` class in `execute_code` for structured results with typed fields.

**Important:** `max_results` must always be passed as a **keyword argument** — positional usage raises an error on all methods.

### Text Search

Best for: general research, companies, documentation.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.text("python async programming", max_results=5):
        print(r["title"])
        print(r["href"])
        print(r.get("body", "")[:200])
        print()
```

Returns: `title`, `href`, `body`

### News Search

Best for: current events, breaking news, latest updates.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.news("AI regulation 2026", max_results=5):
        print(r["date"], "-", r["title"])
        print(r.get("source", ""), "|", r["url"])
        print(r.get("body", "")[:200])
        print()
```

Returns: `date`, `title`, `body`, `url`, `image`, `source`

### Image Search

Best for: visual references, product images, diagrams.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.images("semiconductor chip", max_results=5):
        print(r["title"])
        print(r["image"])       # direct image URL
        print(r.get("thumbnail", ""))
        print(r.get("source", ""))
        print()
```

Returns: `title`, `image`, `thumbnail`, `url`, `height`, `width`, `source`

### Video Search

Best for: tutorials, demos, explainers.

```python
from ddgs import DDGS

with DDGS() as ddgs:
    for r in ddgs.videos("FastAPI tutorial", max_results=5):
        print(r["title"])
        print(r.get("content", ""))       # video URL
        print(r.get("duration", ""))       # e.g. "26:03"
        print(r.get("provider", ""))       # YouTube, etc.
        print(r.get("published", ""))
        print()
```

Returns: `title`, `content`, `description`, `duration`, `provider`, `published`, `statistics`, `uploader`

### Quick Reference

| Method | Use When | Key Fields |
|--------|----------|------------|
| `text()` | General research, companies | title, href, body |
| `news()` | Current events, updates | date, title, source, body, url |
| `images()` | Visuals, diagrams | title, image, thumbnail, url |
| `videos()` | Tutorials, demos | title, content, duration, provider |

## CLI (Alternative)

Use the `ddgs` command via terminal when you don't need structured field access.

```bash
# Text search
ddgs text -k "python async programming" -m 5

# News search
ddgs news -k "artificial intelligence" -m 5

# Image search
ddgs images -k "landscape photography" -m 10

# Video search
ddgs videos -k "python tutorial" -m 5

# With region filter
ddgs text -k "best restaurants" -m 5 -r us-en

# Recent results only (d=day, w=week, m=month, y=year)
ddgs text -k "latest AI news" -m 5 -t w

# JSON output for parsing
ddgs text -k "fastapi tutorial" -m 5 -o json
```

### CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `-k` | Keywords (query) — **required** | `-k "search terms"` |
| `-m` | Max results | `-m 5` |
| `-r` | Region | `-r us-en` |
| `-t` | Time limit | `-t w` (week) |
| `-s` | Safe search | `-s off` |
| `-o` | Output format | `-o json` |

## Workflow: Search then Extract

DuckDuckGo returns titles, URLs, and snippets — not full page content. To get full content, follow up with `web_extract`:

1. **Search** with ddgs to find relevant URLs
2. **Extract** content using the `web_extract` tool (if available) or curl

```python
from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("fastapi deployment guide", max_results=3))
    for r in results:
        print(r["title"], "->", r["href"])

# Then use web_extract tool on the best URL
```

## Limitations

- **Rate limiting**: DuckDuckGo may throttle after many rapid requests. Add a short delay between searches if needed.
- **No content extraction**: ddgs returns snippets, not full page content. Use `web_extract` or curl for that.
- **Results quality**: Generally good but less configurable than Firecrawl's search.
- **Availability**: DuckDuckGo may block requests from some cloud IPs. If searches return empty, try different keywords or wait a few seconds.
- **Field variability**: Return fields may vary between results or ddgs versions. Use `.get()` for optional fields to avoid KeyError.

## Pitfalls

- **`max_results` is keyword-only**: `ddgs.text("query", 5)` raises an error. Use `ddgs.text("query", max_results=5)`.
- **Don't confuse `-k` and `-m`** (CLI): `-k` is for keywords, `-m` is for max results count.
- **Package name**: The package is `ddgs` (was previously `duckduckgo-search`). Install with `pip install ddgs`.
- **Empty results**: If ddgs returns nothing, it may be rate-limited. Wait a few seconds and retry.

## Validated With

Smoke-tested with `ddgs==9.11.2` on Python 3.13. All four methods (text, news, images, videos) confirmed working with keyword `max_results`.
