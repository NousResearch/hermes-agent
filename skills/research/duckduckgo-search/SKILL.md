---
name: duckduckgo-search
description: Free web search via DuckDuckGo when Firecrawl is unavailable. No API key needed. Use ddgs CLI or Python library to find URLs, then web_extract for content.
version: 1.1.0
author: gamedevCloudy
license: MIT
metadata:
  hermes:
    tags: [search, duckduckgo, web-search, free, fallback]
    related_skills: [arxiv]
---

# DuckDuckGo Search (Firecrawl Fallback)

Free web search using DuckDuckGo. **No API key required.**

## When to Use This

Use this skill ONLY when the `web_search` tool is not available (i.e., `FIRECRAWL_API_KEY` is not set). If `web_search` works, prefer it — it returns richer results with built-in content extraction.

Signs you need this fallback:
- `web_search` tool is not listed in your available tools
- `web_search` returns an error about missing FIRECRAWL_API_KEY

## Setup

```bash
# Install the ddgs package (one-time)
pip install ddgs
```

## Web Search (Primary Use Case)

### Via Terminal (ddgs CLI)

```bash
# Basic search — returns titles, URLs, and snippets
ddgs text -k "python async programming" -m 5

# With region filter
ddgs text -k "best restaurants" -m 5 -r us-en

# Recent results only (d=day, w=week, m=month, y=year)
ddgs text -k "latest AI news" -m 5 -t w

# JSON output for parsing
ddgs text -k "fastapi tutorial" -m 5 -o json
```

### Via Python (in execute_code)

```python
from hermes_tools import terminal

# Search and get results
result = terminal("ddgs text -k 'python web framework comparison' -m 5")
print(result["output"])
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

## Other Search Types

```bash
# Image search
ddgs images -k "landscape photography" -m 10

# News search
ddgs news -k "artificial intelligence" -m 5

# Video search
ddgs videos -k "python tutorial" -m 5
```

## Workflow: Search → Extract

DuckDuckGo finds URLs. To get full page content, follow up with `web_extract`:

1. **Search** with ddgs to find relevant URLs
2. **Extract** content using the `web_extract` tool (if available) or curl

```bash
# Step 1: Find URLs
ddgs text -k "fastapi tutorial" -m 3

# Step 2: Extract full content from a result URL
# (use web_extract tool if available, otherwise curl)
curl -s "https://example.com/article" | head -200
```

## Limitations

- **Rate limiting**: DuckDuckGo may throttle after many rapid requests. Add `sleep 1` between searches if needed.
- **No content extraction**: ddgs only returns titles, URLs, and snippets — not full page content. Use `web_extract` or curl for that.
- **Results quality**: Generally good but less configurable than Firecrawl's search.
- **Availability**: DuckDuckGo may block requests from some cloud IPs. If searches return empty, try different keywords or add a short delay.

## Pitfalls

- **Don't confuse `-k` and `-m`**: `-k` is for keywords (the query), `-m` is for max results count.
- **Package name**: The package is `ddgs` (was previously `duckduckgo-search`). Install with `pip install ddgs`.
- **Empty results**: If ddgs returns nothing, it may be rate-limited. Wait a few seconds and retry.
