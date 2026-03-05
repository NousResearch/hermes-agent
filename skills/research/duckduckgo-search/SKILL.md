---
name: duckduckgo-search
description: Get web search results from DuckDuckGo. Use as fallback when Firecrawl unavailable. No API key needed.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Search, DuckDuckGo, Web Search, API, Free]
    related_skills: [arxiv, ocr-and-documents]
---

# DuckDuckGo Search

Fast, free web search. No API key required. Use when Firecrawl is unavailable.

## Quick Reference

| Action | Command |
|--------|---------|
| Web search | `ddgs text "python async" -k 5` |
| Images | `ddgs images "cat"` |
| News | `ddgs news "AI"` |
| Videos | `ddgs videos "tutorial"` |
| **Curl fallback** | `curl "https://api.duckduckgo.com/?q=QUERY&format=json"` |

## Prerequisites

### Option 1: Python Library (Recommended)

```bash
pip install ddgs
ddgs --help
```

### Option 2: Curl Only (No Dependencies)

```bash
# Verify curl is available
curl --version
```

No installation needed — curl is standard on all platforms.

## Web Search

### Library (ddgs)

```bash
# Basic search
ddgs text "python async programming" -k 5

# With region filter
ddgs text "best restaurants Tokyo" -k 3 -r jp-jp

# Safe search
ddgs text "medical advice" -k 5 -s off
```

### Parameters

| Flag | Description | Example |
|------|-------------|---------|
| `-k` | Max results | `-k 5` |
| `-r` | Region | `-r us-en` |
| `-s` | Safe search | `-s off` |

### Curl Fallback

```bash
# Basic search
curl -s "https://api.duckduckgo.com/?q=python+async&format=json&limit=5"

# Parse results
curl -s "..." | jq -r '.RelatedTopics[] | "\(.Text) - \(.FirstURL)"'
```

## Other Search Types

```bash
# Images
ddgs images "landscape" -k 10

# News
ddgs news "artificial intelligence" -k 5

# Videos  
ddgs videos "python tutorial" -k 5
```

## Integration

After finding URLs, retrieve full content with `web_extract`:

```bash
# Find with DDG, then extract
ddgs text "fastapi tutorial" -k 3
# Copy URL from output
web_extract(urls=["https://fastapi.tiangolo.com/tutorial/"])
```

This is the standard pattern:
1. **DuckDuckGo** → finds URLs
2. **web_extract** → retrieves full content

## Use Cases

| Scenario | Tool | Reason |
|----------|------|--------|
| "Find tutorials on X" | `ddgs text` + `web_extract` | Need full content |
| Firecrawl unavailable | `ddgs text` | Free fallback |
| Quick image search | `ddgs images` | Find images |
| Latest news | `ddgs news` | News results |

## Error Handling

```bash
# Check if library installed
ddgs --help 2>/dev/null || echo "Using curl fallback"

# Rate limiting - add delay
sleep 1

# No results - try different query
ddgs text "different keywords" -k 5
```

## Notes

- **No API key required** — completely free
- **Rate limit**: ~1 request/second recommended
- **Always follow up** with `web_extract` for full content
- Curl fallback has limited results (DDG API restrictions)
