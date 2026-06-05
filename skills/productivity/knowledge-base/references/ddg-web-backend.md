# DuckDuckGo Web Search Backend

DuckDuckGo (`ddgs`) is the free, unlimited search backend for Hermes. No API key, no rate limits, no credits.

## Setup

```bash
hermes config set web.backend ddgs
pip install ddgs
```

## Capabilities

| Feature | Supported |
|---------|-----------|
| `web_search` | ✅ Yes |
| `web_extract` | ❌ No (search-only) |
| `web_crawl` | ❌ No |

## Extract fallback

When `web_extract` is needed but DDG can't handle it, use `curl` directly:

```bash
# Instagram metadata
curl -sL "URL" -H "User-Agent: Googlebot/2.1"

# GitHub raw content
curl -sL "https://raw.githubusercontent.com/OWNER/REPO/BRANCH/path"

# General pages (limited, JS-heavy pages won't render)
curl -sL "URL"
```

## Other free backends

| Backend | Config key | Requirements |
|---------|-----------|--------------|
| `ddgs` | `web.backend = ddgs` | `pip install ddgs` |
| `brave_free` | `web.backend = brave_free` | Free Brave API key |
| `searxng` | `web.backend = searxng` | Self-hosted SearXNG instance |
| `firecrawl` | `web.backend = firecrawl` | API key or self-hosted URL |

## Switching backends

Backend changes take effect on next session (`/reset`). Mid-session tool calls use the backend loaded at session start.

```bash
hermes config set web.backend firecrawl    # switch back
hermes config set web.extract_backend tavily  # separate extract backend
```
