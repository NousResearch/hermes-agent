# Web Search & Extract Providers

Free and pay-as-you-go alternatives to Firecrawl for Hermes web search and extraction.

## Search backends

### DuckDuckGo (ddgs) — FREE, unlimited, no API key

```bash
pip install ddgs
hermes config set web.backend ddgs
```

Restart the gateway or start a new session after changing. Supports `web_search` only — NOT `web_extract`.

### Brave Free (brave_free) — FREE, needs free API key

```bash
hermes config set web.backend brave_free
```

Get a free API key at https://brave.com/search/api/ (2,000 queries/month free).

### SearXNG (searxng) — self-hosted, FREE

```bash
hermes config set web.backend searxng
```

Requires a running SearXNG instance URL in config.

## Extract backends

DuckDuckGo and Brave Free are **search-only** — they don't support `web_extract`.
For extraction, the options are:

### curl + Python (free, unlimited)
Works for Instagram metadata, articles, simple pages. Use Googlebot user-agent to get
full metadata from Instagram and similar JS-heavy platforms:

```bash
curl -sL "https://www.instagram.com/reel/ID/" \
  -H "User-Agent: Googlebot/2.1" | \
  python3 -c "import sys,re,html; ..."
```

### Tavily — 1,000 calls/month free, then pay-as-you-go

```bash
hermes config set web.extract_backend tavily
```

No subscription — pay only for what you use above the free quota.
API key from https://tavily.com.

### Firecrawl — paid, subscription-based
The old default. Move away if you don't want a monthly subscription.

## Per-capability overrides

You can set different backends for search and extract:

```yaml
web:
  backend: ddgs              # fallback for both
  search_backend: ddgs       # overrides web.backend for search
  extract_backend: tavily    # overrides web.backend for extract
```

```bash
hermes config set web.search_backend ddgs
hermes config set web.extract_backend tavily
```

`/reset` or gateway restart required after changing.
