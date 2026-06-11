# Self-Hosted Web Backends

Hermes can route web search and URL extraction through local services instead
of the Nous Tool Gateway or hosted vendor APIs. This page covers the supported
configuration for SearXNG and Firecrawl.

## Backend Roles

SearXNG is search-only. Use it for `web_search`.

Firecrawl supports both search and extraction. It is the usual self-hosted
choice for `web_extract`.

If you run both, configure them per capability:

```yaml
web:
  search_backend: searxng
  extract_backend: firecrawl
```

The shared fallback form also works when one backend should handle both tools:

```yaml
web:
  backend: firecrawl
```

## SearXNG

Hermes reads the SearXNG instance URL from `SEARXNG_URL`.

Add this to `~/.hermes/.env`:

```bash
SEARXNG_URL=http://localhost:8080
```

Then select SearXNG for search:

```bash
hermes config set web.search_backend searxng
```

Use `SEARXNG_URL`, not `SEARXNG_BASE_URL` or
`SEARXNG_INSTANCE_URL`. The bundled SearXNG provider and setup checks both
look for `SEARXNG_URL`.

## Firecrawl

Hermes reads self-hosted Firecrawl from `FIRECRAWL_API_URL`.

Add this to `~/.hermes/.env`:

```bash
FIRECRAWL_API_URL=http://localhost:3002
```

Then select Firecrawl for extraction:

```bash
hermes config set web.extract_backend firecrawl
```

If your Firecrawl server enforces an API key, set the same key in Hermes:

```bash
FIRECRAWL_API_KEY=local-firecrawl-key
```

For Firecrawl cloud, set `FIRECRAWL_API_KEY` without
`FIRECRAWL_API_URL`.

## Plugin Availability

The bundled backend plugins are `web-searxng` and `web-firecrawl`. They are
discovered automatically as backend plugins. If you have customized plugin
loading, make sure those two plugins are not disabled.

## Troubleshooting

If Hermes falls back to another provider, check these first:

1. `~/.hermes/.env` contains `SEARXNG_URL` for SearXNG and
   `FIRECRAWL_API_URL` for self-hosted Firecrawl.
2. `web.search_backend` and `web.extract_backend` are set when you want
   different providers for search and extraction.
3. The selected backend plugin is enabled and visible in `hermes tools`.
4. Restart the gateway after changing `.env` or backend config so the running
   process sees the new values.
