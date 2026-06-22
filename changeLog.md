# Hermes ChangeLog

## 2026-06-22 — Local SearXNG / Firecrawl routing guard

- Added regression coverage for the production web backend split:
  - `web.search_backend = searxng` routes `web_search` to local SearXNG when `SEARXNG_URL` is configured.
  - `web.extract_backend = firecrawl` keeps `web_extract` on Firecrawl because SearXNG is search-only.
  - Missing `SEARXNG_URL` falls back to the shared `web.backend` path instead of silently selecting an unavailable SearXNG backend.
- Updated the desktop app entry in `package-lock.json` from `0.15.1` to `0.17.0` so it matches `apps/desktop/package.json`.

Validation:

```text
python -m pytest tests/tools/test_web_providers_searxng.py -q -o 'addopts='
29 passed in 1.12s

git diff --check
# no whitespace errors
```
