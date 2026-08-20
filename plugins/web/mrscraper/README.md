# MrScraper for Hermes

This bundled integration registers MrScraper as a `web_search` / `web_extract`
provider and exposes 15 independently callable `mrscraper_*` tools. Fourteen
tools are registered by this web plugin; the rendered-page tool is registered
by `plugins/browser/mrscraper`.

Configure the API token through `hermes tools`, or add this secret to the
active profile's `~/.hermes/.env`:

```dotenv
MRSCRAPER_API_TOKEN=your-token
```

To route the standard Hermes web tools through MrScraper:

```yaml
web:
  search_backend: mrscraper
  extract_backend: mrscraper
```

Representative native calls include `mrscraper_search_google_serp`,
`mrscraper_fetch_rendered_html`, `mrscraper_extract_page_by_prompt`,
`mrscraper_get_results`, and `mrscraper_run_existing_scraper_batch`.

The rendered-page API does not expose a persistent CDP/WebSocket session, so
MrScraper is not registered as a `browser.cloud_provider`. Interactive
`browser_*` actions still require a CDP-compatible provider.
