---
name: mrscraper
description: Search, crawl, and extract website data with MrScraper.
version: 1.0.0
author: Riandra Diva (riandradiva18), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [MrScraper, Google SERP, Web Scraping, Crawling, Structured Extraction]
    category: research
    related_skills: [scrapling, grounded-citations]
    homepage: https://docs.mrscraper.com/docs
---

# MrScraper Skill

Use this skill to connect the cataloged standalone MrScraper plugin or [official hosted MCP server](https://docs.mrscraper.com/docs/getting-started/mcp-server) to Hermes and choose the right registered tools. The [vendor's CLI skills](https://github.com/mrscraper-com/cli/tree/main/skills) cover `mrscraper` commands; this skill covers native `web_search`, `web_extract`, `mrscraper_*`, and MCP tools. Neither connection provides an interactive browser session for clicking or typing.

## When to Use

- Find relevant pages in Google results before extracting their contents.
- Extract named fields or repeated listings from a known public URL.
- Discover a site's URLs, then inspect the pages that matter.
- Render a JavaScript-heavy page when ordinary `web_extract` lacks content.

## Prerequisites

- A MrScraper account. Calls to the service can incur charges; URLs, prompts, and schemas sent through either connection leave Hermes.
- **Native plugin:** In `terminal`, run `hermes plugins install mrscraper`, then `hermes plugins enable mrscraper`. The catalog install uses a reviewed commit pin and prompts for `MRSCRAPER_API_TOKEN`. Start a new Hermes session after enabling. To route standard `web_search` and `web_extract` through it, select MrScraper under Web Search & Extract in `hermes tools`; the focused `mrscraper_*` tools do not require that provider selection.
- **Hosted MCP alternative:** In `terminal`, run `hermes mcp add mrscraper --url https://mcp.mrscraper.com/mcp --auth oauth`. Complete the browser sign-in, choose the tools to enable, and start a new session. If already configured, use `hermes mcp test mrscraper`. The server currently documents `fetch`, `scrape`, `serp`, `status`, `rerun`, `results`, and `result`; inspect the live MCP schemas before calling them. The MCP connection does not configure the native web provider.

Install this optional skill with `hermes skills install official/research/mrscraper`. Installing the skill does not install either MrScraper connection.

## How to Run

Choose the connection already available in this session. Use `web_search` and `web_extract` when the native plugin is selected as the web provider; use a focused `mrscraper_*` tool when the request needs MrScraper-specific controls. With hosted MCP, Hermes registers server tools as `mcp__mrscraper__<tool>`; discover their live argument schemas before use. If neither connection is available, guide setup under Prerequisites and continue after the connection appears in a new session.

## Quick Reference

| Task | Native plugin | Hosted MCP |
| --- | --- | --- |
| Google results | `web_search` or `mrscraper_search_google_serp` | `mcp__mrscraper__serp` |
| Render one page | `web_extract` or `mrscraper_fetch_rendered_html` | `mcp__mrscraper__fetch` |
| Extract fields or listings | `mrscraper_extract_page_by_prompt`, `mrscraper_extract_listings`, or `mrscraper_extract_structured_data` | `mcp__mrscraper__scrape` |
| Discover site URLs | `mrscraper_crawl_website_urls` | `mcp__mrscraper__scrape` with the map agent, if exposed by its live schema |
| Read saved run output | `mrscraper_get_results` or `mrscraper_get_result_detail` | `mcp__mrscraper__results` or `mcp__mrscraper__result` |

MCP names in the right column assume the server was configured as `mrscraper`. Use the names and schemas registered in the current session if the server name differs.

## Procedure

1. Identify the user's target URL or search query, desired fields, geography, and scope. Keep crawl depth, page count, and result count as small as the task allows. Use only public HTTP or HTTPS targets.
2. For discovery, call `web_search` with MrScraper selected, `mrscraper_search_google_serp` with a query, or `mcp__mrscraper__serp`. Set region and language only from the user's context. Treat search snippets as leads, then fetch the relevant URLs before claiming page facts.
3. For a known page, first inspect the source through `web_extract`, `mrscraper_fetch_rendered_html` (HTML or Markdown with rendering controls), or `mcp__mrscraper__fetch`. Derive simple fields from that content. When the user asks for managed extraction or the page warrants it, use `mrscraper_extract_page_by_prompt` with explicit fields and an `output_schema` if supplied; use `mrscraper_extract_listings` for repeated items. Use `mrscraper_extract_structured_data` only when its available category matches the page. On MCP, inspect `mcp__mrscraper__scrape` for equivalent options rather than copying native arguments.
4. For a multi-page task, use `mrscraper_crawl_website_urls` or the map agent in `mcp__mrscraper__scrape` to discover URLs first. Inspect the response for its scraper or run identifier; use `mrscraper_get_results` / `mrscraper_get_result_detail` or `mcp__mrscraper__results` / `mcp__mrscraper__result` when output is stored rather than returned immediately. Extract only the relevant pages.
5. Return the requested fields with source URLs. Separate missing fields from empty values, identify records that failed extraction, and deduplicate records by a stable source URL or ID. If an extraction is uncertain, verify it against the returned page content or a second source.

## Pitfalls

- MrScraper's rendered fetch is one request per URL, not a persistent CDP session. Use Hermes `browser_navigate` and browser action tools for click/type/scroll workflows.
- The plugin token and hosted MCP OAuth connection are independent. A working MCP connection does not make `web_search` use MrScraper.
- Avoid repeated broad crawls and blind retries: requests can be billed, and a rate-limit or quota error needs a scope or account check. Use `mrscraper_get_account_info` or `mcp__mrscraper__status` when the account state matters.
- Never put an API token in a prompt, URL, result, or `terminal` command. Let the plugin install prompt or MCP OAuth flow store credentials.
- A scraper creation or rerun response may contain an ID instead of final records. Retrieve its results before reporting completion.

## Verification

Confirm the connection exposes the chosen tool. Run one small query or page extraction, inspect the response for a source URL and the requested fields, and check any stored result before expanding the scope. Report inaccessible pages, missing fields, and whether a crawl or run is still pending.
