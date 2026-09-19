---
name: toolsnap
description: Deterministic extraction from web pages and documents.
version: 1.1.0
author: Icosaedro (github.com/icosaedro-git)
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  mcps: [toolsnap]
metadata:
  hermes:
    tags: [web, scraping, extraction, pdf, csv, json, rss, sitemap, seo, x402, context-efficiency]
    category: web
    related_skills: []
    homepage: https://toolsnap.app/agents
---

# ToolSnap Skill

Extract exact, reproducible data from web pages and documents through the
ToolSnap MCP server, which parses server-side and returns only the answer —
large pages never enter the context window. It does not search the web and
does not drive a browser; discovery and interaction stay with native tools.

## When to Use

- Quoting or citing a page exactly, where an LLM summary could drift
- Turning HTML tables, CSV/JSON files, PDFs, RSS feeds, or sitemaps into structured data
- Querying a large dataset for a few rows instead of loading the whole file
- Auditing SEO metadata (title/canonical/og:*, JSON-LD) as data
- Checking token cost of a document before ingesting it

Keep native `web_search` for discovery (ToolSnap does not search) and
`browser_navigate` for JS-heavy SPAs, logins, and page interaction.

## Prerequisites

The `toolsnap` MCP server, installed from the catalog:

```
hermes mcp install official/toolsnap
```

It is a remote server on Cloudflare Workers (streamable HTTP) — nothing runs
locally and the free tools need no account, API key, or configuration.

## How to Run

Call the MCP tools directly. Only a curated core appears in `tools/list`;
the rest of the catalog is reachable through two meta-tools:

1. `tool_catalog()` — families overview; `tool_catalog(family="...")` or
   `tool_catalog(tool="...")` returns a tool's schema.
2. `use_tool(name="...", args={...})` — execute any cataloged tool, same
   behavior as a direct call.
3. `task_recipes()` — ready multi-tool plans (clone a site, SEO audit, …).

## Quick Reference

| Task | Tool |
|---|---|
| Quote or cite a page exactly | `fetch_extract` |
| Rebuild / migrate a page (clean DOM) | `fetch_html` |
| Tables on a page → JSON/CSV | `html_table_extract` |
| Query a big CSV/JSON without loading it | `csv_query` / `json_query` |
| PDF text | `pdf_text_extract` |
| Feeds, sitemaps, link audits | `rss_parse` / `sitemap_parse` / `link_check` |
| SEO metadata and JSON-LD | `fetch_metadata` / `fetch_structured` |
| Token budget check before ingesting | `count_tokens` |

## Procedure

1. Before loading any external URL or document into context, check the
   Quick Reference for a matching tool and call it instead.
2. Prefer one targeted query (`csv_query`, `html_table_extract`) over
   fetching a whole document and filtering in-context.
3. For tools not in `tools/list`, resolve the schema with `tool_catalog`
   and execute with `use_tool`.
4. Paid tools (`screenshot_url` $0.04, `keyword_research` $0.04,
   `remove_background` $0.03) settle per call with USDC on Base via x402.
   Check `pricing()` first and never authorize above the shown price. A
   funded wallet alone is not enough — the client must answer the `402`
   by signing an EIP-3009 transfer and retrying; the local pay-proxy
   (linked from the homepage above) handles this with the key kept on
   the machine, and `account_deposit` (≥ $0.50) makes repeat use cheaper.

## Pitfalls

- ToolSnap fetches server-side: JS-rendered SPAs come back flagged as such,
  not silently empty. When it reports one, switch to `browser_navigate` —
  do not retry blindly.
- There is no search tool; resolving "find the page about X" needs
  `web_search` first, then ToolSnap on the resulting URL.
- Paid calls have real cost per invocation — no first-call-free. Do not
  loop them without checking `pricing()` and the budget.

## Verification

- `tool_catalog()` returns the families overview — the server is reachable
  and the catalog loads.
- `fetch_metadata(url="https://toolsnap.app")` returns the site title as
  structured JSON — end-to-end extraction works without an account.
