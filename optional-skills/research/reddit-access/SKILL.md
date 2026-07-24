---
name: reddit-access
description: Read-only Reddit research with an RSS-first fallback and optional MCP/OAuth backends. Use when Reddit blocks direct browser or JSON access.
version: 1.0.0
author: hermes-agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [reddit, research, rss, mcp, social-research, monitoring]
    related_skills: [duckduckgo-search, scrapling]
---

# Reddit Access

Read-only Reddit research for situations where direct Reddit HTML or `.json` endpoints return an anti-bot challenge.

## Safety and scope

- Read public content only.
- Do not post, comment, vote, message, subscribe, moderate, or automate account actions.
- Prefer RSS and approved APIs over CAPTCHA solving, stealth fingerprints, or rotating anonymous proxies.
- Respect Reddit's terms, rate limits, robots directives, and the source provider's terms.
- Treat third-party indexes as potentially stale; cite the original Reddit URL and retrieval time.

## Backend order

1. **RSS** — default; no credentials and works for a subreddit or keyword search.
2. **Configured MCP server** — use when semantic search, comments, or cross-subreddit discovery is needed.
3. **Official Reddit OAuth MCP/server** — preferred for a durable production integration when credentials and app approval are available.
4. **Browser session** — only for an interactive, user-approved lookup; never use a personal session for unattended crawling.

## RSS quick start

Use the bundled stdlib client; it does not require a Python package:

```bash
python3 optional-skills/research/reddit-access/scripts/reddit_rss.py \
  --subreddit phones --limit 10

python3 optional-skills/research/reddit-access/scripts/reddit_rss.py \
  --query '8 inch phone' --limit 10
```

The command returns normalized JSON with `title`, `url`, `author`, `published`, `text`, `subreddit`, and `source` fields.

Direct feed forms:

```text
https://www.reddit.com/r/phones/.rss
https://www.reddit.com/search.rss?q=8%20inch%20phone
```

If a feed returns 403, do not hammer it. Try the configured MCP/API backend or report the blocker.

## Hermes MCP option

Hermes already supports remote MCP servers. Add a read-only Reddit research server only after checking its operator, freshness, privacy policy, and terms:

```bash
hermes mcp add reddit-research \
  --url https://reddit-research-mcp.fastmcp.app/mcp
```

A self-hosted MCP server can be used instead. Keep it scoped to read-only tools and configure it in `config.yaml`, not `.env`, except for secret credentials. Restart or start a fresh Hermes session after adding an MCP server so its tools are discovered.

## Research workflow

1. Start with the RSS client for a known subreddit or exact keyword.
2. If RSS is insufficient, use the configured Reddit MCP for semantic discovery or comments.
3. Cross-check important claims against the original Reddit permalink and at least one non-Reddit source when the claim is consequential.
4. Include `source` and retrieval time in notes or reports.
5. Keep queries narrow and cache results when monitoring repeatedly.

## Troubleshooting

| Symptom | Action |
|---|---|
| Reddit `.json` or HTML returns 403 | Use RSS; do not retry rapidly |
| RSS works but comments are missing | Use an approved MCP/OAuth backend |
| MCP is unavailable | Check `hermes mcp list`, then `hermes mcp test reddit-research` |
| Results look stale | Show retrieval time and verify the original permalink |
| OAuth is rejected | Check Reddit app credentials, User-Agent, scopes, and rate limits |

## Limitations

RSS is intentionally small and conservative: it does not promise complete search, historical coverage, or full comment trees. For large-scale or commercial collection, use a provider with an explicit contract and review its Reddit data terms before enabling it.
