---
name: reddit-access
description: Research public Reddit content without account actions.
version: 1.0.0
author: Jesse Gonzalez (@ctwhome), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [reddit, research, rss, mcp, social-research, monitoring]
    related_skills: [duckduckgo-search, scrapling]
---

# Reddit Access Skill

Research public Reddit content when direct HTML or `.json` endpoints return an anti-bot challenge. Start with RSS, then use approved read-only backends when RSS lacks needed coverage.

## When to Use

- Research a known subreddit or exact keyword without signing in.
- Use semantic search, comments, or cross-subreddit discovery through an approved read-only backend.
- Read public content only; never post, comment, vote, message, subscribe, moderate, or automate account actions.
- Prefer RSS and approved APIs over CAPTCHA solving, stealth fingerprints, or rotating anonymous proxies.

## Prerequisites

- Python 3; bundled RSS client uses only standard library.
- Installed skill at `${HERMES_HOME:-$HOME/.hermes}/skills/research/reddit-access`.
- For MCP, verify server operator, freshness, privacy policy, terms, and read-only scope.
- Put behavioral configuration in `config.yaml`; use `.env` only for secret credentials.

## How to Run

Subreddit:

```bash
SKILL_DIR="${HERMES_HOME:-$HOME/.hermes}/skills/research/reddit-access"
python3 "$SKILL_DIR/scripts/reddit_rss.py" --subreddit phones --limit 10
```

Keyword search:

```bash
SKILL_DIR="${HERMES_HOME:-$HOME/.hermes}/skills/research/reddit-access"
python3 "$SKILL_DIR/scripts/reddit_rss.py" --query '8 inch phone' --limit 10
```

Output is normalized JSON containing `title`, `url`, `author`, `published`, `text`, `subreddit`, and `source`.

## Quick Reference

Backend order:

1. **RSS** — default; needs no credentials and supports subreddit or keyword searches.
2. **Configured MCP server** — use for semantic search, comments, or cross-subreddit discovery.
3. **Official Reddit OAuth MCP/server** — preferred durable integration when credentials and app approval are available.
4. **Browser session** — interactive, user-approved lookup only; never use a personal session for unattended crawling.

Direct feeds:

```text
https://www.reddit.com/r/phones/.rss
https://www.reddit.com/search.rss?q=8%20inch%20phone
```

Add a vetted remote MCP server with `hermes mcp add reddit-research --url <server-url>`, or use a self-hosted server. Start a fresh Hermes session afterward so tools are discovered.

## Procedure

1. Start with RSS for a known subreddit or exact keyword.
2. If RSS is insufficient, use a configured read-only Reddit MCP for semantic discovery or comments.
3. Cross-check consequential claims against the original Reddit permalink and at least one non-Reddit source.
4. Record `source` and retrieval time.
5. Keep queries narrow and cache repeated monitoring results.
6. Respect Reddit terms, rate limits, robots directives, and provider terms.

## Pitfalls

| Symptom | Action |
|---|---|
| Reddit `.json` or HTML returns 403 | Use RSS; do not retry rapidly |
| RSS returns 403 | Do not hammer it; use an approved MCP/API backend or report the blocker |
| RSS works but comments are missing | Use an approved MCP/OAuth backend |
| MCP is unavailable | Run `hermes mcp list`, then `hermes mcp test reddit-research` |
| Results look stale | Show retrieval time and verify the original permalink |
| OAuth is rejected | Check Reddit app credentials, User-Agent, scopes, and rate limits |

RSS does not promise complete search, historical coverage, or full comment trees. Third-party indexes may be stale, so cite original Reddit URLs and retrieval times. For large-scale or commercial collection, use a provider with an explicit contract and review its Reddit data terms first.

## Verification

- Confirm output is valid JSON and each record contains an original Reddit `url` plus `source`.
- Open important permalinks and record retrieval time before citing results.
- Confirm configured MCP tools are read-only before use.
