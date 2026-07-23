---
name: reddit
description: "Use when the user wants to search Reddit posts, read threads and comments, browse subreddits, or check hot/popular feeds. Requires rdt-cli with cookie-based login — no anonymous access exists since Reddit blocked it in 2024."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [reddit, social-media, forum, search, rdt-cli]
    related_skills: [v2ex, xurl]
prerequisites:
  commands: [rdt]
---

# Reddit Skill

Reddit is a massive community/forum platform. Since 2024, Reddit requires
authentication for ALL API access — anonymous `.json` endpoints return 403,
and the official API closed self-service registration in 2025-11.

The `rdt-cli` tool provides read access via cookie-based auth. On desktop,
OpenCLI (Chrome extension that reuses browser login) is also an option.

## When to Use

- User wants to search Reddit for information, opinions, or discussions
- User shares a Reddit post URL and wants the content + comments
- User wants to browse a specific subreddit
- User asks what's trending on Reddit (hot/popular feeds)

Don't use for: posting, commenting, or voting (rdt-cli supports some write ops
but they're secondary). Don't attempt anonymous access — it will always 403.

## Prerequisites

rdt-cli must be installed from git (PyPI version 0.4.1 is behind):

```bash
uv tool install 'git+https://github.com/public-clis/rdt-cli.git@5e4fb3720d5c174e976cd425ccc3b879d52cac66'
```

Then log in (requires being logged into reddit.com in Chrome first):

```bash
rdt login
```

Verify: `rdt status --json` should show `"authenticated": true`.

### Manual Cookie Setup (headless / servers)

If `rdt login` can't access Chrome cookies, use Cookie-Editor to export:

1. Install Cookie-Editor Chrome extension
2. Log into reddit.com → click extension → find `reddit_session` → copy Value
3. Create `~/.config/rdt-cli/credential.json`:
```json
{"cookies": {"reddit_session": "PASTE_VALUE"}, "source": "manual",
 "username": "YOUR_USERNAME", "modhash": null, "saved_at": 0,
 "last_verified_at": null}
```

## Quick Reference

```bash
# Search Reddit
rdt search "query" --limit 10

# Read a post + comments (by ID or full URL)
rdt read POST_ID

# Browse a subreddit
rdt sub LocalLLaMA --limit 20

# Hot / Popular / All feeds
rdt hot --limit 10
rdt popular --limit 10
rdt all --limit 10

# Structured output for AI processing
rdt search "query" --json --limit 5
```

Post IDs are the 6-char string in URLs like
`https://www.reddit.com/r/sub/comments/1abc123/title/`. rdt accepts full URLs.

## Desktop Alternative: OpenCLI

If OpenCLI is installed, it reuses the browser's Reddit login — no setup:

```bash
opencli reddit search "query" -f yaml
opencli reddit read POST_ID -f yaml
opencli reddit subreddit LocalLLaMA -f yaml
opencli reddit hot -f yaml
opencli reddit popular -f yaml
```

## Procedure

1. Verify login: `rdt status --json`
2. If not authenticated, guide the user through login or cookie setup.
3. For research: `rdt search "keyword" --limit 10`
4. For specific posts: `rdt read POST_ID`
5. For community pulse: `rdt sub SUBREDDIT --limit 20`
6. For trending: `rdt hot` or `rdt popular`
7. Use `--json` for structured output when processing results further.

## Common Pitfalls

1. **Must be logged in.** Reddit blocks all anonymous API requests (403).
2. **rdt-cli upstream may be stale.** Last updated 2026-03. If commands break,
   try OpenCLI as fallback.
3. **Cookie expiry.** `reddit_session` cookies can expire. Re-export from
   browser and update `credential.json`.
4. **PyPI version outdated.** Always install from the git source, not
   `pip install rdt-cli`.
5. **Rate limiting.** Space requests by 1-2 seconds.
6. **China access.** Reddit is blocked inside China — proxy required.

## Verification Checklist

- [ ] `rdt --help` shows available commands
- [ ] `rdt status --json` shows `"authenticated": true` (after login)
- [ ] `rdt hot --limit 3` returns trending posts
