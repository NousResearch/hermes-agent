---
name: x-research-mcp-api
description: "X research via MCP tools and v2 API: search, user tweets, article fetch."
version: 1.0.0
author: Axl Ibiza, MBA
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [x, twitter, research, mcp, api, search, osint, social]
    related_skills: [x-api-publishing, x-profile-research, x-profile-interaction-research]
---

# X Research — MCP tools + v2 API

Use when researching X/Twitter: searching recent posts, pulling a user's
tweets, reading a long-form Article's full body, or gathering public
evidence (OSINT, interaction timelines, citation sources). Covers the
`mcp__xai__*` tools when the xai MCP server is connected, plus the direct
v2 read API when you need fields the MCP tools don't expose.

## Tool surface (xai MCP, when connected)

| Tool | Use |
|---|---|
| `mcp__xai__search_tweets` | Search recent posts (last ~7 days). `query` supports X operators (`from:`, `to:`, `has:`, `is:`); `limit` 1–100. |
| `mcp__xai__get_user_tweets` | Recent posts by a username. `limit` 1–100; pagination via `next_token` in the response `meta`. |
| `mcp__xai__get_article` | **Full body of an X Article** by its tweet ID — the only tool that returns the complete long-form text (with `entities` for mentions/URLs, `preview_text`, `cover_media`). |
| `mcp__xai__get_tweet` | Single post by ID with full metrics. |
| `mcp__xai__get_user` | User record by username. |
| `mcp__xai__get_user_followers` / `get_user_following` | Connection lists. |
| `mcp__xai__get_user_liked_tweets` / `get_bookmarks` | Engagement history (bookmarks = own account only). |
| `mcp__xai__get_home_timeline` | Authenticated home feed. |

## Search pitfalls (learned the hard way)

1. **Queries with `min_faves:5000` + bare terms can 400.** The search API is
   strict about operator placement and some terms. When a query 400s,
   simplify: drop the metric filter, reduce operators, keep `lang:en` etc.
   short. Retry with a narrower literal query.
2. **`from:NousResearch` + OR-joined terms returned noise** (unrelated
   anime/art posts) in one session — the search endpoint may ignore or
   mis-handle some operator combos at Basic tier. For reliable user-specific
   pulls, prefer `get_user_tweets` over `search_tweets` with `from:`.
3. **Result cap**: `gh`-style pagination is NOT available; the MCP search
   returns what the tier allows (Basic tier ~7-day window). For older posts,
   use the direct v2 API with pagination tokens.
4. **Tweets search ≠ full-text archive.** Search indexes recent posts;
   `get_article` is the only full-body long-form reader.

## v2 read API (when the MCP tools fall short)

Credentials: same OAuth 1.0a as `x-api-publishing` (env: `TWITTER_API_KEY`,
`TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_SECRET`).

- `GET /2/users/me` — verify the token works (200 + your user id).
- `GET /2/users/by/username/{username}` — user lookup, `user.fields=created_at,public_metrics`.
- `GET /2/users/{id}/tweets` — full paginated user timeline
  (`max_results=100`, `pagination_token` loop).
- `GET /2/tweets/search/recent?query=...` — recent search with
  `tweet.fields=created_at,public_metrics,entities`.
- `GET /2/articles/{article_id}` — Article by ID (requires Articles API
  access).
- **Rate limits**: the v2 read endpoints share the generic bucket
  (`x-rate-limit-remaining`); the ARTICLES endpoints have their own burst
  limiter (see `x-api-publishing`). Reads are cheap; writes are not.

## Research workflow (gated)

1. **Model the question first** (who/what/when/evidence needed) — same
   double-blind discipline as the god-file method.
2. **Pull raw data with receipts**: `get_user_tweets` for a user's timeline,
   `search_tweets` for topical evidence, `get_article` for full long-form
   bodies. Save each response with its `created_at` and post ID.
3. **Cross-source verify** every claim you'll publish: the post's own
   timestamp, the user's ID, the article's `edit_history_tweet_ids`.
4. **Cite with real dates** — feed the results into `apa7-references`
   (X posts: first-20-words title, [Post]; articles: [X Article]).
5. **Never fabricate** a post, user, metric, or interaction. If the API
   returns nothing, report the empty result — a gap is data.

## Pitfalls

- The xai MCP can go unreachable mid-session (`MCP server 'xai' is
  unreachable after 3 consecutive failures`) — switch to the direct v2 API
  with the same OAuth env rather than retrying the MCP blindly.
- `get_article` returns the article for a tweet ID — NOT for a bare
  `x.com/.../status/<id>` URL; strip the URL down to the numeric ID first.
- Article responses include `plain_text` (full body) + `entities.urls`
  (all hyperlinks with their text labels) — use those to enumerate a
  document's complete citation set.
- Basic-tier search is a 7-day window; historical research needs the paid
  archive or saved evidence.

## Files

- `templates/x_research.py` — OAuth 1.0a read helpers (users/me, user
  timeline, recent search) — same signing as `x-api-publishing`'s template.
