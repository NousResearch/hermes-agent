---
name: reddit-research
description: Search Reddit posts and pull comments without an API key.
version: 1.0.0
author: Kewe63
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: research
    tags: [reddit, research, search, comments]
---

# Reddit Research

Search Reddit discussions using the public Reddit JSON API. No API key
required; nothing to configure. Returns JSON the agent can post-process
or surface directly.

## When to Use

- Patch a question from a public Reddit thread (post + top comments).
- Track keyword mentions across one or many subreddits.
- Pull recent posts from a specific subreddit or author.

Skip this skill for DMs, private subreddits, anything that needs OAuth
write access, or anything behind a login wall — Reddit's JSON endpoints
only serve public data without authentication.

## Prerequisites

- A POSIX shell and Python 3.10+ on `$PATH`.
- Outbound HTTPS to `www.reddit.com`.

## How to Run

The helper `scripts/reddit_search.py` wraps the API and prints structured
JSON. The agent runs it via `terminal`:

```bash
# Keyword search across Reddit
python3 SKILL_DIR/scripts/reddit_search.py search "Nvidia" --subreddit wallstreetbets --limit 10

# Recent posts in a subreddit (last N hours)
python3 SKILL_DIR/scripts/reddit_search.py hot "AI" --hours 24

# Subreddit browse with optional lower-time bound
python3 SKILL_DIR/scripts/reddit_search.py subreddit "cryptocurrency" --since 2025 --limit 20

# Top comments on a single post (post_id is the bare id, e.g. "abc123")
python3 SKILL_DIR/scripts/reddit_search.py comments abc123 --limit 10

# Posts by a specific author
python3 SKILL_DIR/scripts/reddit_search.py author "DeepFuckingValue" --limit 5
```

## Quick Reference

| Command     | Positional             | Notable flags                                  |
|-------------|------------------------|------------------------------------------------|
| `search`    | `query`                | `--subreddit`, `--since` (days), `--author`    |
| `hot`       | `query`                | `--hours` (default 24), `--subreddit`          |
| `subreddit` | `name`                 | `--since` (days), `--limit`                    |
| `comments`  | `post_id`              | `--limit`                                      |
| `author`    | `name`                 | `--limit`                                      |

Every command prints a single JSON object to stdout with `results` (and
either `query`, `subreddit`, `author`, or `post_id`) and exits non-zero
only on a programming error, never on an empty result.

## Procedure

1. Pick the command that matches the question shape — use `comments` only
   when you already have a specific post id.
2. Tune `--limit` to the smallest number that answers the question; large
   limits add latency and risk rate-limiting.
3. Parse the JSON output rather than scraping stdout text.
4. When summarizing, cite the `permalink` field verbatim so the user can
   open the source thread.

## Pitfalls

- **Rate limiting.** Reddit throttles unauthenticated clients. Keep
  `--limit` modest (10-25) and avoid tight loops; on a `429`/`503`
  response the script returns `{"error": ...}` instead of raising.
- **`post_id` is the bare id**, not a full URL. Strip the
  `https://reddit.com/comments/` prefix if you have a URL.
- **Top-level comments only.** `fetch_post_comments` walks `depth == 0`
  comments to keep the response small; follow `permalink` for nested
  reply chains.
- **Search result ordering.** Reddit's JSON endpoint sorts by relevance
  for `search`; if chronological order matters, omit `q` and use
  `subreddit` instead.
- **Removed/deleted authors** appear as `[deleted]`; don't treat that as
  a missing field error.

## Verification

Run the bundled test suite to confirm the helper still behaves:

```bash
scripts/run_tests.sh tests/skills/test_reddit_research_skill.py -q
```

Tests stub `urllib.request.urlopen` so no live network calls happen.
