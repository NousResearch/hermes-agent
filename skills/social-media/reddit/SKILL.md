---
name: reddit
description: Read and search Reddit — browse subreddits, read posts and comments, search across Reddit, view user activity, and monitor multiple subreddits. Uses public JSON endpoints, no API key required.
version: 1.0.0
author: 0xbyt4
license: MIT
platforms: [linux, macos, windows]
prerequisites:
  commands: [python3]
metadata:
  hermes:
    tags: [reddit, social-media, search, research, community]
---

# Reddit — Read & Search

Browse subreddits, read posts and comments, search Reddit, and track user activity. Uses Reddit's public JSON endpoints — no API key or OAuth required.

This skill is for:
- browsing subreddit feeds (hot, new, top, rising)
- searching posts across Reddit or within a subreddit
- reading full posts with comments
- checking user activity
- getting subreddit info and stats
- monitoring multiple subreddits at once

This skill is NOT for:
- posting, commenting, or voting (read-only)
- accessing private or quarantined subreddits
- bulk data scraping or archival

## Quick Reference

| Command | Description |
|---------|-------------|
| `python3 reddit.py hot <sub>` | Hot posts from a subreddit |
| `python3 reddit.py new <sub>` | Newest posts |
| `python3 reddit.py top <sub> --time week` | Top posts by time period |
| `python3 reddit.py search "query"` | Search all of Reddit |
| `python3 reddit.py search "query" --subreddit <sub>` | Search within a subreddit |
| `python3 reddit.py post <sub> <id>` | Read a post with comments |
| `python3 reddit.py user <name>` | View user activity |
| `python3 reddit.py about <sub>` | Subreddit info and stats |
| `python3 reddit.py multisub <s1,s2,s3>` | Combined feed from multiple subs |

All commands accept `--limit N` (default: 10) and `--json` for machine-readable output.

## Usage

### Browse a Subreddit

```bash
python3 reddit.py hot MachineLearning --limit 5
python3 reddit.py new LocalLLaMA --limit 10
python3 reddit.py top artificial --time week --limit 10
python3 reddit.py rising programming
```

Time filters for `top`: `day`, `week`, `month`, `year`, `all`

### Search

```bash
# Global search
python3 reddit.py search "hermes model" --limit 5

# Search within a subreddit
python3 reddit.py search "fine tuning" --subreddit LocalLLaMA --sort new

# Sort options: relevance, new, hot, top
python3 reddit.py search "RAG pipeline" --sort top --limit 20
```

### Read a Post with Comments

Extract the post ID from the URL. For `https://reddit.com/r/artificial/comments/1sfqix8/...`, the ID is `1sfqix8`.

```bash
python3 reddit.py post artificial 1sfqix8
```

Shows the full post text and top comments with nesting (depth=3).

### User Activity

```bash
python3 reddit.py user spez --limit 10
```

Shows recent posts and comments by the user.

### Subreddit Info

```bash
python3 reddit.py about MachineLearning
```

Shows subscriber count, active users, creation date, description.

### Multi-Subreddit Feed

Combine multiple subreddits into one feed:

```bash
python3 reddit.py multisub "MachineLearning,LocalLLaMA,artificial" --limit 15
python3 reddit.py multisub "python,golang,rust" --sort new --limit 10
```

### JSON Output

Add `--json` to any command for structured output:

```bash
python3 reddit.py --json hot LocalLLaMA --limit 5
python3 reddit.py --json search "LLM agents" --subreddit artificial
python3 reddit.py --json post artificial 1sfqix8
```

## Agent Workflow

1. Start with `about` to understand the subreddit's scope and size
2. Use `hot` or `top --time week` to get a feel for what the community cares about
3. Use `search` to find specific topics or discussions
4. Use `post` to deep-dive into interesting threads with comments
5. Use `multisub` to cross-reference discussions across related communities
6. Use `--json` when you need to extract specific data for further processing

## Rate Limits

Reddit's public JSON endpoints have a low rate limit for unauthenticated requests (~10 req/min). To stay within limits:

- Add reasonable pauses between requests (2-3 seconds)
- Use `--limit` to fetch only what you need
- Prefer `search` over browsing when looking for specific topics
- Cache results when possible instead of re-fetching

## Limitations

- **Read-only**: no posting, commenting, voting, or messaging
- **Public subreddits only**: private and quarantined subs return 403
- **No authentication**: rate limits are lower than authenticated API
- **Comment depth**: nested replies are fetched up to 3 levels deep
- **NSFW content**: flagged in output but not filtered by default
- **Unofficial endpoint**: Reddit may change or restrict these endpoints at any time

## Notes

- Subreddit names are case-insensitive (`MachineLearning` = `machinelearning`)
- Post IDs are the alphanumeric string in Reddit URLs after `/comments/`
- The `multisub` command uses Reddit's built-in multi-subreddit syntax (r/sub1+sub2)
- All timestamps are shown as relative time (e.g., "2h ago", "3d ago")
