---
name: reddit-research
description: Search and analyze Reddit discussions via Reddit's JSON API. Free, no API key required. Search by keyword, subreddit, author, date range. Get posts, comments, scores, engagement metrics.
platforms: [linux, macos, windows]
---

# Reddit Research

Search Reddit discussions using Reddit's public JSON API. **No API key required.**

## Helper script

This skill includes `scripts/reddit_search.py` — a complete CLI tool.

```bash
# Search posts by keyword
python3 SKILL_DIR/scripts/reddit_search.py search "Nvidia" --subreddit wallstreetbets --limit 10

# Recent posts
python3 SKILL_DIR/scripts/reddit_search.py hot "AI" --hours 24

# Search by subreddit
python3 SKILL_DIR/scripts/reddit_search.py subreddit "cryptocurrency" --since 2025 --limit 20

# Get comments from a post
python3 SKILL_DIR/scripts/reddit_search.py comments abc123 --limit 10

# Author activity
python3 SKILL_DIR/scripts/reddit_search.py author "DeepFuckingValue" --limit 5
```

Commands: search, hot, subreddit, comments, author.