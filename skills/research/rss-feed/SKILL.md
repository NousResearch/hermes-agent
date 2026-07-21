---
name: rss-feed
description: "Fetch and digest RSS/Atom feeds with a stateless shell pipeline — curl + xmlstarlet + pandoc. Use whenever the user asks to crawl, read, digest, or summarize a feed, get a daily news brief, or set up a cron-driven RSS cron job. Prefer this over blogwatcher when no subscription state is needed."
version: 1.0.0
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [RSS, Atom, Feed-Reader, News, Cron, Summarization]
    related_skills: [blogwatcher]
prerequisites:
  commands: [curl, xmlstarlet, pandoc]
---

# RSS Feed

Stateless RSS/Atom digest workflow built from standard *nix tools: `curl` fetches, `xmlstarlet` parses, `pandoc` cleans item HTML into plain text. Nothing is persisted — every run is fetch, parse, render, discard. Designed for a daily cron that produces a markdown digest the agent then summarizes, with an optional interactive `fzf` picker for ad-hoc reading.

## When to use

- User pastes one or more feed URLs and wants a digest.
- User asks for a "daily news brief" or "what's new on [blog]".
- User wants a cron job that crawls news and dumps a file for the agent to read.
- User wants to skim today's headlines from a list of feeds without setting up a managed reader.

Do NOT use this skill when:

- The user wants persistent read/unread state, OPML import, or managed subscriptions — use `blogwatcher` instead.

## Installation

Only `curl`, `xmlstarlet`, and `pandoc` are required. `fzf` is optional and only needed for the interactive picker.

- Debian/Ubuntu: `apt install xmlstarlet pandoc fzf`
- macOS: `brew install xmlstarlet pandoc fzf`
- Arch: `pacman -S xmlstarlet pandoc fzf`

## feeds.txt format

One URL per line. Blank lines and `#` comments are ignored.

```
# Anthropic news
https://www.anthropic.com/news/rss.xml
https://hnrss.org/frontpage
# comments and blank lines OK
```

A starter file lives at `feeds.example.txt` next to this SKILL.md.

## Workflow — cron / non-interactive

The main path. `crawl.sh` is non-interactive and writes a markdown digest to stdout.

1. Run the crawler against a feeds file or a single URL.
2. Read the markdown digest from stdout.
3. Summarize, group, or reformat as the user asked.

Example:

```bash
bash skills/research/rss-feed/scripts/crawl.sh ~/.hermes/feeds.txt --limit 10
```

Single-feed form (no feeds.txt needed):

```bash
bash skills/research/rss-feed/scripts/crawl.sh https://hnrss.org/frontpage --limit 5
```

`--limit N` caps items per feed. Omit it to take whatever the feed serves.

## Workflow — interactive picker

```bash
bash skills/research/rss-feed/scripts/pick.sh ~/.hermes/feeds.txt
```

Opens `fzf` with item previews. On select, the article URL is fetched with `curl` and rendered through `pandoc` (falling back to `w3m` if available). Requires a TTY and will not work inside a non-interactive agent run — only suggest this when the user is at a shell.

## Cron wiring

Sample crontab line: crawl daily at 07:00 and drop a dated digest the agent can read on demand.

```
0 7 * * * bash $HOME/.hermes/skills/research/rss-feed/scripts/crawl.sh $HOME/.hermes/feeds.txt --limit 15 > $HOME/.hermes/news/$(date +\%Y-\%m-\%d).md
```

The `%` characters must be escaped as `\%` in crontab. Make sure `$HOME/.hermes/news/` exists.

You can also wire this through hermes's built-in scheduler instead of system `cron` — see `cron/jobs.py` in the repo for the job definition format. That route is preferable when the digest should trigger an agent run rather than just write a file.

## Output format

`crawl.sh` emits one `##` section per feed, then a bullet per item. Parse this directly:

```
## Hacker News — https://hnrss.org/frontpage
- **Show HN: foo** — 2026-05-11
  <one-line summary, cleaned via pandoc>
  https://news.ycombinator.com/item?id=...
- **Ask HN: bar** — 2026-05-11
  <one-line summary>
  https://news.ycombinator.com/item?id=...

## Anthropic — https://www.anthropic.com/news/rss.xml
- **Claude 4.7 released** — 2026-05-10
  <one-line summary>
  https://www.anthropic.com/news/claude-4-7
```

Fields per item: title (bold), publish date, single-line summary, canonical URL.

## Tips

- Feed autodiscovery: if the user gives a homepage URL and not a feed URL, fetch the page first and grep for `<link rel="alternate" type="application/rss+xml" href="...">` to find the real feed before adding it to `feeds.txt`.
- Paywalled or summary-only feeds: the digest only carries what the publisher exposed in the feed body. Surface that limitation to the user rather than fabricating article detail.
- Some feeds are unreliable about `pubDate` (missing, malformed, or all set to the fetch time). When dates look wrong, fall back to source order (feed order in `feeds.txt`, item order within a feed) and tell the user.
- Pipe the digest into the agent context as-is — it is already markdown and does not need preprocessing.

## Differences from blogwatcher

- No external binary install needed; uses standard *nix tools only (`curl`, `xmlstarlet`, `pandoc`).
- Stateless — no read/unread tracking, no SQLite, no OPML. Designed for cron-driven digests, not interactive reading sessions.
- Output is plain markdown the agent consumes directly, instead of a CLI with subcommands and a managed database.
