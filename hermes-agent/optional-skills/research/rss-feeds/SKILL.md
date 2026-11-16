---
name: rss-feeds
description: "Read RSS, Atom, JSON feeds; discover feeds behind a page."
version: 1.0.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [RSS, Atom, Feeds, Monitoring, Research, Blogs, Releases]
    related_skills: [reddit-reading, competitor-news-monitor, grounded-citations, youtube-content, blogwatcher]
---

# RSS Feeds Skill

Reads any RSS 2.0, RSS 1.0/RDF, Atom, or JSON Feed URL into a clean, date-sorted list of
entries, and discovers the feed behind an ordinary page URL (`<link rel="alternate">` or
the usual `/feed`, `/rss.xml`, `/atom.xml` paths). Standard library only, nothing to
install. It does not fetch full article bodies — pass an entry's link to `web_extract` for
that.

## When to Use

- "What's new on <blog/site>", "latest releases of <GitHub repo>", "recent posts in
  <subreddit>", "read this feed", "does this site have an RSS feed".
- Building a recurring digest with `cronjob_manage` (feeds are cheaper and more stable than
  scraping the HTML front page every run). For a persistent read/unread database across
  many feeds install the optional `blogwatcher` skill; this skill is the zero-install read.
- Anything where a structured list of `title / link / date / author / summary` beats a
  rendered page: podcasts, changelogs, YouTube channels, newsrooms, forum categories.

## Prerequisites

None. Python 3.10+, network access to the feed host.

## How to Run

Run through `terminal` with the skill-relative script path:

```bash
python3 scripts/feed.py read https://hnrss.org/frontpage --limit 10
python3 scripts/feed.py read https://simonwillison.net/            # page URL → discovers the feed
python3 scripts/feed.py read URL --since 2026-09-01 --json          # only newer entries, machine-readable
python3 scripts/feed.py discover https://example.com/               # list candidate feed URLs
```

## Quick Reference

| Source | Feed URL pattern |
|---|---|
| GitHub releases / commits / tags | `https://github.com/OWNER/REPO/releases.atom`, `…/commits/BRANCH.atom`, `…/tags.atom` |
| Subreddit / Reddit search | `https://www.reddit.com/r/NAME/.rss`, `https://www.reddit.com/search.rss?q=…` (1 req/min anon; see `reddit-reading`) |
| YouTube channel | `https://www.youtube.com/feeds/videos.xml?channel_id=UC…` |
| Hacker News | `https://hnrss.org/frontpage`, `https://hnrss.org/newest?q=TERM` |
| arXiv category | `https://rss.arxiv.org/rss/cs.CL` |
| Substack / Medium / WordPress / Ghost | `SITE/feed`, `medium.com/feed/@user`, `SITE/rss/` |
| Podcasts | the show's RSS URL from its hosting page (`discover` finds it) |

Output fields per entry: `title`, `link`, `published` (UTC ISO 8601), `author`, `summary`
(HTML stripped, ≤ 2000 chars). Entries are sorted newest-first.

## Procedure

① If you only have a site URL, run `read` on it directly; the script discovers the feed
and reports which URL it used (`discovered_from`). Use `discover` when you want to choose
between several advertised feeds (comments feed vs posts feed, per-category feeds).

② Bound the request: `--limit` for "latest N", `--since YYYY-MM-DD` for "since last
check". For a cron digest persist the last-seen `published` value and pass it as
`--since` next run.

③ For full text, hand the entry `link` to `web_extract`; feed summaries are frequently
truncated or the first paragraph only.

④ Cite the entry `link`, not the feed URL, when the result feeds a report
(`grounded-citations`).

## Pitfalls

- A 200 response with HTML means the URL is a page, not a feed; the script falls through
  to discovery automatically, but a site with no `<link rel="alternate">` and none of the
  common paths reports `no feed found` — check the site's footer or `/sitemap.xml` before
  concluding there is none.
- Reddit feeds share Reddit's anonymous throttle (about one request per minute per IP).
  Chain them through `reddit-reading`, which waits out the window, when you need more
  than one Reddit call.
- Dates: RSS `pubDate` is RFC 822 and Atom uses ISO 8601; the script normalises both to
  UTC. Feeds that omit dates sort to the bottom and are dropped by `--since`.
- Some feeds are Cloudflare-fronted and 403 non-browser clients; `blocked-page-recovery`
  handles that class.

## Verification

`python3 scripts/feed.py read https://github.com/NousResearch/hermes-agent/releases.atom
--limit 1` prints one entry with a `releases/tag/` link and a `[atom]` format tag;
`discover https://simonwillison.net/` prints an `/atom/` URL.
