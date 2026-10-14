---
name: reddit-reading
description: "Read Reddit: subreddits, search, threads, users. No browser."
version: 1.0.0
author: Teknium (teknium1), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Reddit, Social Media, Research, Discussions, Community]
    related_skills: [rss-feeds, grounded-citations, blocked-page-recovery, xurl]
---

# Reddit Reading Skill

Reads Reddit content — subreddit listings, site or subreddit search, full threads with
comments, and user activity — from a server or headless machine where the normal routes
are dead. It does not post, vote, or log in as a user. Idea credit: the per-platform
backend routing in [Agent Reach](https://github.com/Panniantong/Agent-Reach).

## When to Use

- "What is r/LocalLLaMA saying about X", "find Reddit threads on Y", "summarise this
  Reddit thread", "what has u/someone posted lately".
- Any `reddit.com` URL the user shares. `web_extract`, `browser_navigate` and the
  `.json` endpoints all fail from server IPs (403 or a "Prove your humanity" wall);
  this skill is the working path.
- Not for posting, voting, messaging, or anything needing a user login.

## Prerequisites

**None.** No Reddit account, login, cookie, or API key is needed. The default backend is
Reddit's public Atom feeds (`.rss` endpoints), the only unauthenticated route Reddit still
serves to non-residential IPs. It is throttled to about one request per minute per IP and
returns thinner data (no scores, top-level comments only), which is fine for a few calls.

**Optional upgrade (app credentials, still no user login):** for sustained use or full
data, register a free "script" type app at https://www.reddit.com/prefs/apps and put its
two values in `~/.hermes/.env`:

```
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
```

This is an application registration, not a login: the script uses the app-only
`client_credentials` grant, never a username, password, or browser cookie, and never
acts as the user. With both values set it switches to the OAuth API automatically
(~100 requests per minute, scores, nested comments, `num_comments`); if they are
missing or rejected it falls back to the anonymous feeds and says so on stderr.

| | Anonymous feeds (default) | OAuth app credentials |
|---|---|---|
| Setup | nothing | 1-minute app registration, two `.env` values |
| Rate limit | ~1 request / minute / IP | ~100 requests / minute |
| Thread data | post + top-level comments, no scores | nested comments, scores, comment counts |
| Acts as a user | no | no |

## How to Run

Run every command through `terminal` with the skill-relative script path:

```bash
python3 scripts/reddit.py doctor                                  # which backend, current rate-limit window
python3 scripts/reddit.py sub LocalLLaMA --sort hot --limit 15
python3 scripts/reddit.py search "hermes agent" --sub LocalLLaMA --sort new
python3 scripts/reddit.py thread https://www.reddit.com/r/x/comments/abc123/slug/ --limit 40
python3 scripts/reddit.py user spez --limit 10
python3 scripts/reddit.py --json search "topic"                  # machine-readable
```

## Quick Reference

Every command works on both backends; the script chooses the backend, you never pass a flag.

| Need | Command | Anonymous | OAuth |
|---|---|---|---|
| Subreddit front page | `sub NAME --sort hot\|new\|top\|rising [--time week]` | ✔ | ✔ |
| Search all of Reddit | `search "q" --sort relevance\|new\|top\|comments` | ✔ | ✔ |
| Search one subreddit | `search "q" --sub NAME` | ✔ | ✔ |
| Thread + comments | `thread URL --limit N` | ✔ top-level only, no scores | ✔ nested, scores |
| User posts/comments | `user NAME` | ✔ | ✔ |
| Backend + rate limit | `doctor` | ✔ | ✔ |

## Procedure

① `doctor` once per task if you have not called it this session — it tells you which
backend is live and how many seconds remain in the anonymous window.

② Plan your calls before making them. Anonymous Reddit allows roughly **one request per
minute per IP**; the script sleeps until the window resets on a 429 and retries once, so
a five-call plan costs about five minutes. Prefer one `search --sub` over several `sub`
listings, and read one thread rather than the whole listing.

③ For "what is the community saying" questions, read the thread bodies (`thread`) rather
than stopping at titles; the listing only carries the first ~300 characters of each post.

④ Cite the permalink (`url` field), not the listing page, when the result feeds a report.
`grounded-citations` registers these URLs like any other source.

⑤ If the user needs sustained Reddit access (monitoring, more than ~10 calls), stop and
ask them to register the app credentials (Prerequisites) rather than grinding through the
throttle. Tell them plainly: it is a free app registration, not logging Hermes into their
account. Never ask for a Reddit password or browser cookies.

## Pitfalls

- `www.reddit.com/…/.json`, `api.reddit.com` and `old.reddit.com` return 403 or an
  empty "Welcome to Reddit" shell for datacentre IPs. Do not fall back to them; do not
  spoof a browser User-Agent (also 403).
- `r.jina.ai` and the `browser_navigate` tool hit the same block ("blocked by network
  security" / humanity check). `blocked-page-recovery`'s Wayback route can still recover
  an **old** thread that was archived; it cannot fetch fresh ones.
- Anonymous thread feeds only contain the post plus top-level comments (Reddit caps the
  feed at a handful of entries); scores and reply nesting are OAuth-only.
- Reddit's `limit` on feeds is advisory — expect 5–25 entries regardless of what you ask.
- Never paste `REDDIT_CLIENT_SECRET` into a chat or log; the script reads it from the
  environment only.
- Do not "fix" a 429 by retrying in a loop or adding a proxy; the throttle is per IP and
  the script already waits out the window once. More than one 429 in a row means the
  task needs the app credentials.

## Verification

`python3 scripts/reddit.py doctor` prints `anonymous_feed: ok` and an
`x-ratelimit-reset` value; `sub announcements --limit 1` returns one entry with a
`reddit.com/r/announcements/comments/` URL. With credentials set, `doctor` prints
`active_backend: oauth` and `thread …` output shows numeric scores.
