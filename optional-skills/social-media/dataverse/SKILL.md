---
name: dataverse
description: X/Twitter and Reddit search and collection via Bittensor.
version: 1.0.0
author: Volodymyr Truba (Arrmlet), Hermes Agent
license: MIT
platforms: [linux, macos]
prerequisites:
  commands: [dv]
  env_vars: [MC_API]
metadata:
  hermes:
    tags: [x, twitter, reddit, social-media, bittensor, subnet13, dataverse, macrocosmos]
    related_skills: [xurl]
    homepage: https://github.com/macrocosm-os/dataverse-cli
---

# Dataverse Skill

Search real-time X/Twitter and Reddit posts through the decentralized
Bittensor Subnet 13 network with the `dv` CLI, and run Gravity collection
tasks that gather data for up to 7 days and export Parquet datasets.
Read-only — no posting, replying, or account actions, and no X or Reddit
credentials needed; a single free Macrocosmos API key covers both platforms.

---

## When to Use

- User wants to search X/Twitter or Reddit posts by keyword, hashtag, or date range
- User wants X posts from specific accounts (`-u` username filter)
- User wants to look up a specific X post by URL
- User wants to collect a topic over days and export it as a Parquet dataset
- User wants social data without per-platform API credentials

Don't use for: posting, replying, liking, reposting, DMs, or timeline/mention
reads on the user's own account — that is the `xurl` skill.

---

## Prerequisites

The `dv` binary and a free Macrocosmos API key.

```bash
# Install via Cargo (Rust toolchain required)
cargo install dataverse-cli

# Or from source
git clone https://github.com/macrocosm-os/dataverse-cli
cd dataverse-cli && cargo install --path .
```

Get a key at https://app.macrocosmos.ai/account?tab=api-keys, then either:

- run `dv auth` interactively — stores the key with 0600 permissions in
  `~/.config/dataverse/config.toml` (Linux) or
  `~/Library/Application Support/dataverse/config.toml` (macOS), or
- set the `MC_API` environment variable.

Never ask the user to paste the key into chat; `dv auth` and `MC_API` keep it
out of session context. `--dry-run` output redacts the key.

---

## How to Run

Run `dv` through the `terminal` tool. Always pass `-o json` when results will
be parsed or post-processed — the default `table` format is for human display.
Data goes to stdout; diagnostics and errors go to stderr.

```bash
dv search x -k "bittensor" -o json
```

---

## Quick Reference

```bash
dv status                                 # verify key + connectivity
dv auth                                   # interactive key setup

dv search x -k "kw1,kw2" [--mode any|all] [-u user1,user2] \
            [--from YYYY-MM-DD] [--to YYYY-MM-DD] [-l N] [-o json]
dv search x --url "https://x.com/user/status/123"     # single post lookup
dv search reddit -k "r/MachineLearning,LLM" [-l N] [-o json]

dv gravity create -p <x|reddit> -t <topic> [-k keyword] [-n name] [--email addr]
dv gravity status [task_id] [--crawlers]
dv gravity build <crawler_id> [--max-rows N] [--email addr]
dv gravity dataset <dataset_id>           # poll build; returns download URL
dv gravity cancel <task_id>
dv gravity cancel-dataset <dataset_id>
```

Global flags on every command: `-o table|json|csv` (default `table`),
`--api-key <key>`, `--dry-run` (print the request without sending),
`--timeout <seconds>` (default 120).

Limits: ≤ 5 keywords, ≤ 5 usernames (X only), `-l` result limit 1–1000
(default 100). Gravity `-t` topic is a `#hashtag` or `$cashtag` for X, or
`r/subreddit` for Reddit.

---

## Procedure

### 1. Verify setup

```bash
dv status
```

Exits 0 and reports the key as valid when ready. Otherwise walk the user
through Prerequisites.

### 2. Search

```bash
# X: keywords AND-matched, filtered to accounts, custom window
dv search x -k "bittensor,subnet 13" --mode all -u opentensor \
  --from 2026-07-01 --to 2026-07-10 -o json

# Reddit: target a subreddit by passing "r/<name>" as a keyword
dv search reddit -k "r/MachineLearning,decentralized AI" -l 50 -o json
```

Done when the output parses as a JSON array. An empty array usually means the
default 24-hour window was too narrow — retry with an explicit `--from`.

### 3. Bulk collection (Gravity)

```bash
dv gravity create -p x -t "#bittensor" -k "subnet" -n "tao-watch" -o json
dv gravity status <task_id> --crawlers -o json     # note the crawler IDs
dv gravity build <crawler_id> --max-rows 100000 -o json
dv gravity dataset <dataset_id> -o json            # poll until a URL appears
```

Tasks collect for up to 7 days. `build` stops the crawler and snapshots what
was collected; `dataset` reports build progress and, when finished, Parquet
download links. Done when `dataset` returns a download URL.

### 4. Parse results

Every X row carries: `datetime`, `text`, `uri`, `user.username`,
`user.display_name`, `user.followers_count`, `tweet.like_count`,
`tweet.retweet_count`, `tweet.reply_count`, `tweet.view_count`,
`tweet.hashtags`, `tweet.language`. Reddit rows carry `datetime`, `text`,
`uri` plus a smaller metadata set — inspect one row before assuming a field
exists.

---

## Pitfalls

- Date flags are `--from`/`--to`, and there is no `--subreddit` flag —
  target subreddits with an `r/<name>` keyword (search) or `-t r/<name>`
  (gravity create).
- The default search window is the last 24 hours; "no results" often means
  the window, not the query.
- `--mode all` ANDs the keywords; the default `any` ORs them.
- Username filtering is X-only; Reddit ignores `-u`.
- `dv gravity build` stops the crawler — build only when collection should
  finish.
- Data is served by decentralized SN13 miners: coverage and freshness vary
  between runs, and result counts are not deterministic.
- Long queries can exceed the 120 s default; raise `--timeout` instead of
  retrying in a loop.

---

## Verification

- [ ] `dv status` exits 0 and reports a valid API key
- [ ] `dv search x -k "test" -l 5 -o json` returns a parseable JSON array
- [ ] `dv search x -k "test" --dry-run` prints the request without network I/O

---

## Comparison with xurl

| | dataverse | xurl |
|---|-----------|------|
| **Reads X** | Yes (keyword/user search, post lookup) | Yes (search, timelines, mentions, raw v2 API) |
| **Writes to X** | No | Yes (post, reply, like, repost, DM, media) |
| **Reddit** | Yes | No |
| **Auth** | One free Macrocosmos API key | X developer app + OAuth 2.0 PKCE |
| **Data source** | Bittensor SN13 miners | Official X API |
| **Bulk collection** | Yes (Gravity → Parquet) | No |

Use dataverse for cross-platform search and bulk datasets; use `xurl` for
posting, account actions, and official-API reads.
