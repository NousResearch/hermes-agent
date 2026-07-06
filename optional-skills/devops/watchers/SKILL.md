---
name: watchers
description: Poll RSS, JSON APIs, and GitHub with watermark dedup.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [cron, polling, rss, github, http, automation, monitoring]
    category: devops
    requires_toolsets: [terminal]
    related_skills: []
---

# Watchers

Poll external sources on an interval and react only to new items. Three ready-made scripts plus a shared watermark helper; wire them into a cron job (or run them ad-hoc from the terminal).

## When to Use

- User wants to watch an RSS/Atom feed and be notified of new entries
- User wants to watch a GitHub repo's issues / pulls / releases / commits
- User wants to poll an arbitrary JSON endpoint and get notified on new items
- User asks for "a watcher for X" or "notify me when X changes"

## Mental model

A watcher is just a script that:

1. Fetches data from the external source
2. Compares against a watermark file of previously-seen IDs
3. Writes the new watermark back
4. Prints new items to stdout (or nothing on no-change)

The scripts below handle all three. The agent runs them via the terminal tool — from a cron job, a webhook, or an interactive chat — and reports what's new.

## Ready-made scripts

All three live in `$HERMES_HOME/skills/devops/watchers/scripts/` once the skill is installed. Each reads `WATCHER_STATE_DIR` (defaults to `$HERMES_HOME/watcher-state/`) for its state file, keyed by the `--name` argument.

| Script | What it watches | Dedup key |
|---|---|---|
| `watch_rss.py` | RSS 2.0 or Atom feed URL | `<guid>` / `<id>` |
| `watch_http_json.py` | Any JSON endpoint returning a list of objects | Configurable id field |
| `watch_github.py` | GitHub issues / pulls / releases / commits for a repo | `id` / `sha` |

All three:

- First run records a baseline — never replays existing feed
- Watermark is a bounded ID set (max 500) to cap memory
- Output format: `## <title>\n<url>\n\n<optional body>` per item
- Empty stdout on no-new — the caller treats that as silent
- Non-zero exit on fetch errors

## Usage

Run a watcher directly from the terminal tool:

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_rss.py \
  --name hn --url https://news.ycombinator.com/rss --max 5
```

Watch a GitHub repo (set `GITHUB_TOKEN` in `~/.hermes/.env` to avoid the 60 req/hr anonymous rate limit):

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_github.py \
  --name hermes-issues --repo NousResearch/hermes-agent --scope issues
```

Poll an arbitrary JSON API:

```bash
python $HERMES_HOME/skills/devops/watchers/scripts/watch_http_json.py \
  --name api --url https://api.example.com/events \
  --id-field event_id --items-path data.events
```

## Wiring into cron

Ask the agent to schedule a cron job with a prompt like:

> Every 15 minutes, run `watch_rss.py --name hn --url https://news.ycombinator.com/rss`. If it prints anything, summarize the headlines and deliver them. If it prints nothing, stay silent.

The agent invokes the script via the terminal tool inside the cron job's agent loop; no changes to cron's built-in `--script` flag are needed.

## State files

Every watcher writes `$HERMES_HOME/watcher-state/<name>.json`. Inspect:

```bash
cat $HERMES_HOME/watcher-state/hn.json
```

Force a replay (next run treated as first poll):

```bash
rm $HERMES_HOME/watcher-state/hn.json
```

## Writing your own

All three scripts use the same template: load watermark, fetch, diff, save, emit. `scripts/_watermark.py` is the shared helper; import it to get atomic writes + bounded ID set + first-run baseline for free. See any of the three reference scripts for how little boilerplate it takes.

## Common Pitfalls

1. **Printing a "no new items" header every tick.** Callers rely on empty stdout = silent. If you print anything on an empty delta, you spam the channel. The shipped scripts handle this; custom scripts must too.
2. **Expecting the first run to emit items.** It won't — first run records a baseline. If you need an initial digest, delete the state file after the first run or add a `--prime-with-latest N` flag in your own script.
3. **Unbounded watermark growth.** The shared helper caps at 500 IDs. Raise it for high-churn feeds; lower it on constrained filesystems.
4. **Putting the state dir where the agent's sandbox can't write.** `$HERMES_HOME/watcher-state/` is always writable. Docker/Modal backends may not see arbitrary host paths.

