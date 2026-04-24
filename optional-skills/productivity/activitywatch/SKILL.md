---
name: activitywatch
category: productivity
description: Use local ActivityWatch data with Hermes for private self-observation, activity summaries, project heat signals, and context-aware review workflows. Covers querying the local ActivityWatch HTTP API, identifying useful bucket types, turning raw events into daily summaries, and feeding the results into Hermes reviews or cron jobs.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [activitywatch, self-tracking, productivity, observability, review, chronicle]
    related_skills: [native-mcp, hermes-agent]
prerequisites:
  commands: [python3, curl]
---

# ActivityWatch

Use a local ActivityWatch server as a private telemetry source for Hermes.

This skill is for:
- daily activity summaries
- project heat analysis
- understanding what apps, windows, editors, and browser tabs dominated a period
- cross-checking perceived work against actual machine activity
- feeding time-windowed signals into private Hermes reviews, not public output

It is not for surveillance theater. The goal is useful self-observation and cleaner portfolio control.

## What ActivityWatch gives you

A local ActivityWatch instance typically exposes bucket data such as:
- `aw-watcher-window_*` — active app/window titles over time
- `aw-watcher-afk_*` — active vs AFK state
- `aw-watcher-vscode_*` — editor/file activity when the VS Code watcher is installed
- `aw-watcher-web-*` — current browser tab activity for supported browser integrations
- `aw-watcher-input_*` — keyboard/mouse input intensity

All of this is available over the local HTTP API, usually at:

```bash
http://127.0.0.1:5600/api/0/
```

## Helper script

This skill now includes `scripts/activitywatch_summary.py` for repeatable local ActivityWatch queries.

```bash
# Server info
python3 SKILL_DIR/scripts/activitywatch_summary.py info --pretty

# List buckets
python3 SKILL_DIR/scripts/activitywatch_summary.py buckets --pretty

# Summarize top windows over the last 24h
python3 SKILL_DIR/scripts/activitywatch_summary.py summary --hours 24 --pretty

# Generate a lightweight project heat hint from window titles
python3 SKILL_DIR/scripts/activitywatch_summary.py heat --hours 24 --pretty
```

`SKILL_DIR` is the directory containing this SKILL.md file.

## Quick checks

### 1. Verify the server is up

```bash
curl -s http://127.0.0.1:5600/api/0/info
```

Expected output looks like JSON with hostname and version.

### 2. List buckets

```bash
curl -s http://127.0.0.1:5600/api/0/buckets/
```

Or use the helper:

```bash
python3 SKILL_DIR/scripts/activitywatch_summary.py buckets --pretty
```

## Most useful pattern: private summary, not raw dump

Do not dump raw ActivityWatch event streams into canon.

Preferred flow:
1. Query only the buckets relevant to the question
2. Summarize a bounded time window (today, yesterday, last 4h)
3. Extract durable or operationally useful signals
4. Write the summary into:
   - a private review artifact
   - `~/brain/reviews/...`
   - or a private note for later curation

## Example: summarize app/window activity for the last 24 hours

Use Python instead of shell parsing so the output stays readable:

```bash
python3 - <<'PY'
import urllib.request, json, datetime as dt
from collections import Counter

BASE = 'http://127.0.0.1:5600/api/0'
now = dt.datetime.now(dt.timezone.utc)
start = (now - dt.timedelta(hours=24)).isoformat()
end = now.isoformat()

buckets = json.loads(urllib.request.urlopen(f'{BASE}/buckets/', timeout=5).read().decode())
window_buckets = [bid for bid, meta in buckets.items() if meta.get('type') == 'currentwindow']

counter = Counter()
for bid in window_buckets:
    url = f'{BASE}/buckets/{bid}/events?start={start}&end={end}'
    events = json.loads(urllib.request.urlopen(url, timeout=10).read().decode())
    for ev in events:
        app = (ev.get('data') or {}).get('app') or 'unknown'
        title = (ev.get('data') or {}).get('title') or ''
        counter[(app, title[:80])] += ev.get('duration', 0) or 0

for (app, title), secs in counter.most_common(20):
    print(f'{secs/3600:5.2f}h  {app:20}  {title}')
PY
```

## Example: build a project heat hint

A lightweight heuristic:
- if active windows repeatedly include repo names, IDE titles, or terminals in a project directory,
- and git activity also shows changes,
- that project may deserve higher heat in Hermes' daily review.

This means ActivityWatch should be treated as an additional signal layer, not the sole truth source.

## Suggested Hermes use cases

### 1. Daily heat scan enrichment
Use ActivityWatch to validate:
- which repos got real focused attention
- whether the hottest repos by file changes match actual attention
- whether a project is emotionally salient but not actually being worked on

### 2. Weekly review reality check
Ask:
- what did Umut think he worked on?
- what did the machine actually show?
- where is there a mismatch between intention and time allocation?

### 3. Context feed for "what was I doing?"
When Umut asks:
- "what was I working on yesterday?"
- "what had my attention this week?"
- "what did I keep context-switching between?"

ActivityWatch is a strong source.

## Good questions to ask with ActivityWatch

- Which apps and windows dominated the last 24 hours?
- Did actual attention align with the current Core projects?
- Which repos had both git movement and real attention?
- What looked hot in git, but not in actual usage?
- Which browser tabs or docs kept recurring during a project push?

## Bad uses

- dumping raw event streams into the brain
- pretending a window title alone proves meaningful work
- using ActivityWatch as the only project truth source
- using private telemetry for public outputs without a very explicit reason

## Relationship to Hermes and OpenClaw

### Hermes
Hermes is a good fit for ActivityWatch when used as:
- a private review input
- a daily/weekly telemetry source
- a context helper for portfolio control

### OpenClaw
OpenClaw can also consume ActivityWatch-like telemetry, but the cleanest first wedge is to integrate it with Hermes review workflows first, then evaluate whether the same logic should be ported or generalized.

Do not build a two-headed integration monster first. Start with one local workflow that produces clearly useful output.

## Practical recommendation

First milestone:
- add ActivityWatch-backed summaries to Hermes private review workflows

Second milestone:
- compare repo activity and ActivityWatch activity together

Third milestone:
- consider a reusable tool, MCP wrapper, or plugin surface if the workflow proves consistently useful

## Output discipline

When using ActivityWatch data, Hermes should return:
1. bounded time window
2. top observed signals
3. interpretation with uncertainty
4. suggested action or implication

Never treat noisy telemetry as unquestionable truth.

---

This skill is intentionally focused on practical review workflows, not maximal telemetry ingestion.
