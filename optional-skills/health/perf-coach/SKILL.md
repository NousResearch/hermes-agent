---
name: perf-coach
description: Read-only training load, scores, today's plan, and weight.
version: 1.0.0
author: zealchaiwut, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [health, fitness, training, perf-coach, discord, cron]
    category: health
    related_skills: [fitness-nutrition]
    requires_toolsets: [terminal]
    config:
      - key: perf_coach.port
        description: Local port the perf-coach read API listens on
        default: 8000
        prompt: "perf-coach API port (localhost)"
      - key: perf_coach.bedtime_time
        description: Local 24h time (HH:MM) to send the nightly bedtime check-in
        default: "21:30"
        prompt: "Bedtime check-in time (HH:MM, 24h)"
      - key: perf_coach.render_base_url
        description: Base URL of the Render web app used for deeplinks (log weight, habit checker)
        default: "https://perf-coach.onrender.com"
        prompt: "Render app base URL"
---

# Perf Coach

Read-only bridge between Hermes and a locally running **perf-coach** training
app. Surfaces training load (CTL/ATL/TSB/ACWR plus a pre-computed verdict),
performance scores, today's planned session, and recent weight. Nothing in
this skill writes back to perf-coach — there is no write path in its script,
and it never collects or stores weight/habit data on Hermes's side.

## When to Use

- User asks about today's training load, readiness, or whether to push or back off
- User asks "what does my load say for today?" or similar training-talk questions
- The nightly bedtime check-in cron job fires (see "Bedtime Check-In" below)
- User wants a quick look at performance scores, today's planned session, or recent weight trend

## Prerequisites

- perf-coach running locally with its read API reachable at `http://localhost:<port>`
- `python3` (stdlib only — no pip installs)
- Skill config (set via `hermes skills config`, or read from the `[Skill config]`
  block injected when this skill loads):
  - `perf_coach.port` — local API port (default `8000`)
  - `perf_coach.bedtime_time` — nightly check-in time, 24h `HH:MM` (default `21:30`)
  - `perf_coach.render_base_url` — Render web app base URL for deeplinks
    (default is a placeholder — replace it with your actual deployment before
    relying on the bedtime nudge's links)

## How to Run

All four reads go through one helper script, one `GET` per call, run via
`terminal`. `$HERMES_HOME/skills/health/perf-coach/scripts/` is where this
skill lives once installed:

```bash
python3 $HERMES_HOME/skills/health/perf-coach/scripts/perf_coach.py training_load --port <perf_coach.port>
python3 $HERMES_HOME/skills/health/perf-coach/scripts/perf_coach.py scores        --port <perf_coach.port>
python3 $HERMES_HOME/skills/health/perf-coach/scripts/perf_coach.py today         --port <perf_coach.port>
python3 $HERMES_HOME/skills/health/perf-coach/scripts/perf_coach.py weight        --port <perf_coach.port>
```

Each call issues exactly one `GET` and nothing else — `perf_coach.py` has no
POST/PUT/PATCH/DELETE path, by design. Output is shrunk before it reaches
you: list fields (like weight history) are capped at the last 10 entries,
and any still-oversized response is replaced with a truncated preview so a
single read can never blow the context budget.

## Quick Reference

| Tool | Endpoint | Purpose |
|---|---|---|
| `training_load` | `GET /api/training/load` | CTL/ATL/TSB/ACWR + pre-computed verdict (`back_off` / `hold` / `build`) |
| `scores` | `GET /api/scores` | Performance scores |
| `today` | `GET /api/plan/today` | Today's planned session |
| `weight` | `GET /api/weight/recent` | Recent weight entries |
| `bedtime` | combines `training_load` + `today` + `weight` | One-call snapshot for the nightly cron job |

## Procedure

### Training Talk

When asked things like "what does my load say for today?", "should I push
today?", or "how am I trending?":

1. Run `training_load` then `today` — two calls, no more, no retries.
2. Read the `verdict` field straight from the `training_load` response
   (`back_off` / `hold` / `build`, whatever value the API returns).
   **Never infer your own push/back-off judgment from CTL/ATL/TSB/ACWR — the
   API already computed the verdict; your job is to narrate it, not
   re-derive it.**
3. Reply in plain language: lead with the verdict, then CTL/ATL/TSB/ACWR as
   supporting numbers, then today's planned session from `today`. Keep it to
   a short paragraph — this is a quick check-in, not a report.
4. If perf-coach is unreachable (the script prints an `error` field), say so
   plainly and stop. Do not guess at numbers or invent a verdict.

### Bedtime Check-In (cron)

A nightly, read-only nudge. Hermes never collects or stores weight or habit
data itself — it reports a snapshot and links out to the Render app for
anything that needs to be entered.

**One-time setup** (when the user asks to turn on bedtime check-ins):

1. Read `perf_coach.bedtime_time` (`HH:MM`) and `perf_coach.render_base_url`
   from the injected skill config.
2. Convert the time to a 5-field cron expression: `"<minute> <hour> * * *"`
   (Hermes cron evaluates this in the user's configured local timezone, so
   no extra timezone handling is needed).
3. Call the `cronjob` tool with `action="create"`:
   - `schedule` — the cron expression from step 2
   - `skills` — `["perf-coach"]`
   - `prompt` — a self-contained instruction, e.g.: *"Run
     `python3 $HERMES_HOME/skills/health/perf-coach/scripts/perf_coach.py
     bedtime --port <perf_coach.port>` once via terminal, then send tonight's
     bedtime check-in using the Bedtime Check-In message format from the
     perf-coach skill and `perf_coach.render_base_url` = `<resolved URL>`.
     One tool call only — do not retry or call any other tool."*
   - `enabled_toolsets` — `["terminal"]` (nothing else — keeps the run to one
     tool call and stops the agent from wandering into unrelated tools)
   - `deliver` — `"origin"` (delivers back to wherever the job was created —
     set it up from the Discord conversation you want nudges in)
   - `name` — `"perf-coach bedtime check-in"`
4. Confirm the schedule and target channel back to the user.

**Every night:** the agent runs the `bedtime` snapshot once via `terminal`
(one combined `GET` × 3 endpoints in a single process), then composes the
final response — no further tool calls. This keeps the whole run to a single
tool call, which matters because cron jobs run unattended on a rate-limited
free model with no one there to interrupt a stuck loop.

Message format (compose as the final response — cron delivers it verbatim):

```
Bedtime check-in
Verdict: <verdict from training_load, verbatim>
CTL <ctl> / ATL <atl> / TSB <tsb> / ACWR <acwr>
Weight trend: <short read of the last few weight entries, e.g. "142.1 lb, -0.4 lb this week">

Log tonight's weight: <perf_coach.render_base_url>/weight/log
Tonight's habits — tick them off: <perf_coach.render_base_url>/habits?date=today
```

Both links are **deeplinks into the Render web app** — Hermes does not host
a weight-entry form or a habit checklist, and no habit state is stored in
Hermes. If tonight's habit names are visible in the `today` snapshot, list
them as a plain-text reminder above the link; otherwise omit the list and
send just the link.

## Pitfalls

- Don't add a `--method` flag or any write path to `scripts/perf_coach.py` —
  this skill is read-only by design. A future write-capable perf-coach
  integration belongs in a different skill, not a mode flag on this one.
- Don't recompute `back_off` / `hold` / `build` from CTL/ATL/TSB/ACWR
  yourself — always use the API's own `verdict` field.
- Don't build an in-chat habit form. The habit checker is a deeplink
  (`/habits?date=today`) into the Render app, never a conversation flow, and
  Hermes must not store habit-completion state anywhere.
- Keep the bedtime cron job's `enabled_toolsets` restricted to `["terminal"]`
  and call the `bedtime` snapshot exactly once — every extra tool available
  to an unattended cron run is a chance for a rate-limited free model to
  burn iterations it doesn't have.
- If `perf_coach.render_base_url` is still the placeholder default, tell the
  user to set it (`hermes config set skills.config.perf_coach.render_base_url
  <url>`) before relying on the bedtime nudge's links.

## Verification

- `python3 scripts/perf_coach.py today --port <port>` returns valid JSON with
  no `error` key when perf-coach is running.
- Training-talk replies always name the verdict using the API's own word,
  never "you should push" language invented independently — and never use
  more than 2 tool calls.
- `cronjob` `action="list"` shows the bedtime job with `enabled_toolsets:
  ["terminal"]` and a prompt that calls for exactly one `terminal` invocation
  — confirming it can't loop.
