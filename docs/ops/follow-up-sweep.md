# Runbook — `follow-up-sweep` Hermes cron

**Phase:** 028-B (Plan 028 — Linear `follow-up:` auto-open)
**Owner:** Blake
**Cadence:** Weekly, Monday 09:00 America/Los_Angeles

## Purpose

Closes the feedback loop on Plan 028. Every Monday morning, Hermes
queries Linear for issues labeled `follow-up:*` older than 30 days and
DMs Blake on Slack with the list. Without this sweep, the
`follow-up:<plan-id>` rows that 028-A / 028-A1 open at plan close would
silently rot inside Linear — the exact failure mode 028 was built to
eliminate.

## Behaviour

1. Query Linear via GraphQL for issues matching
   `labels.name.startsWith == "follow-up:" AND createdAt < now - 30d
    AND state.type IN (backlog, unstarted, started, triage)`.
2. Compute age in days per row; sort oldest first.
3. Post one block-kit summary message to Blake's Slack DM:
   `:clipboard: Follow-up backlog sweep (YYYY-MM-DD): NN issues > 30
   days old, MM > 90 days.`
4. Thread one reply per row with Linear URL, plan id, age, and title.
5. **Empty case is loud.** Zero stale rows still produces one positive-
   confirmation message (`:white_check_mark: ... 0 stale follow-ups
   this week.`) so silence cannot be confused with a broken cron.

## Configuration

| Env var | Required | Purpose |
|---|---|---|
| `LINEAR_API_KEY` | yes | Linear personal API key (`lin_api_...`). Already set for the rest of the Linear surface. |
| `SLACK_BOT_TOKEN` | yes | Slack bot token (`xoxb-...`). Same token the gateway uses (reuse-over-create). |
| `SLACK_FOLLOW_UP_CHANNEL` | yes | DM channel id for Blake (e.g. `D0123ABCD`). One-time fetch via `chat.openConversation`. |

All three resolve from `~/.hermes/.env` via the existing scheduler
config loader.

## Registering the cron entry

The cron scheduler stores jobs in `~/.hermes/cron/jobs.json` via the
`cron.jobs.create_job()` API; there is no static `registry.yaml` in
this codebase. Register the sweep once with:

```bash
hermes cron add \
  --name "follow-up-sweep" \
  --schedule "0 9 * * 1" \
  --script "$(pwd)/cron/follow_up_sweep.py" \
  --no-agent
```

This pins the job to `0 9 * * 1` (Monday 09:00). The scheduler honours
`HERMES_TZ` (default `America/Los_Angeles` on Blake's laptop), so the
cron field is local PT — no UTC math required.

## Manual trigger (smoke test)

```bash
LINEAR_API_KEY=$(grep ^LINEAR_API_KEY ~/.hermes/.env | cut -d= -f2-) \
SLACK_BOT_TOKEN=$(grep ^SLACK_BOT_TOKEN ~/.hermes/.env | cut -d= -f2-) \
SLACK_FOLLOW_UP_CHANNEL=$(grep ^SLACK_FOLLOW_UP_CHANNEL ~/.hermes/.env | cut -d= -f2-) \
python -m cron.follow_up_sweep
```

The script prints one JSON line on success::

```
{"count": 19, "very_stale": 0, "summary_ts": "1717000000.000100",
 "thread_ts_count": 19}
```

On failure, exits non-zero and writes the cause to stderr. The cron
scheduler captures both into `~/.hermes/cron/output/<job_id>/<ts>.md`.

## What to do if the sweep goes silent

Two-week silence threshold — if no digest message lands on consecutive
Mondays, treat as a broken cron. Diagnosis order:

1. `hermes cron list` — confirm the job is enabled and `next_run_at`
   is in the next 7 days.
2. `hermes cron history follow-up-sweep` — inspect the last 4 output
   blobs. A non-zero exit with `LINEAR_API_KEY not set` or `Slack API
   error: invalid_auth` points to env / token rotation.
3. Run the manual trigger above. If it succeeds in foreground but the
   cron run failed, the scheduler is not picking up env updates —
   `hermes restart` reloads `~/.hermes/.env`.
4. If the manual run also fails with a Linear error, check Linear's
   status page; if Slack, regenerate the bot token.

## Future work (tracked as follow-ups, not in this phase)

- Reaction-driven triage (`:wontfix:`, `:defer:`, `:schedule:`) — gated
  on Plan 029-F's reaction control surface. When 029-F lands, wire the
  reaction handler to the `digest.thread` ts list captured in the
  scheduler output.
- Edit-in-place vs fresh weekly message — v1 ships fresh-each-Monday;
  edit-in-place behind a feature flag is a candidate `follow-up:028-B`
  row if Blake finds the noise excessive.
