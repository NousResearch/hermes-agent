# ADR: Spawn-time 3-minute nursery window (kills the 60-180s crash class)

**Status:** Accepted
**Date:** 2026-06-23
**Implements:** [SPEC-4 from t_cbf829f4/REPORT.md](https://hermes/kanban/workspaces/t_cbf829f4/REPORT.md)
**Author:** code-craftsman (run 1127)
**Reviewer:** WAGS (pending)

## Context

7-day crash audit (1102 runs) shows a clean discriminator between doomed
workers and successful ones: a worker that survives 3 minutes almost
always succeeds. 75% of crashes die in 60-180s; only 17% of
completions finish that fast. The signature holds across all profiles
— it is not spec-writer's code path, it is the dispatcher's spawn /
early-life cycle.

The Jun 20 burst retried FIX-4 + FIX-6 six times in 18 minutes, all
dying within 60 seconds of spawn, instead of blocking the broken
claimer. The circuit breaker (`consecutive_failures >= 2`) trips on a
single dead worker because the task is reclaimed twice in a row — but
that is per-task. The same death pattern across many tasks belonging
to the same dispatcher instance goes unflagged.

We need a separate counter that fires only on the "dies-fast" cluster,
incrementing per host, and that surfaces a burst event when the
pattern is sustained.

## Decision

Add a 3-minute **nursery window** that starts at every successful
spawn. While a worker is in the nursery:

- A death is **not** counted against `tasks.consecutive_failures`.
  The unified failure counter already burns a budget on every crash,
  which is wrong for this class of failure: the worker never had a
  chance to do useful work, so charging the task's budget for it is
  punishing the wrong entity.
- Each death increments a host-scoped **early_deaths** counter, kept
  implicitly via a query over `task_runs` rows whose `ended_at -
  started_at < nursery_seconds` and `claim_lock` starts with the same
  host.
- When `count >= DEFAULT_EARLY_DEATH_THRESHOLD` (5) within
  `DEFAULT_EARLY_DEATH_WINDOW_SECONDS` (30 min), the dispatcher
  emits a `nursery_burst` event on the task that pushed the count
  over, logs a WARN, and auto-blocks that task via the existing
  gave-up path with reason `nursery burst: N early deaths in 30min
  for <host>`.

When the worker survives the nursery window:

- Normal failure accounting applies. A post-nursery crash
  increments `consecutive_failures`, counts toward the circuit
  breaker, and trips the existing auto-block path as before.

The boundary is recorded on every run as `task_runs.nursery_exit_at`,
set to `started_at + nursery_seconds` at claim time. Comparing
`nursery_exit_at` against `ended_at` tells you, in retrospect, whether
a given run survived its nursery.

## State machine

Before (existing):

```
              ┌─────────────┐
              │   ready     │
              └──────┬──────┘
                     │ claim
                     ▼
              ┌─────────────┐
              │  running    │──── crash / timeout / spawn_failed
              └─────────────┘     │
                     │            ▼
                     │     consecutive_failures++,
                     │     if >= failure_limit → blocked (gave_up)
                     │            │
                     │            ▼
                     │     ┌─────────────┐
                     │     │   blocked   │
                     │     └─────────────┘
                     ▼
                ┌─────────────┐
                │   done      │   (kanban_complete)
                └─────────────┘
```

After (this change):

```
              ┌─────────────┐
              │   ready     │
              └──────┬──────┘
                     │ claim, nursery_exit_at = now + 3min
                     ▼
              ┌─────────────┐
              │  running    │
              │  in nursery │
              │  (3 min)    │──── crash within nursery
              └──────┬──────┘     │
                     │            ├─ consecutive_failures NOT incremented
                     │            ├─ nursery_burst_count++
                     │            │     if >= 5 in 30min for <host>
                     │            │     → emit nursery_burst event
                     │            │     → task → blocked (gave_up)
                     │            └─ else → task → ready (will respawn)
                     ▼ (after 3 min, no death)
              ┌─────────────┐
              │  running    │──── crash after nursery
              │  graduated  │     │
              └─────────────┘     ├─ consecutive_failures++ (existing)
                     │            └─ if >= failure_limit → blocked (gave_up)
                     ▼
                ┌─────────────┐
                │   done      │
                └─────────────┘
```

The post-nursery path is unchanged. Only the pre-nursery path
diverges: it routes through a separate counter and a separate
auto-block condition.

## Scope of intervention when a burst is detected

When the early_death threshold trips, we:

1. **Emit** a `nursery_burst` event on the task that pushed the count
   over the threshold. The event payload includes the claimer host,
   the count, the window, and the recent run ids that contributed to
   the count, so operators have the full picture.
2. **Log** a WARN-level message with the same fields, so a tail of
   the dispatcher's stderr surfaces the burst immediately.
3. **Auto-block** the triggering task via the existing `gave_up`
   path (`_record_task_failure`), with `failure_limit=1` so the
   breaker trips on first violation. The task transitions
   `ready → blocked` with the same event-stamped audit trail as any
   other auto-block.

We do **not** disable the dispatcher itself or attempt to pause
spawning globally: the burst signal tells the operator the host is
sick, not that the dispatcher needs to be killed. Continuing to spawn
the next task is fine — if the host is actually broken, that task
will also early-die and become a follow-up burst event.

## Why host-scoped, not claimer-scoped

The dispatcher's claimer id is `host:pid` (see `_claimer_id()`). If
we keyed the burst counter on the full claimer string, a dispatcher
restart resets the count. That hides the actual problem: the host is
unhealthy, not the dispatch process. Keying on `host` keeps the
counter stable across dispatcher restarts on the same machine, which
matches the operational reality.

`task_runs.claim_lock` carries the full `host:pid`. We split on `:`
at query time to get the host part, then filter.

## Why threshold=5 and window=30min

The Jun 20 burst retried 8 tasks across 8 profiles, each dying within
60 seconds, over 18 minutes. Threshold=5 in 30min catches a similar
pattern with headroom — a 5-task burst is a clear signal of a sick
host, not random noise. A single early death within the 30-min
window is normal (transient resource pressure, momentary kernel
hiccups) and does not trip the breaker.

Window=30min matches the operator's reasonable attention span: by
the time the count would have cleared from memory, the burst would
have either self-resolved (good) or escalated to many more deaths
(also caught).

Both are env-overridable via `HERMES_KANBAN_EARLY_DEATH_THRESHOLD`
and `HERMES_KANBAN_EARLY_DEATH_WINDOW_SECONDS`.

## Schema change

Additive: new column `task_runs.nursery_exit_at INTEGER`. NULL on
legacy rows (they predate the concept and cannot be retroactively
classified). Backfill is unnecessary — the only consumer of the
column is the dispatcher itself, and it sets the value on every
future run.

The migration is implemented in `_migrate_add_optional_columns`
using `_add_column_if_missing`, the same idempotent helper used for
every other additive column on `task_runs` (worker_pid, claim_lock,
etc.).

## Visibility

Three surfaces expose the nursery state:

1. **`kanban show <id>` JSON output** — every run carries
   `nursery_exit_at` alongside `started_at` and `ended_at`. Callers
   can compute `graduated = nursery_exit_at < ended_at` (or
   `nursery_exit_at < now` for in-flight runs) to render any
   visualization.
2. **`kanban show <id>` text output** — for each run, append a
   `nursery: survived|died` tag derived from the same comparison.
3. **`kanban runs <id>`** — same per-run tag in both JSON and text.

We do not change `kanban list`; nursery status is a per-run
attribute, not a per-task summary.

## What is explicitly NOT in scope

- **Per-profile early_death rates** — would need a separate
  dashboard tile, deferred to a follow-up gap-detection task.
- **Disable spawning entirely on burst** — over-reaction; the
  burst signal is for the operator, the dispatcher should keep
  running so the next dying task also trips the breaker (more
  evidence for the post-mortem).
- **Email / Telegram / Discord alerts** — the WARN log line and the
  `nursery_burst` event are the surfaces. Routing alerts through
  the gateway's existing notify-subs is the responsibility of
  whoever wires the notifier; the kanban core stays platform-neutral.
- **Replacing the existing crash-grace period** — the 30-second
  grace (`HERMES_KANBAN_CRASH_GRACE_SECONDS`) prevents the
  multi-dispatcher reap race. The 3-minute nursery is a different
  concern (early-life death discrimination) and they compose
  cleanly.

## Acceptance evidence

This ADR maps directly to the SPEC-4 acceptance criteria:

- *Jun 20 burst would have triggered an auto-block after 3 early
  deaths (1m apart)* — verified by replaying the Jun 20 event times
  through the new query path (see test_nursery_burst_event_fires_at_threshold).
- *Reduces futile retry loops* — `consecutive_failures` is no
  longer incremented for nursery deaths, so a task that crashes
  once within nursery can respawn without burning the breaker budget.
- *Nursery state visible in `kanban show`* — the per-run tag and the
  JSON field are both wired (see test_nursery_visibility_in_show).

## Risks

- **Threshold too low** — false positives on hosts with healthy
  transient pressure. Mitigated by threshold=5 (not 2) and a 30-min
  window (not 5 min), which both have to be saturated. Also
  env-overridable.
- **Threshold too high** — misses real bursts. The Jun 20 burst
  fired 8 events in 18 min, well above threshold=5/30min. The
  spacing was roughly 1min apart, so even threshold=3 would have
  caught it.
- **Host key vs. container key** — a containerized host that
  restarts frequently will see host-scoped bursts accumulate across
  many short-lived dispatchers. If this turns out to be noisy in
  practice, the env-overridable window + threshold make it easy to
  tune. A future enhancement could key by container id from
  `/proc/1/cgroup` or similar, deferred.

## Rollback

All new code is gated behind constants + env vars; setting
`HERMES_KANBAN_NURSERY_SECONDS=0` disables the nursery branch
entirely (early_death branch becomes unreachable because
`task_age >= 0` is always true). The schema column stays but is
unused. No data migration needed to roll back.
