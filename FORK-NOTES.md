# Fork Notes — Rian23-R4/hermes-agent

This fork carries focused, individually revertable patches on top of
`NousResearch/hermes-agent`. Everything here is upstream-safe: default behavior
is unchanged unless the related opt-in configuration is explicitly set.

## 1. Idle compaction measured from durable session recency

Upstream issue: [NousResearch/hermes-agent#79357](https://github.com/NousResearch/hermes-agent/issues/79357)
— `compression.idle_compact_after_seconds` never fires because every turn entry
re-stamps `_last_activity_ts` (the liveness watchdog's clock) before the idle
check reads it.

### What this patch does

- Adds `_durable_last_activity_at`, a dedicated field seeded at agent
  construction from the session row's durable recency
  (`store.sessions.last_active()` — the store's own rule: freshest of
  `last_activity_at` and the newest stored message timestamp), falling back to
  the deprecated `SessionDB.get_session()` for older stores.
- `DurableTurnLease.start()` snapshots `_pending_idle_gap_seconds` BEFORE
  `_touch_activity("starting new turn")`, from the older of the durable stamp
  and the in-memory clock.
- `_idle_compaction()` consumes-and-clears that snapshot BEFORE its trigger
  guard, so a skipped idle compaction never leaks a stale gap into a later
  turn.
- `_last_activity_ts` is deliberately NEVER backdated: it feeds the gateway
  liveness watchdog, and an old stamp there makes the gateway abandon the first
  turn of a resumed session as if it had stalled.

### Configuration key

`compression.idle_compact_after_seconds` (default `0` = disabled), exactly as
upstream defines it in `hermes_cli/config_defaults.py` and the configuration
docs. No new key is introduced; the fix only makes the existing opt-in trigger
fire on resumed sessions.

### Test commands and results (run on this branch)

```
# targeted idle + docs
pytest tests/agent/test_idle_compaction_seeding.py \
       tests/agent/test_idle_compaction_lock_and_guards.py \
       tests/agent/test_idle_compaction.py \
       tests/agent/test_turn_facade_lease.py \
       tests/agent/test_session_activity.py \
       tests/ci/test_docs_drift.py -q
# -> 42 passed

# compression / turn-context regressions
pytest tests/agent/test_compression_concurrent_fork.py \
       tests/agent/test_413_compression.py \
       tests/agent/test_turn_context_compaction.py \
       tests/agent/test_cross_process_turn_lease.py -q
# -> 102 passed

ruff check agent/agent_init.py agent/turn_facade_lease.py \
          agent/turn_context_compaction.py \
          tests/agent/test_idle_compaction_seeding.py
# -> All checks passed!
```

Red proof: with the patch stashed, 7 of the 9 tests in
`tests/agent/test_idle_compaction_seeding.py` fail.

### Upstream behavior

Unchanged unless `compression.idle_compact_after_seconds` is set above `0`:
the durable-recency path is only consulted by the idle-compaction trigger, the
watchdog keeps its fresh clock, and every other consumer of
`_last_activity_ts` is untouched.

## 2. Relationship to other open upstream PRs for #79357

- #79456 / #79372 stash the pre-reset timestamp inside
  `_init_cached_agent_for_turn`, which covers only the cached-agent path.
- #135759 measures the gap from the transcript's own timestamps.
- This patch reads the store's durable recency
  (`sessions.last_active()`), which survives the lease reset, gateway cache
  eviction, and process restarts, without touching the watchdog clock.
