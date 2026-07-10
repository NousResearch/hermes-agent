# Routing-isolation reconciliation — 2026-07-10

## Result

**Confirmed and fixed in this audit worktree:** Kanban completion wake-ups could re-enter a different conversation. The notifier persisted a target `(platform, chat_id, thread_id, user_id)` but discarded `chat_type`, then recreated every internal wake with `chat_type="group"`. DM tasks therefore created a fresh group-shaped session instead of returning to their originating DM.

Fix commit: `7221727726 fix: preserve kanban notify session chat type`

No production gateway, cron configuration, Kanban board, or live database was modified.

## Evidence

### Reproduction (RED)

A new end-to-end test created a DM-owned task, registered an intended subscription with `chat_type="dm"`, completed the task, and ran the real notifier watcher.

Before the implementation, canonical runner output was:

```
TypeError: add_notify_sub() got an unexpected keyword argument 'chat_type'
```

The missing API/schema field was the first observable failure in the reproducible path.

### Fix design

1. Added `HERMES_SESSION_CHAT_TYPE` to the ContextVar identity map, gateway binding, subprocess bridge, and test reset list.
2. Added nullable `chat_type` to `kanban_notify_subs` in both fresh schema and additive migration.
3. Captured it at the two gateway-originated subscription paths:
   - tool `kanban_create` auto-subscribe;
   - gateway `/kanban create` auto-subscribe.
4. Reconstructed the notifier wake with the persisted `chat_type`.
5. For old rows where that value is unknowable, deliver the status notification but **skip the internal agent wake**. This is fail-closed: no fresh/wrong session is created.

### Verification (GREEN)

Canonical runner, targeted end-to-end regression:

```
scripts/run_tests.sh tests/hermes_cli/test_kanban_notify.py \
  -k notifier_wakes_dm_origin_in_same_dm_session -q
```

Result: `1 tests passed, 0 failed`.

Focused group:

```
scripts/run_tests.sh \
  tests/hermes_cli/test_kanban_notify.py \
  tests/tools/test_kanban_tools.py \
  tests/gateway/test_session_env.py -q
```

Result: **132 passed, 0 failed**.

### Repaired notifier-artifact fixture

The original missing-artifact test timed out both at detached base `0da22bf07d` and audit HEAD. An isolated diagnostic established why: the production completion tool correctly rejects nonexistent artifact paths before it creates a completion event, while the test purported to exercise the *notifier* handling a completion event containing a stale path.

The fixture now writes that legacy/external completion event directly, with one real and one missing artifact. This isolates the notifier contract and is covered by the 132-pass canonical run above. The fixture repair is committed separately as `47e11de55d`.

## Other reported paths

### Detached delegation

`tools/async_delegation.py` snapshots a `session_key` and queues the completion. Gateway consumption is guarded by the existing regression `tests/gateway/test_internal_event_never_interrupts_busy_session.py`, which queues an internal event behind an active user turn. No alternative-session reproduction was found in this audit.

### Cron

Cron target resolution is deterministic. `deliver=origin` intentionally falls back to the configured home channel if an origin target is absent. That behavior can be perceived as a wrong destination, but it is an explicit policy fallback rather than current-session leakage. Changing it to fail-closed would be a separate product decision and needs its own test/rollout.

### Compression continuation routing (new confirmed recurrence)

A live Discord session that ended with `end_reason="compression"` had a complete peer tuple on its parent row (`session_key`, `user_id`, `chat_id`, `chat_type`, and `thread_id`). Its continuation child was stored as `source="discord"` but all five peer fields were `NULL`.

`agent/conversation_compression.py` retained this tuple on the in-memory agent but did not pass it to `SessionDB.create_session()` when creating the rotation child. A deferred event arriving after compression therefore had no authoritative child destination and could be routed via a fallback rather than the original Discord thread.

The rotation now forwards the exact retained peer tuple to the child row. The new behavioral regression creates a real Discord-thread parent, forces the real rotation path, and requires the persisted child tuple to match exactly.

Canonical verification after the fix:

```
scripts/run_tests.sh tests/agent/test_compression*.py \
  tests/gateway/test_compression*.py \
  tests/run_agent/test_compression*.py -q
```

Result: **90 passed, 0 failed**.

## External audit receipt

The independent Claude Max and GLM 5.2 audit runners both exhausted their bounded 12-turn source-exploration budget without emitting findings. Their receipts and failure handling are recorded in `AUDIT_ATTEMPT_RECEIPT.md`; they were not used as evidence.

## Rollout status

**Deployed 2026-07-10 14:11 PDT.** Live checkout `lfdm-v018` is at `925dfaf1ff`; `hermes-gateway` was restarted and is active as process `3374049`.

The additive `chat_type` migration was applied via the repository's idempotent `init_db()` path to the default board and all 52 named boards (53 live databases total). Consistent SQLite online backups are retained at:

```
/home/lfdm/.hermes/backups/kanban-chat-type-pre-migration-20260710T141314
```

All 53 live `kanban_notify_subs` tables now contain `chat_type`. The 15 existing legacy subscriptions have no authoritative historical type, so the notifier sends their direct status notification but deliberately skips their internal agent wake; they can populate `chat_type` only when recreated or through a new authoritative subscription event. A read-only production audit also found no active compression children missing a routable peer tuple.
