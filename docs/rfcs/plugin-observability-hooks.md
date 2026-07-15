# Plugin Observability Hooks — Design Proposal

**Branch:** `feat/plugin-observability-hooks`
**Target:** `hermes_cli/plugins.py` (VALID_HOOKS), `hermes_cli/kanban_db.py`, `gateway/run.py`
**Concrete consumer:** kanban-advanced (board_keeper, intervention tracking, postmortem data)

---

## Summary

Today plugins that need to observe kanban worker lifecycle (spawn, crash, stale
claim) or track manual card mutations must poll the board via cron. This adds
latency (1-minute cron ticks) and burns tokens on polling queries.

This proposal adds four observer hooks that let plugins react to events
immediately instead of polling.

---

## Proposal 1: Worker lifecycle hooks

### `kanban_task_claimed` (pre-spawn observer)

Fires in the **DISPATCHER** process BEFORE `_default_spawn` at
`kanban_db.py:3485`. Carries `task_id`, `assignee`, `board`.
Observer-only — callbacks run before the worker subprocess is created
and cannot affect the spawn decision.

### `kanban_task_spawned` (post-spawn observer)

Fires in the **DISPATCHER** process AFTER `_default_spawn` returns
successfully and the worker PID is persisted in the tasks table at
`kanban_db.py:8058-8069`. Carries `task_id`, `board`, `assignee`,
`run_id`, `profile_name`, `worker_pid: int`. Use this for "worker is
actually running" events.

### `kanban_worker_exited` (tick-derived observer)

Fires in the **DISPATCHER** process from `detect_crashed_workers`
(`kanban_db.py:6641`) when a worker PID is discovered dead. Carries
`task_id`, `board`, `assignee`, `run_id`, `profile_name`,
`exit_kind` (`clean_exit` | `nonzero_exit` | `signaled`), `exit_code`.
Rate-limited exits emit a separate `kanban_worker_rate_limited` event.

### `kanban_worker_stale_claim` (tick-derived observer)

Fires in the **DISPATCHER** process when `release_stale_claims` reclaims a
timed-out claim.

```python
# Kwargs: task_id, board, assignee, run_id, profile_name,
#         seconds_stale: int
```

### Concrete use case (kanban-advanced)

Our `board_keeper` cron polls the board every minute to detect:
- Stale `running` cards → trigger salvage
- Crashed workers → trigger re-dispatch
- Worker exit → trigger auto_unblock

With these hooks, `board_keeper` becomes event-driven — zero polling latency,
zero token burn on "nothing changed" checks.

---

## Proposal 2: `kanban_task_updated` hook

Fires after any UPDATE to a task row (body edit, assignee change, title
change, description update). Observer-only — return values ignored.

```python
# Kwargs: task_id, board, assignee, profile_name,
#         changed_fields: list[str]  # e.g. ["body", "assignee"]
```

### Concrete use case (kanban-advanced)

Our intervention tracking requires operators to manually run
`kanban_intervention_inc.sh` after editing a card. With this hook, the
intervention counter increments automatically — no manual step, no forgotten
increments.

---

## Implementation notes

All hooks follow the same pattern as existing kanban lifecycle hooks:
- Fire AFTER the write txn commits (observer safety)
- Swallowed exceptions (a misbehaving plugin can't break board state)
- Standard kwargs: `task_id`, `board`, `assignee`, `run_id`, `profile_name`

### Mutation boundary

`kanban_task_updated` must fire for every write path that mutates task
state, regardless of which module performs the write:

**kanban_db.py paths:**
| Function | Changed fields |
|----------|---------------|
| `create_task` | All fields (initial) |
| `complete_task` | status→done, summary, result, metadata |
| `block_task` | status→blocked, last_failure_error |
| `unblock_task` | status→ready/todo, block_recurrences |
| `_set_status_direct` | status |
| `recompute_ready` | status→ready (promotion) |
| `_record_task_failure` | status→blocked, consecutive_failures |

**Dashboard plugin_api.py paths:**
| Endpoint | Changed fields |
|----------|---------------|
| `PATCH /tasks/:id` (L838) | priority, title, body |
| Bulk update (L1254) | priority |

**Payload contract:** `changed_fields` is a list of field names that were
mutated, not just a generic "updated" signal. Consumers can filter on
specific fields without diffing DB state.

### Timing contract

| Hook | Fires | Latency |
|------|-------|---------|
| `kanban_task_claimed` | Pre-spawn, before `_default_spawn` | Immediate |
| `kanban_task_spawned` | Post-spawn, after `_default_spawn` + PID persistence | Immediate |
| `kanban_worker_exited` | Tick-derived via `detect_crashed_workers` | ≤ dispatcher tick interval |
| `kanban_worker_rate_limited` | Tick-derived, separate event kind | ≤ dispatcher tick interval |
| `kanban_worker_stale_claim` | Tick-derived via `release_stale_claims` | ≤ dispatcher tick interval |
| `kanban_task_updated` | Post-commit, after any task mutation write | Immediate |

Exit events are NOT immediate — `_default_spawn` is fire-and-forget.
Workers are discovered dead on the next dispatcher tick. Plugins
requiring lower latency should combine hook observation with
`kanban_heartbeat` liveness tracking.

---

## Related

- PR #58541 — kanban lifecycle hooks (pre_complete, unblocked, created)
- PR #58542 — plugin config & state bridge (cron API would obsolete polling)
- kanban-advanced board_keeper architecture
