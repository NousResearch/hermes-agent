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

### `kanban_worker_spawned`

Fires in the **DISPATCHER** process after a worker subprocess is successfully
spawned.

```python
# Kwargs: task_id, board, assignee, run_id, profile_name, worker_pid: int
```

### `kanban_worker_exited`

Fires in the **DISPATCHER** process when a worker subprocess exits (whether
success, failure, or crash).

```python
# Kwargs: task_id, board, assignee, run_id, profile_name,
#         exit_code: int, outcome: str ("completed" | "failed" | "crashed")
```

### `kanban_worker_stale_claim`

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

All four hooks follow the same pattern as existing kanban lifecycle hooks:
- Fire AFTER the write txn commits (observer safety)
- Swallowed exceptions (a misbehaving plugin can't break board state)
- Standard kwargs: `task_id`, `board`, `assignee`, `run_id`, `profile_name`

---

## Related

- PR #58541 — kanban lifecycle hooks (pre_complete, unblocked, created)
- PR #58542 — plugin config & state bridge (cron API would obsolete polling)
- kanban-advanced board_keeper architecture
