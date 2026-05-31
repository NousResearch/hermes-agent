# Hermes Lane Reliability — Issue-to-Merge Plan

## Goal
Guarantee deterministic, reliable task handoffs from GitHub issue intake through to merge without ghost states or lost routing.

## Audit Findings

### 1. Source-card and action-task routing invariants
- `claim_task` already checks parent dependencies before transitioning ready→running
- `dispatch_once` already handles ready→running with proper CAS
- `recompute_ready` promotes tasks when all parents are done
- **Fix needed**: Ensure `recompute_ready` also handles the case where a task's parent was reclaimed (status changed back to ready) and demotes the child accordingly

### 2. Manual-claim ghost running states
- `claim_task` already closes stale runs before claiming
- `reclaim_task` handles operator-driven reclaim
- **Fix needed**: Add explicit check in `claim_task` for the case where `current_run_id` points to a run that has `ended_at` set but the task is still in `ready` status (invariant violation from unknown code path)

### 3. Every claimed task results in spawn or auto-reclaim
- `dispatch_once` handles spawn with `spawn_fn`
- Auto-block circuit breaker prevents thrashing
- **Fix needed**: Verify that `dispatch_once` properly handles the case where a task is claimed but spawn_fn raises — the task should be re-queued to ready, not left in a limbo state

### 4. Visible GitHub replies on intake and status transitions
- `_kanban_notifier_watcher` polls `kanban_notify_subs` and delivers events
- TERMINAL_KINDS = (completed, blocked, gave_up, crashed, timed_out)
- **Fix needed**: Ensure that all status transitions (not just terminal ones) post visible replies. Currently, only terminal events are delivered. Need to add delivery for intermediate transitions (todo→ready, ready→running, running→blocked, etc.)

### 5. End-to-end handoff determinism
- The complete flow: triage → todo → ready → running → done
- **Fix needed**: Verify that the `specify_triage_task` function correctly promotes tasks and that the dispatcher's next tick properly promotes them to ready

## Implementation Steps

1. **Fix recompute_ready to handle reclaimed parents** — Add demotion logic when a parent's status changes from done to ready
2. **Fix claim_task ghost state handling** — Add explicit check for stale current_run_id
3. **Fix dispatch_once spawn failure handling** — Ensure tasks are re-queued to ready on spawn failure
4. **Enhance GitHub reply delivery** — Add intermediate status transitions to the notifier
5. **Verify end-to-end flow** — Run the existing tests and add any missing ones

## Testing
- Run existing kanban tests
- Run the reclaim race test
- Add tests for the new fixes
