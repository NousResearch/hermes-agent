# Phase 3 Structured Task Registry Notes

## SUMMARY

Phase 3 adds a purpose-fit focused task registry substrate for Hermes' concierge/front-desk direction. It gives Hermes a small, explicit place to record user task identity, status, origin metadata, pending follow-ups, artifacts, notes, and future worker linkage fields without yet changing routing behavior.

## PURPOSE-FIT DESIGN RATIONALE

The registry is intentionally small and boring:

- `FocusedTask` records the task Hermes is accountable for.
- `TaskOrigin` records where the task came from.
- `TaskRegistry` creates, lists, updates, cancels, serializes, and optionally persists tasks.
- Pending follow-ups use Phase 2 `PendingTurnItem`, preserving order and boundaries.
- `active_worker_id` and `worker_kind` are recorded for Phase 4 but do not start workers.
- JSON persistence is optional and atomic; SQLite/durable multi-writer behavior is deferred.

This supports the final concierge goal by letting the orchestrator remember “which user goal is active” and attach late refinements without prematurely deciding routing semantics.

## WHAT YOU INTENTIONALLY DID NOT BUILD

Not included in Phase 3:

- Ralph/focused-agent runtime.
- Follow-up classifier.
- Automatic Telegram/CLI/TUI routing into tasks.
- Background or detached worker lanes.
- `delegate_task(background=True)`.
- Gateway delivery or notification changes.
- `/stop <task>` behavior.
- Broad SQLite migration.

## RALPH / FUTURE FOCUSED-AGENT NOTES

Future phases can use this registry as the control plane:

- Phase 4 worker lanes can create/claim tasks and set `active_worker_id`.
- Phase 5 follow-up routing can attach `PendingTurnItem` objects as corrections, appends, or status/cancel intents.
- Phase 6 synthesis can compare worker output against `user_goal`, pending follow-ups, artifacts, and notes before delivering to Woo.

The registry deliberately stores worker linkage fields now so those phases do not need to redefine task identity later.

## TEST NOTES

Target tests should cover:

- create/list/session filtering
- status transitions and invalid status rejection
- worker linkage recording only
- follow-up order preservation
- raw passthrough exclusion from serialization
- artifact/note storage
- active filtering
- optional JSON persistence
