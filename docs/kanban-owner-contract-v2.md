# Kanban owner-read contract 2

Owner snapshot and event routes remain bearer-authenticated, server-profile/board scoped, bounded and transactionally cursor-consistent. Version 2 is a coordinated producer/consumer change; do not advertise these payloads as v1.

Snapshot envelope: `contract_version` (integer 2), `profile_id`, `board_id`, `complete`, `task_count`, `event_cursor`, `receipts`, `retired_task_ids`. Receipt keys: `task`, `created_event`, `archived`. Task keys: `id`, `title`, `created_by`, `created_at`. Creation-event keys: `id`, `kind`, `payload`, `created_at`.

Snapshot receipts include durable hard-deletion tombstones, counted against the requested task limit. They preserve deleted-parent creation ancestry. Their `archived` flag is true (not active); restored extant tasks have false. `retired_task_ids` separately records archive/deletion evidence, including original task_events archive history, so a consumer can retire links even when an archive and restoration occurred between snapshots. Consumers must not equate restoration of a native row with recreation of external ownership.

Event envelope: `contract_version`, `profile_id`, `board_id`, `after`, `next_cursor`, `has_more`, `events`. Each event contains `id`, `kind`, `payload`, `created_at`, `task`, `archived`. Kinds: created, updated, archived, deleted. The archive flag is the native row's state at emission; deleted always removes it from active execution. An updated event may restore archive state to false. Non-creation payloads are null. Cursor numbering belongs to durable owner_events, not task_events; an ahead cursor is rejected, never silently reset.

Creation evidence fails closed. Only `by` and `from_decompose_of` are exposed; supplied allowlisted identifiers are validated without trimming. A valid object with no allowlisted keys remains `{}`. Invalid/missing evidence remains unknown and causes HTTP409, not a fabricated empty payload. Missing creator identity and ambiguous original creation receipts cannot be called complete provenance. Duplicate originals in initial backfill produce invalid evidence instead of silently selecting the first.

Already migrated lossy nulls are not repaired by endpoint reads or by this version bump. An administrator must separately prove them from exact original history on backed-up copies and execute an approved finite CAS repair. Existing consumer caches with task_events-numbered cursors likewise need an independently reviewed, atomic one-time replacement. Keep ordinary backward-cursor rejection and authentication intact.

Regression gates: `tests/plugins/test_kanban_owner_lifecycle.py`, `test_kanban_owner_reconciliation.py`, `test_kanban_dashboard_plugin.py`, plus the coordinated consumer's real mounted-response tests. Passing producer-only tests is not deployment acceptance.
