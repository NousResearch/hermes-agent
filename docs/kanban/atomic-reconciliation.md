# Atomic Kanban reconciliation protocol

`hermes kanban --board <slug> reconcile --input -` is the automation-only boundary for synchronizing an authoritative external source into Hermes Kanban without dispatcher races.

## Transport

- Input is one UTF-8, canonical JSON object terminated by `\n` (sorted keys, no insignificant whitespace), maximum 1 MiB.
- `--input -` reads stdin; a file path is also accepted.
- Output is exactly one canonical JSON object, including delegated-child denial
  (`permission-denied`).
- Invalid input exits `2`; internal failure exits `1`. Error output is bounded and never reflects paths, bodies, credentials, or exception strings.

## Schema version 1

Every request contains:

- `schema_version`: `1`
- `operation`: `create-if-absent`, `replace-unclaimed`, or `cancel-unclaimed`
- `canonical_ref`: lowercase `owner/repo#number`
- `idempotency_key`: `sha256:<64 lowercase hex>` for this source event
- `idempotency_lineage`: stable `sha256:<64 lowercase hex>` for the source item
- `source_revision`: lexicographically monotonic UTC timestamp (`YYYY-MM-DDTHH:MM:SSZ`)

Create and replace include `task`; replace and cancel include `expected`. The strict field schema is enforced in `kanban_db.reconcile_task` rather than only at the CLI.

`expected` carries the projected task id, its unclaimed status (`triage`, `todo`,
or `ready`), previous source revision, lineage, and idempotency key, plus null
`claim_lock` and `run_id` expectations.

## Atomic behavior

Each operation runs under the board database's `BEGIN IMMEDIATE` writer transaction:

- **create-if-absent** creates at most one active task for a lineage.
- **replace-unclaimed** archives the expected unclaimed task and creates its replacement in the same transaction.
- **cancel-unclaimed** archives the expected unclaimed task and marks the lineage inactive in the same transaction.
- A dispatcher claim that wins first produces `claimed`; reconciliation never overwrites its work.
- If reconciliation wins first, a later claim of the superseded task cannot succeed.
- A failed replacement creation rolls back the archive, lineage update, audit event, and replay ledger together.

The current lineage state and an append-only request replay ledger are durable. Replaying an identical request key returns the exact prior bounded result. Reusing a key with different input returns `conflict`. Older source revisions return `stale-source-revision` without mutation.

## Redaction boundary

Task title/body live only in the ordinary task row. Reconciliation state stores lineage, task id, source revision, and active state. The replay ledger stores only a SHA-256 request digest and bounded result. Reconciliation audit events contain only operation, hashes, revision, and task ids.
