# Cross-Profile Execution Provenance

> **Audience:** Gateway operators and contributors
> **Source files:** `hermes_cli/execution_provenance.py`, `hermes_cli/main.py`, `hermes_cli/kanban_db.py`, `gateway/slash_commands.py`

## Purpose

Hermes profiles isolate memory, configuration, and runtime state. A noninteractive agent process that selects a different profile can otherwise look like an ordinary human `hermes -p <profile>` invocation. The profile entry-point gate distinguishes those paths, rejects cross-profile agent work without bounded authority, and writes accepted launches to one shared provenance ledger.

This is a **same-user policy boundary**, not hostile-process isolation. A process running as the same operating-system user can alter its environment and files. The gate prevents untracked Hermes launch paths and provides auditable provenance; it does not provide a cryptographic trust boundary.

## Behavior

- Same-profile commands remain unchanged.
- Read-only administrative inspection, such as `gateway status`, remains available across profiles.
- Human profile selection remains available when stdin is an interactive terminal and no noninteractive prompt flag is present.
- Noninteractive cross-profile agent execution fails closed unless it has either:
  - live matching Kanban task/run/claim custody; or
  - a bounded, expiring, one-shot direct-authority record.
- Prompt bodies and conventionally sensitive command-line values are redacted before ledger storage.
- One-shot execution IDs are atomically consumed in durable marker files before the ledger row is appended. Replays are rejected even if a later ledger write fails.

Kanban source labels are audit context only. Authorization is established by read-only verification of the live canonical Kanban database: assignee, active run, claim lock, claim expiry, status, and worker PID compatibility must agree.

## Internal execution contract

The launcher and Kanban dispatcher pass authority and custody fields to the child process through internal environment variables. These variables are an execution transport contract, **not user configuration** and must not be placed in `.env` or documented as operator-set feature flags.

The normal operator interface remains `hermes -p <profile> ...`. Behavioral configuration remains in `config.yaml`.

## Ledger and visibility

The ledger is stored at:

```text
<canonical Hermes root>/execution-provenance.jsonl
```

Named profile homes therefore fan into the same root ledger. Accepted records include source, authority class/reference, target, redacted execution path, Kanban tracking, scope, one-shot/expiry fields, evidence, terminal condition, PID, start time, and observed state.

Recent bounded records appear in gateway `/status` output. Status reads only a bounded tail of the ledger, ignores oversized rows, sanitizes control characters, and applies per-field, per-line, record-count, and total-size bounds so malformed or oversized ledger values cannot break platform status delivery.

The ledger is append-only in this implementation. There is no automatic retention or migration policy; operators may archive it while no launch is being recorded. Malformed JSONL rows are ignored by status readers and cause new authority consumption to fail closed until the audit history is repaired.

## Failure behavior

The gate exits with code `77` when authority is missing, malformed, expired, mismatched, replayed, or not backed by live Kanban custody. It also fails closed when the custody database or cross-process locking facility is unavailable.
