# Proposal: Isolate cron agent execution

## Why

The cron scheduler currently dispatches agent runs into in-process worker threads. A non-cooperative run cannot be terminated, so timeout cleanup can retain scheduler ownership and process-global `TERMINAL_CWD` state indefinitely.

## What

Move agent-backed cron execution into a killable child process group supervised by the parent scheduler. The parent owns dispatch deduplication, execution-ledger transitions, runtime state, timeout escalation, reaping, and delivery. Add a startup handshake so shutdown cannot miss a child between spawn and registration, preserve bounded reader/writer wakeup behavior, and expose runtime state in cron status.

## Architectural anchor

This change serves ADR 0006 (`/home/linh/hpladrs/0006-spec-driven-development-via-openspec.md`) and does not alter the accepted ADR layer.
