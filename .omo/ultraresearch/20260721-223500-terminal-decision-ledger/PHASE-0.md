# Phase 0 — Terminal decision receipts and goal-learning handoff

## Core question

What is the smallest Hermes extension that records an append-only, durable
receipt for each terminal write-approval decision, preserves existing replay
and ownership semantics, and exposes verified receipts as trustworthy input to
the next goal-learning cycle?

## Axes

1. **Approval lifecycle boundary** — trace claim, apply, reject, release, and
   terminal no-op paths to identify the single correct receipt boundary and
   prevent duplicate receipts.
2. **Durable local persistence** — inspect current atomic JSON/JSONL writers,
   recovery conventions, backup/doctor behavior, and Windows-safe path rules
   for an append-only receipt ledger.
3. **Goal-learning integration** — map current outcome/learning evidence
   filters and determine whether receipts can become provenance-only evidence
   without turning an approval into an unverified learning result.
4. **External agent-systems grounding** — check primary/official guidance for
   auditable approvals, provenance, and long-horizon agent evaluation; apply it
   only where it is compatible with Hermes's existing contracts.

## Scope and success criteria

Codebase relevant: yes. External: yes. Browsing: yes. Verification likely:
yes. Report: tracked Markdown research journal plus design/test evidence.

Success requires a dedicated first-wave researcher for every axis, at least two
expansion waves or all leads explicitly closed, runtime tests for any contested
behavior, one focused implementation branch, and a clean merge to local main.
