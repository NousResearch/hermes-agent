# Wave 2 — Immutable receipt schema

## Decision

Use a separate `approval_decision_receipts` table in the existing
profile-scoped `verification_evidence.db`, not a new terminal kind in mutable
`outcome_receipts`.

## Required contract

- Immutable `INSERT` only; database triggers reject UPDATE and DELETE.
- A unique `(subsystem, pending_id, proposal_digest)` key makes retries
  idempotent but treats a mismatched decision/outcome as a conflict.
- Store only receipt metadata and a proposal digest: no payload, summary, or
  raw content.
- Valid terminal states: approved/applied, rejected/rejected, and
  approved/apply-terminal-failed with a safe failure code.
- Reject writes the receipt before claim cleanup. Approval first applies; only
  successful or terminal non-retry outcome writes the receipt; then cleanup.
  A receipt failure holds the claim for manual recovery, never automatic replay.

## Why

`outcome_receipts` is updated during explicit goal confirmation. Combining it
with approval decisions would let a mutable learning-candidate lifecycle alter
an audit record. The shared database already has WAL/busy-timeout and
WAL-safe backup support.

## EXPAND

- LEAD: terminal-claim manual reconciliation has no API — WHY: a post-apply receipt failure must remain non-replayable but observable — ANGLE: bounded read-only reconciliation view.
- LEAD: shared approval handler has no explicit session identity — WHY: an audit row needs honest actor/session semantics — ANGLE: use an explicit stable synthetic approval-session scope or trace caller plumbing.
- LEAD: document outcome receipt mutability separately from approval receipt immutability — WHY: prevents future semantic conflation — ANGLE: update design document with tests.
