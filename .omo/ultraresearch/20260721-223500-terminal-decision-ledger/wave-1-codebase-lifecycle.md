# Wave 1 — Pending approval lifecycle

## Findings

- The durable queue lifecycle is staged JSON → atomic hidden claim → apply or
  reject → terminal cleanup, with requeue only for non-terminal apply failure.
  Evidence: `tools/write_approval.py:217,342,375,407` and
  `hermes_cli/write_approval_commands.py:108,163,184`.
- The only valid receipt boundary is a claimed record with known terminal
  outcome: applied success, explicit rejection, or memory's terminal
  freshness/integrity no-op. Staging, failed claim, and requeue must emit none.
- Receipt persistence must happen before terminal claim cleanup. If persistence
  fails after a side effect, the claim must stay non-actionable for manual
  recovery; automatic replay risks duplicate mutation.
- The generic terminal command-approval queue is a separate in-memory system;
  this milestone must not conflate it with durable write-proposal decisions.

## Required invariants

1. One claimant can apply one pending record.
2. A non-terminal failure requeues without an audit receipt.
3. A terminal decision produces exactly one receipt, without replaying payload.
4. No receipt stores replay payload or raw content.

## EXPAND

- LEAD: Verify receipt persistence can be atomically coupled with claim terminalization — WHY: best-effort post-hooks leave duplicate or missing audit windows — ANGLE: SQLite/event ledger and filesystem crash matrix.
- LEAD: Existing verification-evidence persistence may be reusable but needs transaction-boundary proof — WHY: a second ledger should not be created unnecessarily — ANGLE: inspect receipt transactions and crash/retry semantics.
- LEAD: Generic terminal command approvals remain separate scope — WHY: their in-memory FIFO queue needs its own durability and dedupe contract — ANGLE: deferred separate investigation.
