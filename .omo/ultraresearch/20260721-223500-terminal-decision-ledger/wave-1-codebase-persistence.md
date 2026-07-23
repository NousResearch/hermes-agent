# Wave 1 — Receipt persistence and recovery

## Findings

- JSONL is not a sufficient source-of-truth primitive for concurrent approval
  decisions. Hermes already has SQLite/WAL/FULL-sync receipt and execution
  patterns.
- `verification_evidence.db` is profile-scoped, WAL-safe, and already included
  in quick backups (`agent/verification_evidence.py`,
  `hermes_cli/backup.py:788`). Reusing the database avoids a second backup
  integration surface.
- `outcome_receipts` itself has confirmation UPDATE semantics, so a strict
  approval audit record must use a separate immutable table in the same DB (or
  independently prove a new terminal kind cannot enter any mutable path).
- Separate queue filesystem, side-effecting mutation, and SQLite cannot share a
  transaction. Receipt-first terminalization must permit an orphan receipt or a
  held claim for manual recovery; automatic replay is unsafe.

## Existing test evidence

- Delivery-ledger and atomic JSON precedent tests: 53 passed in the worker
  environment.
- The worker observed workspace-root failures in verification-evidence tests
  and Windows symlink privilege failures. Neither is attributed to this
  unimplemented ledger without independent reproduction.

## EXPAND

- LEAD: Prove an immutable separate approval table and its reader boundary — WHY: existing outcome rows have confirmation UPDATE semantics — ANGLE: trace schema/reader/transaction contracts.
- LEAD: Reproduce verification-evidence root isolation failures — WHY: goal-learning eligibility is a prerequisite baseline — ANGLE: focused test execution and project-root cache inspection.
- LEAD: Evaluate read-only health diagnostics for the evidence database — WHY: a new audit table shares a DB not covered by doctor — ANGLE: compare doctor SQLite practices and cost.
