# Wave 1 — Goal-learning evidence boundary

## Findings

- A goal is marked `done` before a `judge_done_unconfirmed` outcome receipt is
  recorded (`hermes_cli/goals.py:1736-1743`). Reusable learning evidence still
  requires explicit user confirmation and current passing verification
  (`agent/verification_evidence.py:761-784`).
- Pending memory/skill claims retain proposal integrity and provenance but are
  deleted after a terminal decision; no independent decision history remains
  (`tools/write_approval.py:191,342,407`).
- Existing generic terminal approval has observer hooks, but the bounded
  milestone remains pending memory/skill decisions; expanding to all terminal
  command approvals would require separately validating every producer.
- The worker found a possible baseline issue: two outcome-receipt tests became
  `unverified` after recording a terminal result. This is unverified until an
  independent focused test run isolates the cause.

## EXPAND

- LEAD: Verify the two failing outcome-receipt baseline tests — WHY: approval provenance must not be layered onto an invalid learning-evidence baseline — ANGLE: execute the tests and inspect root/session resolution.
- LEAD: Generic terminal approval observer hooks could support a later separate authorization ledger — WHY: common correlation metadata already exists — ANGLE: trace all post-approval producers and preserve timeout/smart semantics.
- LEAD: Pending claims need a durable terminal-decision record — WHY: their current cleanup removes human-decision provenance — ANGLE: lifecycle and persistence workers.
