# Wave 3 — Held-claim reconciliation scope

## Finding

A `.claim` is intentionally hidden from `pending` because it is non-actionable.
Expanding the pending view would make an already-applied mutation appear
replayable. No new diagnostic command is required for this bounded milestone.

## Required recovery response

If a terminal receipt cannot persist after a mutation or terminal no-op, retain
the claim and return the pending ID, exact claim path, receipt-missing state,
and an explicit statement that the decision is final and must not be reapplied.
Add a fault-injection test proving the mutation occurs once and the held claim
remains.

## EXPAND

none — a restart-time claim inventory is a distinct product requirement.
