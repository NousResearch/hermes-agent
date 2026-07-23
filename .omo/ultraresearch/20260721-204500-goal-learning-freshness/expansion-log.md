# Expansion log

## Wave 1

- Workers: memory freshness, skill integrity, approval-agent architecture.
- Reused prior ULR journals instead of duplicating their completed goal-wake
  and verified-outcome work.
- Confirmed implementation leads: memory V2 freshness, stale terminal
  handling, original skill provenance on replay, and a portable
  verification-evidence test invocation.
- Deferred leads: full Skills V2 review artifacts/CAS, queue decision ledger,
  cancelled/blocked outcome receipts, and hostile external-writer guarantees.

## Wave 2

- Terminal stale semantics: validated.  Only V2 integrity/freshness failures
  are terminal; valid apply failures keep release-and-retry behavior.
- Provenance replay: the worker failed at model-capacity before delivering;
  parent independently traced the trusted ContextVar and limited replay to two
  accepted origins.
- Verification baseline: validated with an explicit outside-home pytest base
  temp; all 29 outcome-evidence tests passed.

## Wave 3

- Implementation integration and parent readback completed.  The independent
  reviewer attempt failed due model capacity and is recorded in
  `wave-3-parent-review.md`; it is not counted as a PASS.
- All in-scope leads are closed or explicitly deferred with their required
  product/API contract, so research convergence is reached.
