# ADR-001: Trust Write Path Reconciliation

## Status
Accepted — 2026-05-07

## Context
Two write paths currently mutate the trust-ranking signal:

1. `MemoryStore.record_feedback` (`store.py:465`) writes both `trust_score`
   (±0.05/−0.10, clamped) and `helpful_count` (+1 when `helpful=True`).
2. `_reinforce_facts` (`retrieval.py:519`) writes `helpful_count` (+1
   unconditionally) on every fact returned by an HRR probe.

Ranking math in `_reinforced_trust` (`retrieval.py:515`):

    trust_score * (1 + 0.1 * min(helpful_count, 20))

`helpful_count` is the multiplier. Because both paths write it:

- `record_feedback(helpful=True)` moves the ranking signal twice from a
  single user action: once via `trust_score`, again via the multiplier.
- `_reinforce_facts` inflates the multiplier silently on every retrieval —
  passive appearance in results is rewarded the same as explicit user
  judgment.
- `record_feedback(helpful=False)` decrements `trust_score` only;
  `helpful_count` is monotonic upward, so a fact users have repeatedly
  marked unhelpful still accumulates a multiplier from probe hits.

This is the class of bug that does not surface as a test failure but
produces "the system feels off" — facts rank weirdly high because
retrieval frequency is being laundered into the trust signal.

## Decision
`record_feedback` is the sole writer of `helpful_count`. `_reinforce_facts`
no longer writes any trust-related column.

Concretely:

- Remove the `UPDATE facts SET helpful_count = helpful_count + 1 ...`
  block from `_reinforce_facts` in `retrieval.py`.
- Keep the `reinforce_on_retrieval` flag plumbing for now; the function
  becomes a no-op with a docstring noting it is reserved for future
  signals (e.g., recency) but currently writes nothing.
- `record_feedback` continues to write both `trust_score` and
  `helpful_count`. The compounding within a single user action is
  acceptable because it reflects one explicit judgment; the cross-path
  compounding with passive retrieval is what we are removing.
- Ranking math is unchanged.

## Consequences

**Behavior change.** `helpful_count` for any given fact will only grow
when a user explicitly marks it helpful. Existing values are not reset —
they reflect historical reality (mostly retrieval inflation) and will be
diluted by new feedback over time.

**Migration.** None required. No schema change.

**`hermes doctor` verification.** Add a new check `trust_signal_writers`
that asserts the only call site writing `helpful_count` is
`record_feedback`. Implementation: AST grep for `helpful_count` in
`plugins/memory/holographic/*.py`, fail if it appears as an assignment
target outside `store.py:record_feedback`. Roll-up `error` if violated.

**What breaks.** Nothing in production paths. Tests that asserted
`helpful_count` increased after a probe call need to be updated to assert
it did not.

**Reversibility.** Fully reversible by re-adding the write block.
