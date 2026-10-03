# Decision Model

The relationship classification is about behavior, not names.

## CONFIRMED DUPLICATE

Use only when the selected skills have materially equivalent:

- activation intent,
- core workflow,
- outputs,
- authority,
- safety boundaries,
- platform behavior.

Preferred action: one canonical skill, preserving the strongest version of every contract.

## LIKELY REDUNDANT

Use when one skill appears superseded by another but usage, dependency, or behavior evidence is incomplete.

Preferred action: do not delete. Clarify triggers, gather missing evidence, or mark for later deprecation review.

## PARTIAL OVERLAP

Use when meaningful workflow or reference material overlaps while distinct responsibilities remain.

Preferred action: extract shared references or tighten scope rather than merge blindly.

## COMPLEMENTARY

Use when skills operate in the same domain but solve different lifecycle phases or have different authority.

Preferred action: keep separate, cross-reference, or add an orchestrator.

## PARENT OR ORCHESTRATOR

Use when one skill should route to focused child skills rather than absorb their implementation.

Preferred action: preserve child safety boundaries and keep orchestration thin.

## SHARED REFERENCE CANDIDATE

Use when procedures, definitions, platform setup, or verification guidance are repeated but user-facing trigger responsibilities remain distinct.

Preferred action: extract reusable reference material without merging the skill contracts.

## INTENTIONALLY SEPARATE

Use when merging would blur:

- read-only versus mutating behavior,
- normal operation versus destructive recovery,
- different credential scopes,
- different platform guarantees,
- materially different counter-triggers,
- separate approval boundaries.

Preferred action: keep separate. Shared references are allowed only when they do not weaken boundaries.

## INSUFFICIENT EVIDENCE

Use when the requested relationship cannot be supported from verified content.

Preferred action: no mutation.

## Canonical selection

When a canonical skill is needed, rank evidence in this order:

1. stronger and clearer safety contract,
2. more precise positive and negative triggers,
3. broader verified dependency adoption,
4. more complete behavior tests,
5. current authoritative source or explicit supersession evidence,
6. clearer maintained version history.

Do not choose canonical status from name length, creation date, or apparent popularity alone.

## Oversized umbrella rule

A skill that has absorbed many workflows may need splitting when:

- counter-triggers become hard to express,
- unrelated tools or authority are always loaded together,
- safety boundaries differ within the same skill,
- trigger precision degrades,
- verification requires unrelated test families.

Splitting is consolidation work when it restores clearer boundaries.
