# Canonical Supersession Resolver — pattern (content-addressed evidence systems)

Class: receipt-based evidence/proof-state systems where a newer positive receipt must explicitly
supersede older negative receipts WITHOUT deleting or rewriting immutable ledger history.
Established during `CLOE_LIVING_CAPABILITY_EVIDENCE_SYSTEM_V1_EXPLICIT_SUPERSESSION_POLICY_REPAIR_V1`
(PR #677, merge `6cda879d`, openclaw-runtime).

## Core semantics (do not relitigate)

- Supersession changes ACTIVE status only; it never deletes/rewrites/hides a receipt.
- A validly superseded receipt is excluded from ACTIVE classification for its
  (capability, dimension) but remains fully visible in history and provenance.
- An invalid supersession edge NEVER hides its target (target stays active).
- A later VERIFIED receipt WITHOUT `supersedes_receipt_ids` still conflicts (no implicit supersession).
- Forbidden: last-wins, first-wins, arbitrary authority precedence, UNKNOWN silently superseding VERIFIED.

## Resolver design (deterministic)

- Module: `src/self-audit/evidence/supersession-resolver.mjs` (pure; no fs/network).
- Input: `receipts` (dimension-filtered), `allReceipts`, `acceptedReceiptIds`, SHAs, environment,
  allowedCollectors, optional `allowNegativeSupersession`.
- Pipeline: validate receipts → build superseding→target edges from `evidence.supersedes_receipt_ids`
  → validate every edge with typed reasons → detect/reject cycles → compute transitive closure →
  active = accepted \ superseded → expose chains, invalid edges, unresolved ids, diagnostics.
- Deterministic ordering: sort receipt ids and edges (`superseding_id`, then `target_id`); input order,
  append order, object-key order never affect the result.
- Engine integration: `classifyDimension` calls the resolver BEFORE state classification; classify
  ACTIVE receipts only; dimension output gains `superseded_receipt_ids`, `supersession_chains`,
  `invalid_supersession_edges`. The policy engine remains the SOLE proof-state authority — the graph
  builder must NOT re-interpret supersession.

## Edge validation rules (ALL must hold for an edge to be accepted)

1. target receipt exists
2. superseding and target hashes validate (canonical)
3. same capability_id
4. same dimension
5. same source_sha
6. same deployed_sha when the dimension requires deployed binding (HIGH_TRUST)
7. compatible environment (when policy requires one)
8. superseding observed strictly LATER than target
9. superseding receipt_id != target receipt_id (self-supersession rejected)
10. no cycle (edge accepted only when target cannot reach superseding in the accepted-edge graph)
11. superseding collector permitted by policy for the dimension
12. result transition eligible: VERIFIED may supersede NOT_VERIFIED or VERIFIED;
    NOT_VERIFIED supersedes VERIFIED only when policy explicitly opts in;
    UNKNOWN_OR_NOT_PROVEN and CONFLICTED never supersede
13. runtime cannot supersede shadow; shadow cannot supersede runtime; neither supersedes live
    (enforced structurally via CROSS_DIMENSION + dimension-scoped edges)

## PITFALL: `allReceipts` — cross-target classification

If the resolver resolves targets only against the dimension-filtered receipt set, a
cross-capability or cross-dimension target is reported `TARGET_NOT_FOUND` instead of
`CROSS_CAPABILITY` / `CROSS_DIMENSION`. Build the target id→receipt map from ALL receipts
(`allReceipts` passed alongside `receipts`); only receipts of the current dimension may act as
superseding. The engine passes the full ledger receipt array as `config._allReceipts`.

## PITFALL: true cycles are unconstructible with content-addressed ids

Receipt ids are sha256 over content INCLUDING `supersedes_receipt_ids`. A cycle A→B→A requires
idA = f(idB) and idB = f(idA) — a fixed point that does not converge (verified empirically:
iteration over canonicalization never stabilizes). Additionally, the NOT_LATER rule makes mutual
supersession time-impossible. Consequences:

- Through the policy engine, cycles fail closed via NOT_LATER (an earlier edge is rejected) — correct
  and deterministic, but the CYCLE reason never fires on real canonical data.
- Test the resolver's cycle defense DIRECTLY on the edge graph (`buildSupersessionGraph` exported):
  pass fabricated edges (e.g. A→B, B→A) and assert one edge accepted + one rejected with reason
  CYCLE. Use explicit receipt ids for self-supersession tests too (self-reference is also
  unconstructible canonically).

## PITFALL: candidate SHA authority after a policy-engine repair

After merging a policy-engine repair, the candidate graph must STAY bound to the ORIGINAL deployed
SHA (b3b79ee), NOT the repair merge SHA. Runtime/shadow receipts are bound to b3b79ee
(source_sha == deployed_sha == b3b79ee); a repair-SHA-bound candidate rejects every high-trust
receipt with SOURCE_SHA_MISMATCH → runtime/shadow revert to NOT_VERIFIED/UNKNOWN → zero promotions.
Rule: candidate source/deployed SHA follows the RECEIPTS' binding, not the policy code's HEAD.
Document this authority model explicitly before candidate construction (design note).

## Deploy scope for host-side-only repairs

When the repaired module is consumed only by the host-side publisher (imports modules from the
checkout), NO container restart is required or authorized: the container code is unchanged. Run the
publisher from a checkout of the merge SHA. A container restart is needed only if the changed module
is loaded inside the application process.

## Phase 8 rebuild discipline (post-repair candidate)

- Re-evaluate the EXISTING ledger receipts with the repaired engine — do NOT regenerate
  runtime/shadow evidence unless validation proves existing receipts invalid.
- Do not append duplicate receipts: `receipt_set_hash` should be UNCHANGED after a pure
  re-evaluation (a changed hash signals accidental appends).
- Run publisher dry-run against the REAL production graph dir; expected promotions = the repaired
  matrix delta (e.g. 10 promotions: runtime+shadow VERIFIED ×5+5, 0 demotions).
- Keep the previous/current chain: previous = current manifest graph (a7867b10), candidate bound to
  the receipts' SHA.
