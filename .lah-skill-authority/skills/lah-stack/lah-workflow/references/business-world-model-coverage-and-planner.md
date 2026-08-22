# Business World Model — Coverage Matrix & Next Best Test Planner (LOT B)

Deterministic decision layer for the LAH Business World Model mission family.
LOT A = model the world (WHAT EXISTS). LOT B = decide (WHAT WE KNOW /
WHAT WE DON'T KNOW / WHAT SHOULD WE LEARN NEXT). LOT C = act + learn.
LOT B returns a shadow recommendation and executes NOTHING.

## Architecture (reuse contracts, never duplicate)

- LOT A consumed as-is: `canonical-traffic-context.js` (deriveContextId),
  `business-offer-map.js` (propositions/offers/landings), `test-bench-registry.js`
  (bench vocabulary + comparability), `bench-design-validator.js`
  (validateBenchDesign on selected designs), lah-brain
  `affiliate-experiment-memory.js` (read-only evidence records).
- LOT B1 `src/services/coverage-matrix.js` — sparse evidence-driven coverage
  model. Cells materialize ONLY from direct evidence OR map candidates for the
  queried context. No Cartesian product, no new persistent store (pure
  projection: context + evidence + deterministic policy).
- LOT B2 `src/services/next-best-test-planner.js` — deterministic explainable
  planner. Decisions: NEXT_TEST / NO_VALID_TEST / NEED_MORE_AUTHORITY /
  VALIDATE_EXISTING_WINNER / RETEST_STALE_EVIDENCE.
- Cloé: additive items `coverage_matrix_summary_v1` + `next_best_test_proposal_v1`
  in `cloe-canonical-business-context.js`, explicit-query-only, fail-closed.
- Tests: 24 coverage + 26 planner + 4 Cloé integration + 10/10 behavioral sim.

## Coverage state machine (COVERAGE_POLICY_V1, deterministic, conservative)

- UNEXPLORED → PROBING: ≥1 valid measured evidence record.
- PROBING → PROMISING: profit > 0 AND provider_clicks ≥ 200 AND spend ≥ $5
  AND evidence quality not INSUFFICIENT.
- PROMISING → PROVEN: replication ≥ 2 AND clicks ≥ 2000 AND impressions ≥ 20000.
- → REJECTED_IN_CONTEXT: spend ≥ $10 AND profit/spend ≤ -0.3, quality not
  INSUFFICIENT. Zero conversions with tiny sample NEVER rejects.
- PROMISING/PROVEN → DECAYING: evidence older than staleDays (14) or material
  recent deterioration. Freshness: FRESH <7d / AGING 7-14d / STALE >14d.
- Fixture authority (CRAKREVENUE_PHOENIX_*, RESEARCH_SNAPSHOT, DERIVED,
  CONFIGURED) caps state at PROBING — never PROMISING/PROVEN/REJECTED.
  Research snapshot records are HINTS only (prior_hints), never direct proof.
- Directional conflict (positive AND negative-crossing records) → state PROBING,
  confidence LOW, conflicts[] surfaced.
- NO global negatives: rejections are always REJECTED_IN_CONTEXT for one cell.
- Cell identity = hash(context_id + bench_type + hypothesis dims). Metrics /
  timestamps / profit NEVER in identity (evidence updates the same cell).

## Pitfalls that cost real iterations (learned building LOT B1/B2)

1. **Evidence projection must cover EVERY bench a record directly evidences.**
   A record that ran an offer through the provider's default/direct landing
   directly evidences EXPERIENCE_FORMAT_BENCH (`provider_direct_landing`) AND
   OWNED_VS_PROVIDER_BENCH (`PROVIDER_DIRECT_LANDING`) — not just the vertical /
   provider / landing cells. Skipping these makes those benches look fully
   unexplored (novelty 1) and they outrank the correct bench. Every resolved
   offer record should emit both cells.
2. **Novelty / validation / decay scoring must be ARM-coverage-based**, not
   cell-count-based. Benches whose candidate cells are NOT materialized
   (EXPERIENCE, OWNED) had cells=0 → novelty 1 → they beat VERTICAL/PROVIDER.
   Score from the share of candidate arms by `coverage_state`
   (unexploredArms/totalArms etc.), keep cell counts for reporting only.
3. **Vertical-space-completeness gate is per DISTINCT VERTICAL, not per cell.**
   "Every vertical has ≥1 conclusive verdict (PROVEN / REJECTED_IN_CONTEXT /
   DECAYING)" — NOT "every cell decided". An unexplored proposition cell inside
   a decided vertical (e.g. MFC under CAM when Jerkmate is PROVEN) is exactly
   the PROVIDER-level discovery the deeper bench exists for; treating it as
   vertical incompleteness wrongly blocks PROVIDER_BENCH.
4. **Planner decision order matters**: stale-positive cells (DECAYING or
   PROVEN/PROMISING+STALE) → RETEST_STALE_EVIDENCE BEFORE deeper-bench
   ranking (Case E doctrine); then rank; then fully-decided + no deeper
   unexplored bench → VALIDATE_EXISTING_WINNER; then authority gates
   (tracking/approval/eligibility) → NEED_MORE_AUTHORITY with blocked_reasons.
5. **Same proposition + multiple monetization offers = ONE discovery
   hypothesis.** OurDream 10138 (PPS) + 10139 (RevShare) → one VERTICAL cell,
   novelty only inside ECONOMIC_BENCH. Add a deterministic rejected-alternative
   label: SAME_PROPOSITION_MONETIZATION_IS_ECONOMIC_BENCH_NOT_VERTICAL_NOVELTY.
6. **Smartlink never an ordinary causal arm.** Only inside LAH_VS_SMARTLINK_BENCH
   (DISCOVERY_ONLY), and only when explicit scope authority
   (constraints.smartlink_bench_allowed) is present. Search ALL map offers for
   the smartlink arm — a smartlink may exist WITHOUT a configured EXECUTABLE
   record.
7. **Heterogeneous conversions** (PPS_SALE vs DOI_LEAD): never a
   comparable_conversion_rate anywhere; success_evaluation_contract carries
   COMMON_ECONOMIC_METRICS_ONLY.
8. **Why-narrative readability**: uppercase vertical labels in `why_this_bench`
   (CAM=PROVEN not cam=PROVEN) and append the varied dimension to untested
   candidates (`ourdream.ai - Revshare (REVSHARE)`) so assertions on the
   rationale match contract vocabulary.

## Fixture notes

- The V2 bench (10138 AI / 8780 CAM / 8517 DATING) had NO real canonical
  experiment records at LOT B authority — coverage honestly reports all
  UNEXPLORED; configured/Phoenix facts are existence evidence (prior_hints),
  never direct cell proof.
- Phoenix fixture landings (39103 Interstitial, 35188 Performer, 40001-40007)
  materialize many LANDING candidate cells — a single default-landing record
  leaves them all UNEXPLORED (correct hierarchy).
- Behavioral sim S1-S10 runs the REAL entry points; mock only the
  configured/live/phoenix inventory sources. Merge-scoped receipt:
  commit_before = mission start SHA, commit_after = FINAL merged main
  (including doc PRs), written + validated from a post-merge worktree as the
  LAST artifact.
