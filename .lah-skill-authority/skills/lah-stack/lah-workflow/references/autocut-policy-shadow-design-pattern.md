# Bounded Policy Design + Shadow Certification Pattern (LAH Stack)

Class-level workflow for missions that design/certify a DETERMINISTIC decision
policy (zone auto-cut, budget, pacing, ...) in SHADOW/DESIGN-ONLY mode with
historical replay, TDD and a certification verdict. First certified with
`CLOE_BOUNDED_AUTOCUT_POLICY_V1` (lah-brain PR #213, merge 2f9bcc38, 2026-08-11).

## When this applies
- Mission says "DESIGN + SHADOW SIMULATION ONLY", "must not execute", "fail closed".
- Deliverable includes a policy module, replay fixtures, receipts, certification.json.

## Sequence (proven)

1. **Phase 0 authority freeze** — `git fetch origin main`, record origin/main SHAs
   for all affected repos; inspect existing governance BEFORE writing code:
   approval queues (UNEXECUTABLE action types), readiness gates
   (`autocut_data_readiness`, `EXOCLICK_LIVE_ENABLED`), capability registries,
   zone economics contract fields. Map authority matrix:
   lah-brain = business decision owner; openclaw-runtime = execution/read surface;
   lah-core = outcome ingestion. Never add live execution wiring.

2. **Branch from origin/main, not local HEAD** — local worktree HEAD is often BEHIND
   the certified origin/main. `git checkout -b <mission> origin/main` guarantees the
   branch sits on the certified authority. Verify `git merge-base --is-ancestor`.

3. **Deterministic policy module** — pure functions, no LLM, no I/O inside the
   decision path. Structure:
   `profiles.js` (frozen threshold variants CONSERVATIVE/BALANCED/AGGRESSIVE),
   `gates.js` (hard safety gates → exact blocker codes), `engine.js` (decision
   precedence + campaign global-failure + shadow cycle with rate limits),
   `receipt.js` (required receipt fields), `index.js` (public exports).

4. **TDD strictly** — write the full test suite FIRST, run → RED (module absent),
   implement → GREEN, keep regression green. 49 tests covered: gates G1-G10,
   decisions D1-D10, false positives FP1-FP9, campaign C1-C4, rate limits R1-R3,
   T01 replay, sensitivity S1-S3, profiles.

5. **Zero-vs-missing + precedence** — `paid_conversions = 0` (observed) and
   `null` (unavailable) MUST diverge (test it). Precedence that works:
   paid ≥ 1 → KEEP (strongest positive evidence) → downstream ≥ 1 → KEEP →
   hard gates → NO_AUTO_ACTION + blocker → too-early low spend → KEEP →
   STRONG_CUT_CANDIDATE (volume OR share path) → CUT_CANDIDATE → WATCH → default KEEP.
   Positive signals short-circuit BEFORE gates: a zone with a real event stays KEEP
   even under BLOCKED readiness. Corollary: in the PRODUCTION pass, fixtures must
   carry the OBSERVED zeros (downstream=0) so the gate failure actually fires
   NO_AUTO_ACTION — otherwise the event-protect path masks fail-closed behavior.

6. **Fixtures with provenance** — reconstruct checkpoint rows from certified
   evidence only; label every row `FIXTURE_RECONSTRUCTED` / `FIXTURE_SYNTHETIC_*`;
   declare model assumptions explicitly (`ASSUMPTION_MODEL=SPEND_SHARE` for
   arrivals); NEVER fabricate production outcome data. Run TWO passes:
   production readiness (BLOCKED → every zone NO_AUTO_ACTION, the fail-closed demo)
   and fixture READY (normal evaluation).

7. **Rate limits / cooldown** — `max_cut_count_per_checkpoint` +
   `max_removed_spend_share_per_checkpoint`. KEY SEMANTIC: the FIRST cut of a
   checkpoint is always permitted (a single dominant dead zone must stay
   actionable even above the share cap); subsequent cuts bounded by the cap.
   Track `previously_cut` across checkpoints → no double cut.

8. **Global failure supersedes per-zone cuts** — when campaign-level rule fires
   (spend ≥ floor, arrivals large, downstream negligible, paid = 0), per-zone
   hypothetical cuts become `NONE` + `superseded_by_global_pause` → pause
   recommendation instead of a mass-cut cascade.

9. **Sensitivity** — 3 profiles over a labeled synthetic archetype corpus;
   score by priority: (1) zero paid-converting cuts, (2) protect event-bearing
   low-volume zones, (3) identify spend concentration, (4) reduce dead spend,
   (5) minimal churn. Pick the default from evidence, not preference.

10. **Counterfactual** — always flag `COUNTERFACTUAL_NOT_CAUSAL`; compute
    `spend_saved_if_future_share_constant = share_at_cut × (campaign_final − campaign_at_cut)`;
    sanity-check against the certified final zone spend (delta ≈ 0 proves the
    model); never claim conversions.

11. **Certification** — statement list (10 for autocut v1) + SAFETY COUNTERS
    (`provider_calls=0, zone_cut_calls=0, campaign_mutations=0`) + verdict.
    Artifacts: `audit/<policy>/` (policy-spec.json, thresholds.json,
    t01-replay.json, sensitivity-analysis.json, false-positive-tests.json,
    shadow-decisions.ndjson, certification.json) + `docs/superpowers/plans/<policy>.md`.

12. **Commit/PR/merge** — scoped `git add <paths>` (NEVER `git add -A`);
    PR via gh; merge commit (traceable SHA); lah-brain auto-deploys origin/main →
    live, so verify `curl https://leanframeworklab.com/version` returns the merge
    SHA. `/business/runtime-context` is auth-gated (401 without token) — that is
    NOT a deploy failure; `/version` is the deploy proof.

## Pitfalls (all hit in v1 certification)
- **Composite read-only shell verification gets DENIED by the user**
  (`ls && ... && node -e ...` chains). Inspect artifacts with read_file /
  search_files; keep shell commands single-purpose. (User rule, standing.)
- **`.mjs` = ESM scope** — `require` throws. lah-brain has no `"type": "module"`;
  scripts must be `scripts/foo.js` (CJS), not `.mjs`.
- **Fixture context placement** — campaign-context fields (`checkpoint`, etc.)
  must live INSIDE the campaign object, not only on the checkpoint wrapper,
  or every receipt carries `checkpoint: null`.
- **Default campaign fixtures mask rate-limit tests** — the default campaign
  (arrivals ≥ global threshold) triggers the global pause rule, which supersedes
  all cuts; override `campaign_affiliate_arrivals` below the global threshold when
  the test targets rate-limit behavior.
- **Thresholds are hypotheses** — profile values (e.g. keep_max_spend 0.25,
  strong_min_share 0.25) are tuned by replay evidence; record rationale +
  confidence per threshold in the plan doc and thresholds.json.
