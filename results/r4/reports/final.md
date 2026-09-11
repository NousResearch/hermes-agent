# R4 Final Report — Temporal + Verified Knowledge Lifecycle

## Baseline
- Commit `f97a4102dd`, Hermes 0.21.1, Python 3.14.3, numpy 2.5.1,
  pytest 9.1.1, SQLite 3.50.4. Frozen: `results/r4/baseline/env.json`.
- Scope note: NO UI Engineering/NameThatUI/project-mapping subsystem exists
  in repo (verified by search) — lifecycle implemented on Holographic facts;
  no subsystem invented.

## Temporal model (PROMOTE)
- States ACTIVE/SUPERSEDED/STALE/REVOKED/CONFLICT; history never deleted.
- Additive columns (lifecycle, superseded_by, verified_at/by, source_ref/sig)
  + `fact_lineage` table. Legacy-safe (incl. primitive 2-col DBs).
- Verification confers only ACTIVE (forged states refused); timestamps
  server-side; source signatures size:mtime:sha8 (1MB cap, read-only).
- Revalidation: UNCHANGED (no mutation) / CHANGED/MISSING→STALE /
  CONFLICT→conflict. No blind TTL (event-driven only — measured decision).

## Retrieval precedence (PROMOTE)
- Partition: verified(0) < active(1) < conflict(2) < stale(3) <
  superseded(4) < revoked(5); stable within-class order. Verified beats
  high-trust stale absolutely. Stale-only results flagged `(stale)` in
  prefetch and `stale:true` in rows. Active-row dicts carry the additive
  R4 columns (schema evolution, both parity arms read them — parity test
  green unmodified); only the computed `stale` flag is conditional.
- Partition is BY DESIGN (mission: stale must never outrank verified),
  not reordering: within-class relevance untouched.
- Vector paths (probe/related/reason) apply lifecycle demotion weights +
  partition; contradict excludes revoked rows (both passes).

## Repair cycle (reviewer FAIL F1-F5, all addressed + re-tested)
- F1: verify refuses revoked/conflict/superseded; mark_stale active-only;
  update content voids attestation; supersede validates both ends +
  cycle guard; remove transactional + demotes orphaned rows to stale.
- F2: demotion/flags on probe/related/reason/contradict/list_facts.
- F3: signatures drop mtime; conflicts scoped to category.
- F4: lifecycle_reason persisted; canonical successor; fetch guarded;
  file-oracle documented as trusted-paths-only.
- F5: savepoint-based _atomic() everywhere (nest-safe); supersede/update/
  add/batch/remove all atomic.
- Reviewer minors-2: env pattern prefix floor fixed (AWS_SECRET caught);
  zero-width Cf-strip + MixedCase covered with programmatic tests.
- Repair-2 (reviewer minors): remove_fact nulls superseded_by pointers;
  supersede refuses revoked old rows; `.get("stale", False)` contract
  documented (parity shape preserved).

## Self-heal find (PROMOTE)
- Ancient DBs (rows predating facts_fts + helper-first schema mutation) carry
  an FTS image failing trigger writes. Savepoint probe on a real legacy row
  detects it; DROP+recreate FTS+triggers from facts heals it. R2-era full
  DBs upgrade cleanly (verified). No data loss in any path.

## Paraphrase: stem-union REJECTED (R2 delta +0.000, 30-set +0.033, FP clean).
## Thai: spaceless candidate delta 0 → KEEP-CURRENT, NO DEPENDENCY.
## TTL: REJECTED (no blind expiration; event-driven revalidation instead).
## Semantic backend: remains REJECTED (no backend, no new evidence).

## Benchmarks
- Temporal: precedence/supersession/lineage/revalidation/concurrency green.
- Paraphrase: 30 cases + R2-106 stem sweep + FP probes.
- Thai: 20 cases x current/candidate.
- R3 suites untouched and green.

## Security: forged timestamps/states/paths/lineage/injection/secrets/
##   fake-authority/cross-project all refused or inert. No new egress.
## Concurrency: verify/stale races, retrieval-during-update,
##   rollback-during-update green. No lost updates/lineage.
## Scale: verify ~3.5ms, supersede ~6.9ms, revalidate ~0.05ms (flat 100→1k);
##   search-with-lifecycle 1.1→3.5ms; R3 10k/50k numbers reused (no write-path change).
## LLM: 0 everywhere. Ad-hoc final: 11/11.
## Regression: bare CI-parity env (numpy present) **187/187 green**;
##   official runner 0 failed (all skips numpy-absence by repo convention).
##   R1/R2/R3/R4 gold identical (parity test unmodified).
## A/B: R1/R2 gold identical post-R4 (parity test green unmodified).
## Promotion: temporal+precedence+self-heal PROMOTE; stem/TTL/semantic REJECT.
## Remaining: stemming-level paraphrase, full temporal lifecycle policy
##   (aging automation), Thai segmenter re-evaluation.
## Reproduce: R4 test files + `bash scripts/run_tests.sh` full gate.
