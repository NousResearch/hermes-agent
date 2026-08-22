# Experiment Preparation Fast Path (F0–F3) — composition map & pitfalls

Session: LAH_EXPERIMENT_PREPARATION_FAST_PATH_F0_TO_F3_V1 (openclaw-runtime, merged PR #798 → a78d84b).
Evidence pack: /home/deploy/evidence/lah-experiment-preparation-fast-path-f0-f3-v1/ (report, reuse-matrix, f0-5, f0, f1, f2, f3, performance, t03-like-replay, test-results, runtime-proof, safety-proof).

## The one-liner
Prepare = COMPOSE, never rebuild. 19 canonical components already exist; the facade wires them.
Default action hierarchy: REUSE_AS_IS → WIRE_EXISTING → EXTEND_EXISTING → BUILD_MINIMAL (orchestration only).

## Canonical composition chain (new facade)
REQUEST → Business Watch (runBusinessWatchEvaluation) → candidate scope (loadCandidateScope) →
offer map (loadBusinessOfferMap) → runExplorationBridge (coverage + NBT + Ramanujan; only planner) →
getBenchPreparationMetadata + recommendBenchTopology (F1) → buildPacketAuthorities →
buildExecutionPacket + withProposedBudget → evaluateExecutionGates (DRY_RUN) →
evaluatePreLiveEnforcement (preparation report only) → executePacketDryRun (zero writes) →
createPreparationReadinessCache snapshot (F2) → classifyPreparationStatus (F3) → receipt.

Key modules: src/services/experiment-preparation-service.js (facade),
src/services/preparation-readiness-cache.js (F2), src/services/preparation-exception-routing.js (F3),
src/routes/experiment-preparation.routes.js (POST /experiments/prepare + GET /prepare/health).

## Key contracts
- Status vocabulary: READY_FOR_OPERATOR_AUTHORIZATION | BLOCKED_CONFIGURATION | BLOCKED_EVIDENCE |
  BLOCKED_REPAIR_REQUIRED | BLOCKED_AUTHORITY. Preparation NEVER returns LAUNCHED (test-enforced).
- Execution-mode aliases: DIRECT_PROVIDER → resolved by bench topology (VERTICAL_BENCH →
  PARALLEL_CAMPAIGNS when per-arm spend attribution required); LAH_ROUTER → ONE_CAMPAIGN_LAH_ROUTER;
  PARALLEL_CAMPAIGNS / ONE_CAMPAIGN_LAH_ROUTER as-is; anything else → BLOCKED_CONFIGURATION
  (operator choice, NOT Hermes).
- Repair contract (F3): requirement_id, component, capability, defect_class, observed, expected,
  blocking_checks, evidence_refs, affected_dependency_hashes, suggested_action_class
  (WIRE_EXISTING|REPAIR_EXISTING|EXTEND_EXISTING), provider_mutation_required=false, spend_required=false.
  ONLY BLOCKED_REPAIR_REQUIRED routes to Hermes; authority/evidence/configuration blockers stay out.
- Readiness cache (F2): reuse iff exists ∧ !expired ∧ dependency_hash matches ∧ not never-reuse.
  dependency_hash = sha256 of canonical (key-sorted) dependency VALUES. PACKET_INTEGRITY has ttl 0
  (never reused across packet contents). Budget-only delta must NOT redo platform certification.
  Provider-auth checks (EXOCLICK_READ_AUTH, CRAKREVENUE_READ_AUTH, PROVIDER_NATIVE_CAP_READABILITY)
  are UNVERIFIED in preparation mode unless reused from valid cached evidence — never fabricated PASS.

## F0.5 memory-wiring rule (missing != zero)
File-backed stores (experiment-memory-file-store, BI memory, execution-receipt-store) share the
atomic tmp+rename JSON convention. Keep a health() contract: missing/valid-empty file = REAL empty
(ok:true, count 0); corrupt file = fail-closed (ok:false, read_error:true). Live routes must never
pass hard-coded `experimentRecords: []` — load the durable store and fail closed when unreadable;
expose provenance (count / records_used) in the response.

## Pitfalls
- patch tool on big frozen object literals (e.g. TEST_BENCH_REGISTRY): fuzzy matching consumed a
  sibling key header (`SOURCE_VERTICAL_EXPERIENCE_BENCH: Object.freeze({`) and turned a property
  separator into `;` (SyntaxError). Anchor on unique strings INCLUDING the next sibling's header;
  after batch edits re-read the file and `node --check`. Inside object literals, property separators
  are `,` — a trailing `;` after `Object.freeze({...})` breaks the literal.
- Full-suite regression attribution: run the suite on BOTH branch and baseline (npm ci in the
  wt-deploy worktree is fine — node_modules is transient/ignored), grep `^not ok`, sort, then
  `comm -23` to isolate branch-only failures; run those files in isolation at both SHAs. Watch for
  environmental artifacts: the pr-autopilot CLI auto-detects the open PR for the current branch
  (a test expecting "PR number is required" fails while a PR exists for that branch — not a
  regression); fixture flakiness may reproduce at baseline.
- CI gate `ci-governance` can fail in ~2s for ACCOUNT reasons ("recent account payments have failed /
  spending limit needs to be increased") — attribute with `gh run view <id>` / annotations before
  assuming code failure. Base-branch policy requiring CI + broken billing → `gh pr merge --admin`
  under explicit mission deployment authority (single-maintainer repo).
- Forcing BLOCKED_EVIDENCE in tests: the V2 fixture map has no proposition with 2 monetization
  contracts, so `bench_type: 'ECONOMIC_BENCH'` makes the planner return NO_VALID_TEST deterministically.
- Offer business view: execution_status / provider_approval live under `offer.eligibility_facts`
  (NOT top-level); tracking under `offer.tracking_readiness`; `canonical_id` is the map offer id key;
  destination URLs are provenance-gated (may be absent → dry-run payloads skipped with a caveat).
- Verify provider_write=false / spend=0 / no-LAUNCHED on EVERY receipt path (service + route +
  acceptance), not just the happy path.
- Runtime verification etiquette: if the operator denies credential/container access, do NOT retry or
  rephrase — switch to keyless probes (/health, 401-without-key proves the route is registered and
  auth-gated), document the exact keyed command for the operator in the evidence, finalize.
