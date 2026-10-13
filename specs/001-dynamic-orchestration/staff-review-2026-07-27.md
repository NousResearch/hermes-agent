# Staff review — dynamic orchestration contract slice

**Date:** 2026-07-27
**Scope:** pure contract/domain package in `agent/dynamic_orchestration/`; no production runtime wiring
**Review mode:** isolated independent-context review, read-only
**Cross-family status:** Terra returned PASS on the Wave 2H, Wave 3A2, Wave 3B and final expanded closure snapshots

## Initial verdict

**BLOCK** — `C: 0`, `H: 0`, `M: 4`, `L: 2`.

The independent reviewer verified the then-current source and tests, ran the focused suite, and reproduced the findings below. The full original report is retained outside the repository in the Hermes delegation cache; this file records the actionable, reviewable result and its resolution.

## Findings and resolution

### M1 — candidate and registry cardinality was unbounded — fixed

- **Symbols:** `_revalidated_route_registry`, `_validated_decision_candidates`, `_validated_decision_authorities`, `RouteDecisionV1.from_mapping`, `score_eligible_candidates`, `evaluate_route_eligibility`, `replan_after_capacity_exhaustion`.
- **Failure:** 257 candidates and 513 routes could be materialized, copied, revalidated and sorted without an earlier domain bound.
- **Impact:** avoidable CPU/memory amplification on public and trusted-authority boundaries.
- **Resolution:** candidates/facts are bounded at 256; route/scoring/authority registries are bounded at 512 before iteration or conversion. Direct, persisted, scoring and replanning paths use the same limits.
- **Regression:** `test_route_decision_bounds_candidate_and_route_registry_cardinality`.

### M2 — textual fields bypassed the 8,192-character bound — fixed after re-review

- **Symbols:** `AcceptanceThresholdV1.__post_init__`, `RuntimeErrorClassificationV1`, `_validate_audit_event`, `_ascii_trimmed_nfc`, `_normalized_text`, `_reject_sensitive`.
- **Failure:** the initial fix covered threshold/audit text, but the first re-review proved direct and persisted `quota_pool_id` still accepted 8,193 characters.
- **Impact:** unbounded retained text and avoidable preprocessing work.
- **Resolution:** raw strings are bounded before trim/NFC; threshold, classification and audit fields use the bounded helper. The sensitive-value scanner skips regex for oversized text so the owning field validator preserves its stable error code.
- **Regression:** `test_public_contracts_bound_all_textual_audit_and_threshold_fields` and `test_runtime_classification_bounds_quota_pool_text_direct_and_persisted` cover N/N+1.

### M3 — graph budget was too late for collection materialization — fixed after re-review

- **Symbols:** `AuditedModelJustification.__post_init__`, `evaluate_route_eligibility`, `QualityCompensationPlanV1.__post_init__`, `QualityCompensationPlanV1.from_mapping`, scoring/decision/replan boundaries.
- **Failure:** the initial fix bounded primary collections, but the first re-review proved Task identity claims, persisted Decision reason/evidence and direct Replan evidence could still iterate or copy before their field bounds.
- **Impact:** the graph budget did not bound all pre-validation work.
- **Resolution:** identity claims, thresholds, attestations, persisted decision collections and recheck evidence are cardinality-checked before sort/tuple/comprehensions; eligibility iterables consume at most `limit + 1`; oversized containers are not traversed by the secret scanner before their owner validates them.
- **Regression:** collection tests now include N/N+1 and iteration-bomb probes for Task, persisted Decision, Replan, eligibility and scoring.

### M4 — review and test evidence was stale — fixed

- **Files:** `tasks.md` and this review.
- **Failure:** they claimed 144 focused tests and no H/M while the reviewed snapshot had an additional failing bound test and four MEDIUM findings.
- **Impact:** misleading merge evidence.
- **Resolution:** evidence records the BLOCK sequence, the final 332-test Wave 2H snapshot, current hashes, broad-suite limitations and the narrow cross-family PASS.

### Wave 2 review — BLOCK with six MEDIUM findings; remediated in Wave 2B

- Terra reproduced incompatible v1 wire fields, stale/unknown/expired capacity becoming healthy, numeric minima across incompatible units, input-order/self-reported authority, caller-authorized execution claims, and raw `ValueError` leakage.
- Wave 2B replaced the invented DTO shape with the canonical contract, added external unit/authority registries, fail-closed execution rehydration against trusted Task/Decision/route/validator authority, coherent state/timestamp invariants and stable domain errors.
- No provider, credential material, storage, dispatch, gateway or runtime wiring.
- Focused tests cover strict schema, deep immutability, unknown epistemics, expiry/conflict policy, canonical/hash-seed determinism and credential-material rejection.

### Wave 2B re-review — BLOCK with one HIGH and five MEDIUM findings; remediated in Wave 2C

- **HIGH:** caller-supplied maps named “trusted” could fabricate terminal `ATTESTED` without an external authority store.
- **MEDIUM:** execution wire-field drift, future observations treated as fresh, unregistered authoritative sources accepted, malformed inputs leaking implementation exceptions, and forged typed capacity DTOs retained.
- **Resolution:** the pure phase has no path to `ATTESTED`; terminal envelopes remain explicitly activation-blocked. Capacity and execution nested DTOs are rehydrated, registries are bounded/typed, future and non-authoritative observations fail closed, and the canonical v1 field set is exact.

### Expanded Wave 2C reviews — BLOCK; remediated in Wave 2E

- A Terra review found four MEDIUM gaps: capacity wire drift, incompatible authoritative-unit behavior, recursion through typed dataclass cycles, and missing pure reservation/breaker contracts.
- A separate isolated review returned `C: 0`, `H: 1`, `M: 4`, `L: 2`. The HIGH proved caller-provided reviewer/evidence/approval/reservation inputs could still produce `AUTHORIZED`, `dispatchable=True`, or `DISPATCHED` even though terminal `ATTESTED` had been removed.
- Its MEDIUM findings covered coercive `str()` boundaries, forged nested quality DTOs, fallback execution authority reconstruction and stale evidence.
- **Resolution:** while no external authority store exists, every route decision is non-dispatchable and every execution envelope is reservation-less, outcome-less, activation-blocked `PENDING`; V2/V3/V4 invariants fail closed; scalar inputs require exact bounded strings; quality plans structurally rehydrate nested DTOs; fallback execution preserves the explicit external-authority block; capacity wire and authoritative semantics match the canonical contracts; typed rehydration is cycle/depth/node bounded; and immutable secret-free `CapacityReservationV1`/`CircuitBreakerV1` contracts exist without claiming storage or runtime wiring.
- The combined Wave 2E RED run reproduced 18 failures before implementation; the focused GREEN runs pass 230 tests across the isolated runner and hash seeds 1/99.

### Wave 2E review — BLOCK with four MEDIUM findings; remediated in Wave 2F

- Pool identities could split by case/whitespace across observations, views, authority registries and route-bound reservation/breaker contracts.
- Huge integers and hostile numeric values could leak `OverflowError`/conversion exceptions or survive without a semantic upper bound.
- Persisted candidate/status/relation and peer enum parsers retained arbitrary `str()` coercions.
- **Resolution:** pool IDs now share RouteV1's bounded canonical identity; authority-key collisions fail closed; one exact finite-number converter contains all public conversion failures; public status/enum parsers accept only exact supported values; and context, concurrency, probe-budget and version integers have explicit persistence-safe N/N+1 bounds.
- Wave 2F reproduced 27 failures in RED and added 54 focused regressions; the complete focused suite passes 296 tests under default and hash seeds 1/99.

### Wave 2F/2G reviews — BLOCK; remediated in Wave 2G/2H

- Terra found caller-supplied review/evidence maps could still alter the quality-compensation block reason, typed dataclass rehydration could leak attacker `RuntimeError`, and effort/verification labels reached membership/rank lookup before exact type validation.
- Wave 2G made all pure-phase review maps non-authoritative, contained hostile deepcopy failures and required exact labels before hash/rank lookup. The executor hit its wall timeout after writing source/tests; the supervisor resumed from the preserved hashes and independently verified all gates.
- The next review found only two deterministic persistence gaps: candidate order and set-like capacity provenance order. Wave 2H sorts candidates by canonical route ID, canonicalizes provenance collections and rejects post-normalization duplicates without changing ordered fields or score selection.
- Terra's isolated narrow closure review verified both findings with 13 targeted tests and one 332-test full focused run, confirmed hashes before/after and returned `PASS` with no HIGH/MEDIUM regression.

### Wave 3A performance review — BLOCK; remediated in Wave 3A2

- The initial 256-candidate `replan_after_capacity_exhaustion` baseline was 653.6 ms median. Wave 3A removed redundant route revalidation after validation in the same top-level call and added a deterministic JSON benchmark.
- Terra blocked the first optimization because a lexical fast path could accept a sensitive `decision_id`, oversized sensitive mappings had unstable error precedence, and benchmark iteration arguments lacked a practical upper bound.
- Wave 3A2 restored full sensitive-value validation and stable precedence, bounded benchmark arguments, added adversarial regressions, and retained a 256-candidate median of approximately 34–39 ms with an unchanged decision fingerprint.
- The focused suite increased to 375 tests. Terra independently reproduced the original HIGH/MEDIUM probes and returned `PASS` on the fixed hashes.

### Wave 3B/3C maintainability and DX — fixed and verified

- The 5,570-line god-file was replaced by an acyclic 11-module package (`validation`, `route`, `task`, `eligibility`, `quality`, `policy`, `decision`, `state`, `capacity`, `execution`, and thin root reexports). No implementation was hidden or duplicated.
- Runtime/AST API manifests match: 72 public root exports, signatures, dataclass fields/defaults/frozen metadata, enum values, constants and required private compatibility attributes. The required-surface diff is empty.
- Terra reconstructed the pre-extraction monolith, compared all 154 definitions, exercised 23 fresh import orders, verified legacy pickle loading and reran the original/package suites and benchmark. It returned `PASS` with no HIGH/MEDIUM/LOW findings.
- The 5,000+ line focused test was mechanically split into six domain files plus one non-test support module. The semantic node-ID multiset is exactly 375 before and after, with zero missing, extra or duplicate cases; the package compatibility module adds two cases.
- Combined, hash-seeded, reverse-order, shuffled-module and isolated per-file runs all pass 377 tests.

### L1 — adversarial custom `Mapping` can leak implementation exceptions — closed (executor lane)

- **Symbol:** new `_mapping_snapshot`, `_validated_mapping_keys`, `_reject_sensitive`, and every public `from_mapping` boundary.
- **Scenario:** a custom `Mapping` whose `__len__`, iteration, `.items()`, `.get()` or `__getitem__` raises could surface that implementation exception rather than `DomainValidationError`.
- **Fix:** a bounded, hostile-safe input-adapter snapshot (`_mapping_snapshot`) now runs at every public `from_mapping` entry (24 sites across route/task/eligibility/quality/capacity/decision/execution) and at the three nested untrusted-mapping reads (decision trigger, persisted candidate, audited justification). Ordinary decoded `dict` input is returned unchanged (same object), so JSON/dict behaviour and error precedence are identical; any other `Mapping` is copied into a plain `dict` with every container hook contained as a stable `DomainValidationError`, and no value is coerced via `str()`. Oversized mappings keep the bounded key-prefix scan and are rejected without reading a single value. The shared secret scanner `_reject_sensitive` additionally wraps its graph walk so a hostile nested container is contained. Sensitive scanning, graph bounds, canonicalization and fail-closed authority are unchanged.
- **Evidence:** a RED demonstration that disabled the snapshot and the wrap reproduced 67 leaking assertions; the GREEN implementation closes them. Direct probes across all boundaries × the five hostile hooks × `{RuntimeError, TypeError, ValueError}` raise stable domain errors with no attacker message leak; ordinary `dict`/`OrderedDict`/`MappingProxyType` inputs stay equivalent. Regressions: `test_public_from_mapping_snapshots_contain_hostile_mapping_adapters`, `test_hostile_mapping_exception_types_are_all_contained`, `test_multi_authority_from_mapping_boundaries_contain_hostile_mappings`, `test_nested_hostile_mapping_inside_decoded_payload_is_contained`, `test_sensitive_scanner_contains_hostile_nested_container_adapters`, `test_oversized_hostile_mapping_scans_only_bounded_keys_across_all_peers`, `test_snapshot_preserves_ordinary_decoded_mapping_semantics`, `test_mapping_snapshot_primitive_preserves_dicts_and_rejects_non_mappings`.
- **Scope boundary (unchanged):** trusted-authority registries (`trusted_routes`, reviewer/execution/threshold maps) remain the future runtime authority/execution gate's responsibility, not this decoded-payload boundary; they were deliberately not broadened.

### L2 — traversal threshold edges are not exhaustively parameterized — closed (executor lane)

- **Covered previously:** oversized scalar, collection overflow, cycle, excessive depth, sensitive mapping key, candidates and registry overflow.
- **Fix:** added permanent focused regressions for the exact graph edges — depth `N` versus `N+1` (nested lists and dicts, and through the public `TaskEnvelope` decoded-object boundary), node budget `N` versus `N+1` in the complete-graph rehydration scan, the collection scan edge (deferred to the owning field validator at `N+1`), benign DAG aliases (accepted and de-duplicated, never mistaken for cycles), the alias fan-out node budget, and hostile containers at the boundaries. Tests prove accepted exact-boundary cases and rejected overflow cases with no timing assertion and no flaky recursion. No implementation defect was exposed; the graph budget was already correct, so no owning helper required a behaviour change.
- **Regressions:** `test_graph_depth_budget_exact_edge_accepts_n_and_rejects_n_plus_one`, `test_graph_depth_budget_enforced_through_public_decoded_object_boundary`, `test_graph_node_budget_exact_edge_in_complete_graph_scan`, `test_benign_dag_aliases_are_accepted_and_cycles_are_rejected`, `test_dag_alias_fanout_cannot_escape_the_complete_graph_node_budget`, `test_graph_collection_scan_exact_edge_defers_oversized_to_field_validator`.

## Current mechanical evidence

```text
Focused suite: 377 passed, 0 failed (default pytest and PYTHONHASHSEED=1/99)
Focused isolated wrapper: 377 passed, 0 failed across 7 files
Original/split semantic node IDs: 375 == 375; missing 0; extra 0; duplicates 0
Ruff: passed
py_compile: passed
git diff --check: passed
package aggregate SHA-256: a94a99dec041e5370628f3b43c62f4debcd2b57959fbd13f6941e3877c97893c
package compatibility test SHA-256: dac3f52120d0004b03f9eb70a5008f3590b63e879e4d0120ef278e3afbe6d7ed
benchmark SHA-256: 5acb53334f129f20cf138ef45805416d30a29cd34fccaa6a924649cf0798bc28
benchmark 256 candidates: 33.96–38.63 ms median on verified runs; fingerprint 5fe840bb34bc5d849b6a07c3860e478ec86fbd311d27da81856df7f6200f3431
independent Wave 2H / Wave 3A2 / Wave 3B closures: PASS (`gpt-5.6-terra`)
```

- The latest broad `tests/agent` run reached 7,057 passed at 96.9%, then timed out with five external failures; it is not a clean or complete global gate. The visible failures were outside this slice (Copilot/credential-pool and compression-concurrency). No Dynamic Orchestration focused case failed.

Gitleaks 8.30.1 was run against the exact final staged publishable patch after the last source/docs edits and found no leaks. A repository-wide scan found historical/pre-existing findings and is not attributed to this patch.

## Final expanded independent review — PASS

Terra reviewed the complete publishable delta read-only, including all untracked package/test/benchmark artifacts, canonical specs and the broad-gate logs. It reproduced the frozen hashes, ran the 377-test isolated focused suite, 18 adversarial authority/capacity/boundary regressions, the benchmark, API/node-ID manifest comparisons, import/pickle/static gates and causal analysis of the broad failures.

- **Counts:** `C: 0, H: 0, M: 0, L: 2`.
- **Performance reproduction:** 256-candidate median 35.369491 ms; decision fingerprint and selected route unchanged; billion-sample benchmark input rejected before workload.
- **Security/authority:** no caller-minted dispatch authority, no runtime I/O/wiring, no credential artifact, and no C/H/M regression in the optimized lexical path.
- **Compatibility/test integrity:** 72 root exports, empty required API diff, acyclic imports, legacy pickle round-trip and exact 375-case semantic node-ID multiset.
- **Broad-failure attribution:** the Copilot HOME and credential-pool files are absent from the delta and do not import this package; the compressor passed 214/214 under the larger isolated timeout.
- **Accepted LOW gates:** contain exceptions from hostile custom `Mapping` adapters and add exhaustive graph depth/node/alias/container boundary matrices before any untrusted runtime adapter is wired.

The independent report's final line was `PASS`. This is a PASS for the proposed pure-contract slice only.

## Claude Opus independent review — final current-worktree validation

Claude reviewed the exact current worktree read-only after the package replacement was discovered. It verified that the tracked monolith was an earlier 3,336-line stage, while the untracked package contained the complete 11-module pure-contract implementation and the capacity/execution/reservation/breaker layer. It found no external consumers outside this slice, confirmed the root API as a 72-name superset of the prior public surface, and reproduced package/import/authority/boundary checks.

- Code verdict before cleanup: `PASS`, `C: 0, H: 0, M: 0, L: 1` new dead-code finding plus two accepted residual LOWs.
- L1 correction: removed the dead `_reject_execution_sensitive` function and unused `re` import from `agent/dynamic_orchestration/execution.py`; the live path already used the stricter validator.
- Current focused evidence after cleanup: `377 passed, 0 failed`; Ruff, compileall and `git diff --check` passed.
- The review's publication BLOCK was mechanical: the monolith deletion and replacement package had to be staged together; the stale zero-byte shared-worktree `index.lock` was verified with no Git process and removed. The full replacement is now staged as one coherent diff.
- Final code classification after L1 cleanup: `C: 0, H: 0, M: 0, L: 2` accepted only for hostile custom Mapping adapters and incomplete graph-boundary parameterization while the slice remains draft and unwired.

## Executor LOW-closure lane — 2026-07-27

An executor lane closed the two accepted LOW residuals before any runtime wiring, keeping the pure-contract scope. This is executor evidence; Terra's prior 377-test `PASS` remains prior evidence for the earlier snapshot.

- **L1 closed:** a bounded hostile-safe `_mapping_snapshot` adapter boundary was added to every public `from_mapping` (24 sites) plus the three nested untrusted-mapping reads, and `_reject_sensitive` now contains hostile nested containers. No `str()` coercion; bounds stay fail-closed; ordinary decoded `dict` behaviour is byte-identical (the snapshot returns the same object for a plain `dict`).
- **L2 closed:** exact depth/node `N`/`N+1`, collection-scan, benign DAG-alias and hostile-container regressions were added; no implementation defect was exposed.
- **Focused suite:** grew from 377 to **518 tests**. Parent-verifiable runs are green under default `pytest`, `PYTHONHASHSEED=1`, `PYTHONHASHSEED=99`, and the isolated per-file runner (`HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh …`: 7 files, 518 passed, 0 failed).
- **Static gates:** `python -m py_compile` on all package/test modules, `ruff check` (PLW1514 scope), and `git diff --check` all pass.
- **RED evidence:** temporarily disabling the snapshot and the scanner wrap reproduced 67 leaking L1 assertions; restoring the fix returns to green.
- **Executor recomputed aggregates (basename-NUL-content, sorted):** 11-module package `1840c2f19865881025193ae20215df14b0889c4112016c389d9403410b5f39b1`; 8-file split/support `307dfb812255fe3ac7dfa13f00a1ef0d6d44686ad738617e92a152befa5d7c0f`. These supersede the prior frozen snapshot hashes above, which described the pre-closure source.
- **Single-writer footprint:** only `agent/dynamic_orchestration/{validation,route,task,eligibility,quality,capacity,decision,execution}.py`, `tests/agent/_dynamic_orchestration_support.py`, `tests/agent/test_dynamic_orchestration_boundaries.py`, and this review plus `tasks.md` were touched. `state.py`, `policy.py`, the package `__init__.py`, and the other split test files were not modified.

## Scope boundary

This branch implements pure CapacityView/Reservation/CircuitBreaker/ExecutionEnvelope contracts but does **not** implement provider/gateway dispatch, `CredentialPool.snapshot_for_capacity()`, live observation ingestion, persistent/atomic reservations, breaker storage/transitions, sealed dispatch authority, legacy resolver projection, observability wiring, shadow/canary/rollback activation or runtime E2E. The Hermes PR must remain draft until those layers and their boundary tests exist.

## Parent review of 377→518 hardening — Claude attestation inconclusive

Two bounded Claude Opus review lanes were dispatched for this delta, both read-only. Neither produced a child/tool action or report within the watchdog window; both were terminated without touching the worktree. This is explicitly **not** counted as a Claude PASS.

The parent read the complete 664-line source/test diff and verified the allowlist, uniform `_mapping_snapshot` placement, hostile-hook containment, ordinary dict/OrderedDict/MappingProxyType compatibility, graph edge semantics, and absence of runtime wiring. Parent gates passed twice: 518 tests under default/`PYTHONHASHSEED=1/99`, Ruff, compileall, diff-check and boundary inventory. Parent classification for this mechanical hardening delta: `C: 0, H: 0, M: 0, L: 0`; independent Claude attestation remains inconclusive and runtime tasks remain unchecked.

## Current verdict

**PASS FOR THE PROPOSED PURE-CONTRACT SLICE — NOT PRODUCTION READY.** The prior independent expanded review classified the 377-test snapshot as `C: 0, H: 0, M: 0, L: 2`; the executor lane then closed both accepted LOWs with permanent regressions, and parent verification of the 518-test delta found no new C/H/M/L finding. This is parent-verified hardening, not a new Claude attestation. Performance, package extraction, API compatibility, deterministic test preservation, current broad regression, patch-scoped secret scan and prior expanded review remain documented above. Runtime wiring remains blocked by the explicit unchecked tasks and pre-wiring gates above — those gates are future work, not defects in this slice.
