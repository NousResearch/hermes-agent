# Requirements Quality Checklist: Dynamic Multimodel Orchestration

**Purpose:** Validate that the specification is complete, unambiguous, testable, and safe before runtime activation. Checked items assess the requirement text and committed contract package; they do **not** imply runtime wiring or production readiness.

## Scope and epistemic integrity

- [x] The specification states that phase 1 is contract-only and does not imply runtime activation.
- [x] OBSERVED, PROPOSED, DECISION REQUIRED, and UNVERIFIED EXTERNAL claims are distinguished.
- [x] Current-runtime observations identify concrete source locations.
- [x] Runtime wiring, provider calls, config mutation, credential mutation, rollout, and production activation are explicit non-goals of the current contract phase.
- [x] The implementation plan preserves a legacy → shadow → canary → gated-cutover sequence.
- [ ] Every source location in the observed baseline has been revalidated against the final rebased upstream commit immediately before PR review.

## Task, attempt, route, and capacity semantics

- [x] Task identity is independent from model, provider, account, credential, and execution attempt.
- [x] Task, attempt, route, credential, reservation, and review use separate state machines and identifiers.
- [x] Route identity includes provider, product, surface, account, entitlement, billing pool, quota pool, model, endpoint, variant, and region.
- [x] Same model names in different billing/quota pools produce different route identities.
- [x] Unknown, stale, estimated, and fresh capacity states are distinguished.
- [x] Missing capacity is never interpreted as healthy, zero, or unlimited.
- [x] CredentialPool remains the mutable credential/lease SSOT while CapacityView is specified as a pure read-only projection.
- [x] Route reservation and credential lease are explicitly separate concepts.
- [ ] The policy owner has approved the final unknown-capacity behavior for every risk tier.
- [ ] The reservation persistence/transaction boundary and ownership rules have been approved.

## Selection, fallback, and recovery

- [x] Deterministic policy/eligibility gates precede scoring.
- [x] Candidate rejection reasons and the selected-route rationale are auditable.
- [x] A static provider/model fallback chain is candidate input, not route truth.
- [x] `route_capacity_exhausted` is scoped to normalized route and quota/billing-pool evidence.
- [x] Capacity exhaustion requires reservation reconciliation/release before a replacement attempt.
- [x] An eligible alternative produces an explicit fallback/replan decision rather than a sole wait job.
- [x] `WAITING_FOR_CAPACITY` is allowed only when no route remains eligible or policy rejects all alternatives, and requires recheck evidence.
- [x] Quality compensation and independent verification requirements are represented in the decision contract.
- [ ] The provider/runtime error-to-`RuntimeErrorClassificationV1` mapping table, evidence threshold, retryability, and breaker policy have been approved.
- [ ] Circuit-breaker half-open/probe budgets and reset semantics have been approved.

## Security, privacy, and cost

- [x] Public contracts prohibit prompts, credentials, authorization material, raw provider/tool payloads, and unbounded error bodies.
- [x] Paid-route eligibility is policy/budget gated rather than inferred from credential presence.
- [x] Telemetry is default-disabled and requires persisted, scoped, revocable consent before emitting events.
- [x] Telemetry failure cannot block dispatch, verification, or recovery.
- [x] Rollback returns execution to the legacy resolver without a corrective deployment.
- [ ] Telemetry consent text, retention period, revocation semantics, setup flow, config key, and `hermes tools` control have been approved.
- [ ] Secret scanning has passed on the final staged diff with a robust scanner or an explicitly documented compensating gate.

## Verification and rollout

- [x] Unit, property, contract, integration, concurrency/chaos, and temp-`HERMES_HOME` E2E layers are mapped to requirements.
- [x] The exact Opus-exhausted/GPT-or-GLM-eligible regression is specified.
- [x] Pool-isolation and last-unit concurrent reservation regressions are specified.
- [x] Requested/resolved/effective route attestation is required before claiming provider/model execution.
- [x] Independent review requirements are distinct from executor self-assessment.
- [ ] Pure-contract tests, default lint, compile, complexity, secret scan, and diff checks pass on the final staged implementation.
- [ ] Shadow parity demonstrates no extra provider call and no dispatch behavior change.
- [ ] Gateway, CLI, cron, and delegation entrypoint integration tests pass in isolated temp homes.
- [ ] Canary cohorts, error/cost budgets, kill switch, and rollback have been exercised.
- [ ] A human activation gate has approved production cutover.

## Final acceptance rule

The feature cannot be labeled GREEN for runtime activation while any unchecked activation, security, integration, canary, rollback, or human-gate item remains. A contract-only PR may be accepted independently when its own scoped gates pass and it makes no wiring or production claim.
