# Spec: Dynamic Multimodel Orchestration

Status: PROPOSED — phase-1 contracts only. No runtime activation is implied.

## Purpose
Define the source-of-truth contracts for dynamic, policy-governed multimodel orchestration. A task is classified by work and constraints, then routes are deterministically filtered, scored, reserved, dispatched, and independently verified. This package does not wire any production entrypoint, mutate config/credentials, call a provider, or create a new core model tool.

## Epistemic labels
- **OBSERVED**: current checkout behavior verified by file and line.
- **PROPOSED**: phase-1 contract; not wired or active.
- **DECISION REQUIRED**: a human policy/gate must decide before activation.
- **UNVERIFIED EXTERNAL**: assertion/evidence outside the current reviewed source set; it must not justify a current-runtime claim.

## Observed baseline
- **OBSERVED** `hermes_cli/runtime_provider.py:1542-1548` exposes `resolve_runtime_provider(requested, explicit_api_key, explicit_base_url, target_model)` and returns a runtime dictionary; it has no TaskEnvelope, RouteDecision, reservation, verification or route identity.
- **OBSERVED** `runtime_provider.py:1743-1749` loads a CredentialPool and calls `pool.select()` in resolution. Selection is execution-side mutable state, not an audited route decision.
- **OBSERVED** `agent/credential_pool.py:582-614` owns mutable provider entries, lock and active lease state; `:632-637` persists entries; `:1563-1565` selects under its lock; `:1567-1696` may sync, refresh, prune and persist while determining availability.
- **OBSERVED** `tools/delegate_tool.py:1066-1098` builds child AIAgents from optional model/provider overrides; `:1320-1324` inherits a parent fallback chain; `:1431-1437` shares a credential pool; `:1818-1828` acquires credential leases and `:2371-2375` releases them. It does not create a typed route decision.
- **OBSERVED** `cron/scheduler.py:3230-3293` resolves a primary then iterates configured fallback entries following auth failure; `:3302-3356` prevents unpinned provider/model drift; `:3358-3410` separately obtains a fallback chain/pool and constructs AIAgent.
- **OBSERVED** `gateway/run.py:2236-2272` resolves a runtime then catches auth/rate-limit errors; `:2346-2393` iterates configured fallback entries. Its fallback is not an explicit auditable decision.
- **OBSERVED** `run_agent.py:425-497` accepts `fallback_model` and `credential_pool` directly in AIAgent construction. It is an execution surface, not the policy authority.
- **OBSERVED** `hermes_otel.py` was not found in this checkout by the file search. No telemetry behavior is claimed from it; any instrumentation is PROPOSED and activation-gated.
- **UNVERIFIED EXTERNAL** Pandora audit `08-dynamic-runtime-audit-2026-07-26.md:61-65` reportedly states prior model-routing-remediation contracts are not consumed by runtime_provider/adapters. This package does not use that assertion to claim current runtime wiring.

## Functional requirements
FR-01. **PROPOSED** Every discovered runtime datum (plan, product, entitlement, quota, reset, price, model capability, concurrency) SHALL carry `source`, `captured_at`, `valid_until`, `freshness`, `confidence`, and `is_estimated`. Missing data remains `unknown`; it is never inferred from configuration, HTTP success, absence of error, or documentation.

FR-02. **PROPOSED** Task, attempt, route, credential, reservation, and review states SHALL be separate state machines with separate IDs and transitions. No attempt state may be inferred from credential/capacity state.

FR-03. **PROPOSED** A normalized Route SHALL include provider, product, surface, account, entitlement, billing_pool, quota_pool, model, endpoint, variant, and region. `RouteV1` SHALL use the versioned canonical serialization algorithm and fixture corpus in `data-model.md`; equal model names in distinct pools produce distinct route IDs.

FR-04. **PROPOSED** CredentialPool remains mutable SSOT for credential/cooldown/lease state. CapacityView is a pure read-only snapshot projection: no select, refresh, persistence, OAuth, network, or mutation.

FR-05. **PROPOSED** TaskEnvelope declares capabilities, risk, allowed tools, permissions, budget, context bounds, privacy classification, and review need. It MUST NOT select or imply a particular model unless an audited policy justification is supplied.

FR-06. **PROPOSED** Deterministic eligibility filters run before scoring: policy, permissions, privacy, required capability/tool, context, freshness/confidence, budget, breaker, concurrency, and reservation compatibility. Scoring only ranks eligible routes.

FR-07. **PROPOSED** Fallback is a new, explicit RouteDecision with policy status, typed trigger, candidate set, per-candidate rejections, rationale and mandatory `QualityCompensationPlanV1` (`contracts/quality-compensation.md`). An absent/insufficient plan is `ACTIVATION_BLOCKED_QUALITY_COMPENSATION` and cannot reserve or dispatch. A configured provider chain is candidate input, never route truth.

FR-08. **PROPOSED** `RuntimeErrorClassificationV1.kind=capacity_exhausted` maps to `route_capacity_exhausted` only with the attempted normalized route ID and quota/billing pool scope; it MUST NOT be inferred from a provider/model name. The current runtime `FailoverReason` lacks this typed mapping. `route_capacity_exhausted` invalidates the affected route/pool view, opens or updates cooldown/breaker as policy permits, reconciles/releases reservation, refreshes eligible evidence, recomputes candidates, records a fallback decision, validates its compensation plan, dispatches it and verifies execution. `WAITING_FOR_CAPACITY` is valid only when no route remains eligible or policy rejects all candidates.

FR-09. **PROPOSED** Effort is E0–E4; verification is V0–V4. Adapters translate only controls they actually support and report unsupported controls as unknown/unattested.

FR-10. **PROPOSED** Legacy resolver compatibility is additive: shadow first, then bounded canary, then gated cutover. Rollback disables the new decision application and releases reservations; it never needs a corrective deployment to restore legacy routing.

FR-11. **PROPOSED — activation gate** Telemetry is default-disabled and may emit no event without persisted user consent. A future activation must offer a user-facing `config.yaml` gate, setup prompt and `hermes tools` toggle; persist the consent scope/activation/revocation state; and limit scope to this orchestration event family. Only after consent, events contain IDs, canonicalization/policy versions, normalized/sanitized metadata and bounded reason codes; never prompts, raw tool contents, API keys, tokens or credentials. Telemetry failure cannot block dispatch or verification. This package makes no claim that such a path is currently implemented.

## Acceptance scenarios
AC-01 Unknown capacity: Given a route has no attested quota, when policy evaluates it, then capacity remains `unknown` and is never healthy, zero or unlimited; high-risk policy fails closed without reservation/dispatch, and lower-risk policy persists an explicit policy disposition and its evidence before any allowed continuation.

AC-02 Pool isolation: Given identical `model` values but distinct quota pools, when routes are canonicalized, then their route IDs differ and exhaustion in one does not change the other.

AC-03 Opus exhaustion: Given an attempted Opus route emits `route_capacity_exhausted` while GPT route `{provider=openai, quota_pool=openai-team-a}` or GLM route `{provider=zai, quota_pool=zai-team-a}` is eligible, when the failed attempt is reconciled, then the system emits a fallback RouteDecision with valid compensation and replans/dispatches; it MUST NOT schedule a sole wait job.

AC-04 Sole wait: Given all candidates are in cooldown, unknown under a fail-closed policy, or otherwise rejected, when candidate recomputation ends, then and only then task state may become `WAITING_FOR_CAPACITY` with next-recheck evidence.

AC-05 Privacy: Given telemetry is enabled by persisted scoped consent, when an execution ends, then stored event payload contains no prompt, secret, credential, authorization header, raw tool result or unbounded error body. Without consent, no event is emitted.

## Non-goals
Runtime wiring; config changes; provider discovery calls; automated price/plan inference; model-specific tool creation; changing fallback chains; changing services/credentials; rollout; commit/push/PR.

## Decisions required
- Activation policy for unknown capacity by risk class, including lower-risk explicit disposition rules.
- E1–E4 mapping and provider-specific controls.
- V1–V4 independence/quorum policy and QualityCompensationPlan acceptance thresholds.
- Reservation storage/transaction boundary and pool ownership rules.
- Exact `RuntimeErrorClassificationV1` mapping table from provider/runtime failures to route/pool-scoped `route_capacity_exhausted`, including evidence threshold, retryability and breaker policy; the current `FailoverReason` does not supply it.
- Telemetry consent text, retention, revocation semantics and scoped event schema before activation.
