# Data Model: Dynamic Multimodel Orchestration

All types are PROPOSED unless stated OBSERVED. Contract boundaries are versioned, bounded, sanitized, immutable and secret-free. `unknown` is explicit.

## Common provenance
Every runtime datum for plan/product/entitlement/model/quota/reset/price/concurrency/capability: `source`, `captured_at`, `valid_until`, `freshness(fresh|stale|unknown)`, `confidence(0..1|unknown)`, `is_estimated`. Missing value remains unknown.

## Route and canonical identity
`RouteV1 {canonicalization_version:"route-v1", provider, product, surface, account_id, entitlement_id, billing_pool_id, quota_pool_id, model, endpoint, variant, region, route_id}`.

`RouteV1` serialization is normative:
1. Required field order is exactly: `canonicalization_version`, `provider`, `product`, `surface`, `account_id`, `entitlement_id`, `billing_pool_id`, `quota_pool_id`, `model`, `endpoint`, `variant`, `region`.
2. String values are Unicode NFC-normalized, trimmed of leading/trailing ASCII whitespace, and case-folded. Empty strings normalize to absent. Identifiers MUST NOT apply locale-dependent casing.
3. `endpoint` is parsed as an absolute URL; scheme and host are case-folded; default ports (`https:443`, `http:80`) are removed; dot-segments are resolved; percent-encoding uses uppercase hex and decodes only unreserved characters; no query or fragment is permitted. The normalized URL has no trailing slash except root `/`.
4. Each absent optional field serializes as JSON `null`; a supplied JSON `null` is identical to absent. No field may be omitted from the canonical object. `canonicalization_version` is always the literal `"route-v1"`.
5. Serialize the ordered object as compact RFC 8259 JSON with property order from step 1, UTF-8 encoding, `ensure_ascii=false`, no insignificant whitespace, and no non-finite numbers. `route_id` is lowercase hexadecimal SHA-256 of those UTF-8 bytes, prefixed `route-v1:`.

Fixtures are versioned under `fixtures/route-v1/`: `equivalent-unicode-case-url.json` proves equivalent NFC/case/default-port/URL forms hash identically; `distinct-pools.json` proves distinct billing/quota pools hash differently; `absent-null.json` proves absent and null match; `different-inputs.json` proves a changed canonical field changes the hash. Implementations in every process/runtime MUST pass the identical fixture corpus before use.

## TaskEnvelope
Contains IDs/objective/deliverables, required capabilities, allowed tools, permissions, bounded context, privacy, risk/reversibility, E0–E4, budget, V0–V4 review policy, policy version and optional audited model justification. It does not identify a model/route unless justification contains policy/version/reason/evidence/author/expiry.

## Independent state machines
Task: NEW->PLANNED->ROUTED->DISPATCHED->VERIFYING->COMPLETED|FAILED|WAITING_FOR_CAPACITY|CANCELLED.
Attempt: CREATED->RESERVED->DISPATCHED->RUNNING->RESULT_RECORDED->VERIFIED|FAILED|CANCELLED.
Route: DISCOVERED->ELIGIBLE|INELIGIBLE|COOLDOWN|BREAKER_OPEN|RETIRED.
Credential: existing CredentialPool-owned availability/exhausted/dead state.
Reservation: PENDING->HELD->CONSUMED|RELEASED|EXPIRED|RECONCILED.
Review: NOT_REQUIRED|PENDING|IN_PROGRESS|PASSED|FAILED|HUMAN_APPROVED|HUMAN_REJECTED.
All transitions have actor/timestamp/reason/correlation IDs; no state machine infers another.

## Capacity/reservation/breaker
`CapacityObservationV1` references route/pools, metric/value/unit and common provenance. `CapacityViewV1` is sorted `PoolCapacityV1` from a pure sanitized CredentialPool snapshot. `CapacityReservationV1` has route/quota/billing pool, owner attempt, unit, estimated/held amount, expiry/status/version. `derived_remaining` is defined only for `metric=remaining` with a fresh, compatible unit and equals `CapacityObservationV1.value`; otherwise it is `unknown`. `dispatchable=derived_remaining-active_reservations-protected_reserve-safety_margin` only for compatible fresh units. `CircuitBreakerV1` is route/quota-pool scoped with closed/open/half-open, cooldown and probe budget.

## Decisions/execution
`RuntimeErrorClassificationV1{kind,source,evidence_code?,attempted_route_id,quota_pool_id,billing_pool_id?,classified_at}` is PROPOSED. It maps `kind=capacity_exhausted` to the route/pool-scoped reason `route_capacity_exhausted` using the attempted normalized route, not provider/model names. The current runtime `FailoverReason` lacks this typed mapping. **DECISION REQUIRED before wiring:** the exact classification table, evidence threshold, retryability and breaker behavior; ambiguous/unmapped errors do not become capacity exhaustion. `QualityCompensationPlanV1` is defined in `contracts/quality-compensation.md`; every fallback decision has a valid plan or remains `ACTIVATION_BLOCKED_QUALITY_COMPENSATION`. RouteDecision records task/attempt/policy/router/view/candidates/rejections/selected route/E/V/reservation/scores/reasons/fallback/compensation. ExecutionEnvelope records decision/route/reservation, permitted context/tools, budget/timeout, requested/resolved/effective identity, results/errors/reconciliation/review and an optional active telemetry consent ID. Both omit prompts, credentials and raw tool/provider payloads.

## Invariants
CredentialPool alone mutates credentials; view is pure/no-I/O; unknown never becomes healthy/zero/infinite; reservation does not select routes; adapter cannot change persisted decision; fallback/wait are explicit transitions; an insufficient compensation plan cannot reserve or dispatch.
