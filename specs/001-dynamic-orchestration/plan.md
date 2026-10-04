# Plan: Dynamic Multimodel Orchestration

Status: PROPOSED — phase-1 contracts only; no runtime activation.

## Target flow
`TaskEnvelope -> deterministic eligibility -> score eligible Routes -> atomic reservation -> ExecutionEnvelope -> independent verification -> reconciliation`.

`CredentialPool` remains the only mutable SSOT for credential/cooldown/lease state. A future snapshot seam locks, copies and sanitizes only required state; `CapacityView` is immutable/read-only and cannot call `select()`, `_available_entries(refresh=True)`, `_available_entries(clear_expired=True)`, `_persist()`, OAuth or network. A no-argument `_available_entries()`/`has_available()` is allowed only under lock after focused tests prove it is read-only and cannot refresh, clear expiry, persist, call OAuth/network, or mutate. Reservation is distinct from credentials and route choice.

## Phases
0. Baseline/contracts: line-level evidence, schemas and tests only.
1. Pure contracts: envelopes, state machines, RouteV1 canonicalization, provenance and redaction.
2. Pure policy/reservation: deterministic filters, scores, breakers and leases; no dispatch.
3. Compatibility/shadow: calculate alongside legacy `resolve_runtime_provider`; compare CLI/gateway/cron/delegation with zero added provider calls.
4. Canary: feature-gated, low-risk reversible tasks, explicit decision application and rollback rehearsal.
5. Gated rollout: human-approved only after shadow/canary quality/cost evidence.
6. Telemetry activation (separate gate): default-disabled consent/config/setup/tools path and scoped-event tests before any telemetry emission.

## E0–E4 / V0–V4
E0 deterministic/no model/no route; E1 narrow mechanical; E2 normal work; E3 complex cross-surface; E4 high-risk/irreversible-impact with human gate. V0 contract evidence; V1 executor artifact; V2 independent validator; V3 independent high-capability review; V4 human/external authority.

## Route algorithm
1. Normalize each candidate according to `RouteV1` in `data-model.md`, serialize with canonicalization version `route-v1`, and validate against `fixtures/route-v1/`.
2. Build immutable CapacityView from verified observations plus pure pool snapshot.
3. Set `derived_remaining` to `CapacityObservation.value` only for a fresh, unit-compatible `metric=remaining`; otherwise it is unknown. Evaluate `dispatchable=derived_remaining-active_reservations-protected_reserve-safety_margin` only for compatible fresh units.
4. Filter before score: policy, privacy/permission, capabilities/tools, context, freshness/confidence, budget, breaker/cooldown, concurrency/reservation compatibility. Unknown capacity remains unknown: high-risk fails closed; lower-risk must persist explicit policy disposition/evidence and never relabel it healthy, zero or unlimited.
5. Atomically reserve selected route/pool; on conflict recompute.
6. Persist RouteDecision before dispatch; adapter cannot recompute route. A fallback decision must pass `QualityCompensationPlanV1` enforcement or remain activation-blocked.
7. Dispatch exact ExecutionEnvelope, verify requested/resolved/effective identity and compensation thresholds, reconcile reservation.

## `route_capacity_exhausted`
**PROPOSED typed mapping:** `RuntimeErrorClassificationV1.kind=capacity_exhausted` maps to `route_capacity_exhausted` only from the attempted normalized route ID and quota/billing pool scope; it never maps from a provider/model label alone. The observed runtime `FailoverReason` has no typed mapping. **DECISION REQUIRED before wiring:** approve the exact classification table, evidence threshold, retryability and breaker policy; ambiguous/unmapped failures do not become capacity exhaustion.

Fail attempt -> invalidate scoped observation -> cooldown/breaker -> reconcile/release reservation -> authorized refresh -> recompute candidates -> explicit fallback RouteDecision -> validate QualityCompensationPlanV1 -> reserve -> dispatch -> verify. `WAITING_FOR_CAPACITY` only if no eligible route remains or policy rejects all. If the attempted Opus route exhausts and an eligible GPT route `{provider=openai, quota_pool=openai-team-a}` or GLM route `{provider=zai, quota_pool=zai-team-a}` remains, replan; never schedule a sole wait job.

## Observed integration targets (future)
- `runtime_provider.py:1542-1548` is dict-only resolution; `:1743-1749` selects a pool credential.
- `gateway/run.py:2236-2272` and `:2346-2393` do fallback on errors.
- `cron/scheduler.py:3230-3410` resolves/loops configured fallback and builds AIAgent.
- `delegate_tool.py:1066-1098`, `:1431-1437`, `:1818-1828` acquires credential leases, and `:2371-2375` releases them; those leases are not route reservations.
- `run_agent.py:425-497` accepts fallback/pool; it is execution consumer, not policy authority.

## Compatibility, canary, rollback
Legacy executes during shadow. Canary is a per-task feature gate. Rollback disables decision application, stops canary dispatch, releases reservations and preserves sanitized evidence; it routes through legacy without corrective deployment. Telemetry remains default-disabled until the separate consent activation gate is passed; it is opt-in/best effort and cannot block execution.

## Gates
G1 schema/property/contract including RouteV1 fixtures and insufficient-compensation rejection; G2 shadow parity/no extra calls; G3 concurrency/chaos including unknown-capacity policy branches; G4 canary+rollback; G5 human activation; G6 telemetry consent activation (config/setup/tools path, default-disabled/no-consent, bounded sanitized post-consent events). Paid routes, config/service/credential changes and remote calls require separate authorization.
