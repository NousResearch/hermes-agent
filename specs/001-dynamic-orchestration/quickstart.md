# Quickstart: Future Contract Consumer

> **FUTURE WORKFLOW ONLY — NOT WIRED.** This package documents PROPOSED contracts only. It does not activate runtime behavior, provider calls, configuration, services, or credentials.

Documentation only. Do not enable runtime from this package.

1. Build TaskEnvelope without provider/model identity.
2. Gather authorized observations with full provenance; unavailable values are unknown.
3. Build pure secret-free CapacityView from locked pool snapshot; do not select/refresh/persist/network.
4. Canonicalize distinct Routes including pool/account/surface/endpoint.
5. Filter deterministically, record rejections, then score eligible only.
6. Reserve atomically; persist RouteDecision; create ExecutionEnvelope; dispatch; verify; reconcile.

Capacity exhaustion: an attempted Opus route emits `route_capacity_exhausted` -> invalidate -> cooldown/breaker -> release/reconcile -> refresh -> recompute -> fallback decision -> reserve/dispatch/verify. If an eligible GPT route `{provider=openai, quota_pool=openai-team-a}` or GLM route `{provider=zai, quota_pool=zai-team-a}` remains, replan; never sole-wait. Wait only if no eligible route/policy rejects all. `route_capacity_exhausted` is a PROPOSED typed mapping from `RuntimeErrorClassificationV1.kind=capacity_exhausted` using attempted route/pool scope; the exact mapping policy is DECISION REQUIRED before wiring.

Before activation run unit/property, contract no-I/O, entrypoint shadow integration, reservation concurrency/chaos, temp-HERMES_HOME E2E, legacy/shadow/canary/rollback and exact Opus-vs-GPT/GLM regression. No real HERMES_HOME/provider/service/credential.
