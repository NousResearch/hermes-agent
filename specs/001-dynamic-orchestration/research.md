# Research and Evidence Register

Status: source-backed baseline. No source means production wiring.

| Label | Finding | Evidence | Consequence |
|---|---|---|---|
| OBSERVED | Resolver returns runtime dict, no task/decision/reservation. | `hermes_cli/runtime_provider.py:1542-1548` | Additive typed decision plus compatibility projection. |
| OBSERVED | Resolver calls `pool.select()`. | `runtime_provider.py:1743-1749` | Separate policy/route/reservation from credential materialization. |
| OBSERVED | Pool is mutable, locked/persistent; availability may refresh/prune/persist. | `credential_pool.py:582-614`, `632-637`, `1563-1696` | Pool is mutable SSOT; view requires pure snapshot seam. |
| OBSERVED | Delegation builds children, shares pool, acquires and releases credential leases. | `delegate_tool.py:1066-1098`, `1431-1437`, `1818-1828` (acquire), `2371-2375` (release) | Existing credential lease is not route reservation. |
| OBSERVED | Cron primary/fallback loop and drift guard. | `scheduler.py:3230-3293`, `3302-3356`, `3358-3410` | Preserve guard; replace implicit fallback only after shadow. |
| OBSERVED | Gateway catches auth/rate limit then loops fallback config. | `gateway/run.py:2236-2272`, `2346-2393` | Capture trigger/candidates/rejections explicitly. |
| OBSERVED | AIAgent accepts fallback_model/credential_pool. | `run_agent.py:425-497` | Execution consumer, not policy authority. |
| OBSERVED | Prior foundation is not consumed by runtime/adapters. | Pandora audit `08-dynamic-runtime-audit-2026-07-26.md:61-65`; prior `spec.md:14-24` | Reuse concepts only; do not claim wiring. |
| PROPOSED | Normalized Route across provider/product/surface/account/entitlement/billing/quota/model/endpoint/variant/region. | Pandora target `09-dynamic-target-architecture.md:42-54` | Prevents pool collapse. |
| PROPOSED | Provenance-rich observation, policy-before-score and explicit fallback. | `09...:42-80` | Contracts implement this. |
| PROPOSED | Shadow/canary/rollback migration. | Pandora migration `10-dynamic-migration-plan.md:47-87` | No activation here. |
| DECISION REQUIRED | Unknown-capacity policy by risk/cost. | `09...:74-80` | Version before canary. |
| DECISION REQUIRED | Reservation backend/TTL/ownership; fallback quality compensation. | `10...:40-45`, `09...:62-68` | Required before activation. |

## Uncertainties
- `hermes_otel.py` was requested but not found by current-worktree file search. No implementation claim is made; telemetry is only proposed.
- Exact current source path that normalizes `route_capacity_exhausted` needs later tracing.
- No remote provider was called; live plan/quota/reset/price are unknown.
