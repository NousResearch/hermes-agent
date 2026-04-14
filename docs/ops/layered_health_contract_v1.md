# Layered Health Contract v1

Date: 2026-03-21  
Status: active (Wave 8 A6 Ops Reliability Lane)  
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose
Moving beyond a binary up/down check, this contract defines the canonical health states of OpenClaw system lanes. It provides explicit criteria for what constitutes a fully functioning lane, degrading gracefully if specific requirements fail.

## 0.1) EX-05 bounded MVP lane set (runtime-enforced)

The current runtime snapshot (`ops/openclaw/continuity/layered_health_snapshot.sh`) enforces a bounded multi-lane set for truthfulness core rollout safety:

- `A1_CONTROL_PLANE`
- `A2_RUNTIME_CONTINUITY`
- `A3_MODEL_ROUTING`
- `A6_OPS_OBSERVABILITY`
- `C1_OPERATOR_SURFACE`
- `C2_RELEASE_SUBSTRATE`

Release-coupled observability gates consume these per-lane rows from `state/continuity/latest/layered_health_snapshot.json` and fail closed when required lanes are missing, below the minimum required layer, or non-pass.

## 1) Health Layers

A lane must satisfy all requirements of a layer to claim it, and must satisfy all lower layers to achieve a higher one.

### Layer 1: Alive
- **Meaning:** The process, script, or primary daemon is running.
- **Criteria:** OS-level process check passes, or the primary endpoint returns any response (e.g., HTTP 200 on `/health`, even if degraded).
- **Failure:** Process crash, timeout, host unreachable.

### Layer 2: Ready
- **Meaning:** Dependencies are met and the lane can process work.
- **Criteria:** Database connections are established, required credentials/tokens are valid, required upstream services are reachable.
- **Failure:** Auth expiry, DB connection refused, upstream timeout.

### Layer 3: Safe-to-Act
- **Meaning:** Pre-flight guards and constraints permit mutation.
- **Criteria:** Gate logic (e.g., `verify_then_resume.sh`) returns `allowed`, rate limits are not exhausted, error budgets are intact.
- **Failure:** Mutation gate returns `forbidden` or `caution`, error budget exhausted, explicit block via operator mission-control.

### Layer 4: Truthful
- **Meaning:** The lane's state and outputs are not drifting from reality.
- **Criteria:** Telemetry/snapshots are fresh (age < threshold), continuity bridges match ground truth, no staleness warnings.
- **Failure:** Telemetry is stale, bridge pointers mismatch, ground truth drift detected.

## 2) Status Aggregation

A lane's overall health is expressed as a combination of its highest achieved layer and a status:
- **`pass`**: All layers achieved.
- **`degraded`**: `alive` and `ready` achieved, but `truthful` is failing (can still act, but blindly or with stale context).
- **`failing`**: Failed `alive`, `ready`, or `safe-to-act` (mutation blocked).

## 3) Enforcement
Observability gates MUST parse this contract's output schema (`layered_health_snapshot.schema.json`) and halt Wave rollouts or block autonomous mutators if the required lane fails to reach `safe-to-act`.

For C2 release ladder coupling, default requirement is stricter: required lanes above must be present and `truthful`, and restore freshness (`SLO-4_RESTORE_DRILL_FRESHNESS`) must be `pass`.

## XR-007 promoted asset checklist (Ops/A6 canonical promotion pack)

Archive/runtime asset promoted to canonical A6 checklist:

1. **Multi-host jitter resilience harness + guard**
   - Guard runtime: `ops/openclaw/continuity/multi_host_failover_guard.py`
   - Harness runtime: `ops/openclaw/continuity/a6_multi_host_jitter_harness.py`
   - Runtime latest artifact: `state/continuity/latest/a6_multi_host_jitter_resilience_evidence.json`
   - Slice evidence: `reports/core_roadmap_slice18_strict_closeout_2026-03-28.md`
   - Required verification refs:
     - `tests/test_multi_host_failover_guard.py`
     - `tests/test_a6_multi_host_jitter_harness.py`
     - `tests/test_verify_then_resume_a6_observability_gate.py`

Promotion rule (fail-closed): A6 multi-host reliability claims require deterministic jitter harness evidence proving degraded jitter resilience and sustained-failure escalation behavior.
