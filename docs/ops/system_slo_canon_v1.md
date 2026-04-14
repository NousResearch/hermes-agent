# OpenClaw System SLO Canon v1

Date: 2026-03-21
Status: active (Wave 8 A6 Ops Reliability Lane)

## 0) Purpose
Define canonical Service Level Objectives (SLOs) for OpenClaw's internal systems. This is an explicit, machine-readable declaration of performance bounds to ensure reliable orchestration.

## 1) SLO Definitions

### SLO-1: Verification Evidence Freshness
- **Target:** `state/continuity/latest/verify_last.json` age is `< 30 minutes` (default 1800s; configurable via `OPENCLAW_SLO_VERIFY_MAX_AGE_SEC`).
- **Reason:** Mutation decisions should not rely on stale verification evidence.

### SLO-2: Continuity Freshness
- **Target:** Ground truth drift checkpoint age is `< 1 hour` (default 3600s; configurable via `OPENCLAW_SLO_GROUND_TRUTH_MAX_AGE_SEC`).
- **Reason:** The main system should not act blindly on data older than an hour.

### SLO-4: Restore Drill Freshness
- **Target:** `state/continuity/latest/restore_drill_latest.json` evidence age is `< 7 days` (default 604800s; configurable via `OPENCLAW_SLO_RESTORE_DRILL_MAX_AGE_SEC`).
- **Reason:** Rollout and mutation safety depends on recent, explicit restore evidence proving fallback viability.

### Deferred SLO candidates (declared, not yet gate-enforced)

#### SLO-3: Queue Arbitration Cycle
- **Target candidate:** `queue_arbitrator` completes a cycle without blocking/locking issues in `< 5 minutes`.
- **Reason:** Ensuring that the sub-agent and orchestration queues aren't deadlocked.
- **Runtime status:** deferred until dedicated metrics surface is wired into `slo_snapshot.json`.

#### SLO-5: Web Capture Runtime Reliability
- **Target candidate:** `web_capture_macro` executes successfully with a failure rate `< 5%` over 24h.
- **Reason:** Critical downstream feeds depend on web data extraction.
- **Runtime status:** deferred until rolling failure-rate accounting is available in continuity latest surfaces.

## 2) Error Budget Policies
- Runtime-enforced today: **SLO-1 + SLO-2 + SLO-4** (via `slo_evaluator_snapshot.sh` + rollout gate consumption).
- When an enforced SLO error budget is depleted, **mutation operations** that depend on that SLO enter a restricted `caution` state.
- Rollouts of new lanes or architectural upgrades are blocked until enforced SLOs recover.
