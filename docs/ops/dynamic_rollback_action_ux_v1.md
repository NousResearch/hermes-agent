# Dynamic Rollback Action UX v1

Date: 2026-03-21
Status: active (Wave 8 C2 DevEx Substrate)
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose
To close the loop between C2 (Release/DevEx substrate) and C1 (Operator Cockpit). When a Release Ladder gate halts due to an Error Budget breach (A6), the operator should not be forced to piece together the rollback command.

## 1) Ergonomics Principle
If `release_evidence_ladder_gate.py` fails due to `a6_observability_failed`, the `cockpit_summary.sh` will parse this context.

## 2) Remediation Hint Overload
When the Cockpit generator sees an active release failure tied to an SLO or Health check, it overrides the standard `blindness_recovery` hint with the explicit rollback path:

**Immediate Action:**
`Run: bash ops/openclaw/continuity/verify_then_resume.sh --run-rollback`

This ensures the operator's first response to a breached rollout is to revert to a safe `chk_latest` checkpoint.
