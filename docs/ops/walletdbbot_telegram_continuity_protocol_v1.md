# WalletDBBot Telegram Continuity Protocol v1

Status: active support protocol
Date: 2026-04-02
Scope: `@walletdbbot` / Telegram direct-lane orchestration continuity and successor-safe recovery

## Problem family
`@walletdbbot` failures in this cluster should be treated first as **OpenClaw Telegram direct-session/runtime overload** problems, not archived WalletDB backend corruption.

Recurring indicators:
- direct session context bloat / overflow
- `readiness=RECONCILE_REQUIRED`
- `mutation_gate.status=forbidden`
- successor proof degraded/refused (`reset_allowed=false`, `resume_allowed=false`)
- Telegram backlog / `429 Too Many Requests`
- misleading long-lived typing indicators
- stale or drifted checkpoint / handover / pointer surfaces

## Recovery goal
Reach a successor-safe posture where all of the following are true:
- `verify_last.status=READY`
- `mutation_gate.status=allowed`
- `proof_state=PROOF_VALID_PASS`
- `reset_allowed=true`
- `resume_allowed=true`
- handover/latest surfaces align with the fresh proof

## Operating doctrine
- Keep the main/orchestrator session lean.
- Do not push heavy orchestration back onto the Telegram direct lane.
- Treat Telegram transport viability and Telegram session-runtime viability as separate questions.
  - Direct send may still work while orchestration turns are wedged.
- Preserve successor comprehension through explicit handover artifacts, not giant always-injected session bootstrap.

## Minimal recovery sequence
1. Refresh continuity truth surfaces:
   - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json`
   - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/reset_ready_refresh.sh --json`
2. Read verify-gate preflight:
   - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh verify-gate-status --json`
3. Refresh failover-runtime evidence:
   - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh failover-stress-runtime-evidence --cycles 2 --json`
4. If continuity is still not ready, classify the remaining blocker bucket:
   - drift-only -> use guarded reconcile path
   - observability/routing/restore-drill blocker -> refresh evidence and rerun verify gate
   - authority-gated mutation blocker -> obtain the required mutation ticket / attestations before forcing reconcile

## Current known hard blockers in this family
### 1) Restore-drill evidence missing
Layered health will not pass while `state/continuity/latest/restore_drill_latest.json` is missing or stale.
Reference playbook: `docs/ops/incident_playbooks/restore_drill.md`

### 2) Routing preflight stale
`verify-gate-status` can stay red when session-routing evidence is stale. A fresh lint snapshot is useful diagnostic evidence, but a stale decision log remains a blocker until the routing evidence path is refreshed to policy-compliant current state.

### 3) High-risk reconcile is authority-gated
`reconcile.sh` is enforced as high-risk and currently requires:
- action token
- mutation ticket
- attestations including:
  - `replay_evidence_pass`
  - `schema_contract_pass`
  - `verify_before_resume_pass`

## Successor-safe handover requirements
A successor should be able to continue without reading the entire repo from scratch. At minimum, preserve or regenerate:
- `state/continuity/current.json`
- `state/continuity/latest/reset_ready_refresh_latest.json`
- `state/continuity/latest/successor_safe_handover_proof.json`
- `state/continuity/latest/successor_safe_handover_proof_status.json`
- `state/handover/latest.json`
- `state/handover/latest.md`
- current incident status note under `reports/`

## Continuity hygiene rules
- Keep `MEMORY.md` compact; push detail into `memory/*.md`.
- Archive bloated Telegram direct sessions rather than trying to continue on an overloaded transcript.
- Do not trust a successful direct send (`pong`) as proof that the orchestration session is healthy.
- Refresh proof + handover serially; avoid overlapping `current --refresh` and `reset_ready_refresh` runs to prevent publish-lock collisions.

## Suggested next bounded automation
1. ✅ Landed in EX-05.1: weekly restore-drill evidence refresh now runs via `run_no_nudge_continuity_watchdog.sh` + `continuity.sh restore-drill-refresh`, updating `state/continuity/latest/restore_drill_latest.json` and `reports/restore_drill_auto_*.md`.
2. Add a continuity canary that warns when Telegram direct-lane session context exceeds threshold before overflow.
3. Add a successor packet builder that snapshots only the canonical continuity/handover/proof surfaces plus current blocker registry.
4. Enforce “heavy work off Telegram direct lane” via routing policy / worker handoff defaults.
