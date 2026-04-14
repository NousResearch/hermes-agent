# Execution Supervisor Artifact-Check Contract v1

Status: support-to-implementation contract  
Date: 2026-04-01

## Purpose

Define deterministic post-run checks so execution supervision does not treat provider errors or no-op runs as successful landings.

## Inputs

For each launched task attempt:
- `task_id`
- `expected_artifacts[]` (relative repo paths)
- optional `required_title_tokens[]`
- child session terminal fields:
  - `stop_reason`
  - `provider_error` (raw text if present)
  - final assistant content length

Runtime note (V3-B): when direct child terminal traces are not yet materialized as a dedicated artifact,
the supervisor may consume terminal-signal hints carried on slice/task metadata (`terminal_signal`,
`stop_reason`, `provider_error`, `empty_content`, `final_content_length`) as a bounded interim source.

Runtime hardening note (EXB-02 bounded): `DONE` tasks with no explicit expected artifact list are still
attempt-classified when terminal hints indicate failure (provider error / error stop reason / empty content),
so no-artifact empty-output runs cannot silently bypass failure accounting.

Malformed-output closure note (EXB-02 broader invalid-output relaunch): when a `DONE` row carries child
completion provenance (`completed_via`/worker/session key) but has neither expected artifacts nor any
terminal-signal evidence, it is treated as `FAILED_OTHER` (`insufficient_completion_evidence`) instead of
being omitted from landed/failure accounting.

Provider quota hint preservation note (CPL-02 control-plane tightening): when synthesizing terminal failure
for missing terminal signal, raw data is checked for provider quota hints (e.g., `usage limit`, `quota`, `limit`,
`exceeded`, `exhausted`). If hints are present, error is set to `provider_quota_suspected` to preserve quota
signal for classification.

## Deterministic checks

1. **Presence check**
   - Every `expected_artifacts[]` path must exist.

2. **Non-empty check**
   - Every expected artifact must have file size `> 0`.

3. **Title sanity (optional)**
   - If `required_title_tokens[]` is configured, at least one token must appear in the first 20 lines.

4. **Terminal signal check**
   - Capture session terminal metadata (`stop_reason`, provider error text, empty-content marker).

## Classification logic

Apply in this order:

1. If terminal signal indicates provider quota/rate-limit exhaustion (for example `usage limit`, `quota exceeded`, `rate limit`, HTTP `429`, `limit exceeded`, `exhausted`) → `FAILED_PROVIDER_QUOTA`. Detection tokens include: `usage limit`, `usage_limit`, `quota`, `rate limit`, `rate_limit`, `429`, `limit`, `exceeded`, `exhausted`.
2. Else if any expected artifact missing/empty/title-failed → `FAILED_NO_ARTIFACT`.
3. Else if terminal content is empty (even without explicit artifact refs) → `FAILED_NO_ARTIFACT` (`empty_terminal_content_invalid_completion`).
4. Else if stop reason is error/failed/aborted with no artifact violation → `FAILED_OTHER`.
5. Else → `LANDED`.

## Required output shape

Each attempt must record:

```json
{
  "artifact_check": {
    "all_expected_present": true,
    "missing_paths": [],
    "empty_paths": [],
    "title_check_passed": true
  },
  "terminal_signal": {
    "stop_reason": "completed",
    "provider_error": null,
    "empty_content": false
  },
  "classification": "LANDED"
}
```

## Control-plane consequences

- `FAILED_PROVIDER_QUOTA` → enqueue reroute to non-exhausted worker class; apply worker/provider cooldown.
- `FAILED_NO_ARTIFACT` → relaunch same task with stricter artifact contract and bounded retry budget.
- `FAILED_OTHER` → bounded retry once, then block with operator-visible reason.
  - For blocker family `terminal_error_requires_operator_review`, operator disposition is now canonicalized via
    `docs/ops/execution_supervisor_terminal_error_disposition_contract_v1.md`.
- `LANDED` → mark slice landed and permit frontier progression.

Reroute readiness note (V3-B): task rows should carry explicit reroute posture (`REROUTE_QUEUED` or
`BLOCKED`) with reason and target worker where deterministically available.

Durable cooldown + floor note (V3-C): reroute posture now also carries worker cooldown memory and
queue-level floor-repair planning contracts:
- `routing_policy.cooldown_state.active_workers[]` persists quota cooldown windows across ticks,
  so reroute selection can skip exhausted workers until cooldown expiry.
- `queue_summary.worker_floor_enforcement` projects minimum-worker-floor deficits and deterministic
  launch plans (or explicit blocked reasons) for core/support lanes when ready work exists.
- `queue_summary.reroute_*` counters (`required`, `dispatch_ready`, `blocked`, plus
  `reroute_ready_task_ids[]`) expose launch-readiness pressure without transcript mining.

Dispatch-qualification consumption note (V3-D bounded):
- `queue_summary.dispatch_qualification_consumption` now records strict consumption of the
  latest dispatch-qualification artifact into supervisor state.
- `dispatch_intent` remains plan-only/fail-closed, but `dispatch_ready` is now gated by
  `qualification_consumption.qualification_gate_passed` so candidate-set churn or missing
  qualification evidence blocks intent until re-qualification is observed.

Canary-gated worker restoration note (V3-D bounded):
- Dispatch qualification now treats worker-health canary evidence as a required fail-closed
  precondition for `qualified_ready` candidates.
- If `state/continuity/latest/execution_supervisor_worker_health_canary_latest.json` is
  missing, candidate qualification is blocked with
  `dispatch_worker_health_canary_missing`.
- If a target worker has no row in the canary payload, qualification is blocked with
  `dispatch_target_worker_health_missing`.
- If a target worker row exists but `health_status` is unknown/unrecognized, qualification is
  blocked with `dispatch_target_worker_health_status_unknown`.
- When health evidence is present and valid, worker restoration rules apply:
  - `quarantined` target workers are blocked,
  - `probationary` target workers require `canary_status=pass` before they can be
    `qualified_ready`.

Support route-coverage refinement note (post route-family alignment):
- Supervisor task rows and dispatch candidates now carry explicit `task_class` alongside
  `task_type` so support work is not forced through a single collapsed planning class.
- Queue slices may provide `dispatch_task_class`/`task_class`; when present and valid, that
  class is consumed by reroute/floor planning and dispatch qualification route selection.
- If no explicit class is provided, runtime behavior remains backward-compatible via
  deterministic `task_type`-based inference.

Executable canary-probe + freshness note (V3-F bounded):
- For `probationary` workers with `canary_status=pass`, restoration is additionally gated by
  executable probe evidence carried in the worker-health canary payload:
  - `probe_status=pass`,
  - probe timestamp within configured freshness window,
  - non-empty probe artifact path present on disk.
- If any probe gate fails, dispatch qualification remains `blocked` with explicit probe reason
  (`dispatch_target_worker_canary_probe_*`).
- Dispatch-intent qualification consumption now also applies freshness gating to the latest
  dispatch-qualification artifact (`dispatch_qualification_stale`/timestamp reasons) so stale
  evidence cannot reopen `dispatch_ready`.

Durable canary scheduling + launch-readiness surfacing note (V3-G bounded):
- Dispatch qualification now emits a durable canary probe schedule projection
  (`state/continuity/latest/execution_supervisor_canary_probe_schedule_latest.json`) and append-only
  schedule history, so probationary/quarantined restoration posture is inspectable without
  transcript mining.
- Schedule rows carry per-worker demotion/restore state (`demoted_quarantined`, `canary_required`,
  `probe_required`, `restored_by_probe`, `healthy`), pending/restored timestamps, and next probe due
  timing when applicable.
- Dispatch qualification and dispatch intent both surface `launch_readiness` with:
  - blocked reason counts,
  - automatic demotion/restore posture counts/lists,
  - canary schedule summary pointers.
- This is projection-only hardening: mutation posture remains plan-only fail-closed
  (`launch_mutation_allowed=false`).

Readiness severity gate note (V3-H bounded):
- `launch_readiness` now also carries `severity_gate` to detect persistent non-ready posture
  for the same demoted/restore-pending worker cohort.
- Gate activation criteria are bounded and projection-only:
  - `launch_readiness.state in {blocked, degraded}`,
  - demoted cohort signature remains stable,
  - consecutive non-ready ticks reach configured threshold
    (`OPENCLAW_EXECUTION_SUPERVISOR_LAUNCH_READINESS_PERSISTENCE_TICKS`, default `3`).
- Severity levels:
  - `critical` for persistent `blocked` posture,
  - `warning` for persistent `degraded` posture,
  - `clear` otherwise.
- Mission-control consumes this gate to emit operator-facing warning/action cards while
  preserving fail-closed, plan-only dispatch posture (`launch_mutation_allowed=false`).

Verify-gate preflight coupling note (V3-I bounded):
- `verify_gate_status.sh` now consumes dispatch-qualification launch-readiness severity
  projection (`state/continuity/latest/execution_supervisor_dispatch_qualification_latest.json`).
- When severity gate is active with fresh evidence, preflight predicts a blocker:
  `execution_supervisor_launch_readiness_severity_gate:<reason>`.
- `continuity_now.sh` propagates this under `verify.gate_preflight.launch_readiness_severity_gate`
  and adds `execution_supervisor_launch_readiness_severity_gate_active` to
  `not_ready_reasons`/`warning_reasons`.
- This remains bounded and fail-closed: no launch mutation path is introduced.

Preflight blocker severity normalization note (V3-K bounded):
- Continuity/operator wrapper surfaces now classify the following verify-preflight
  predicted blocker families as **severe** (blocker-grade) rather than warn-only:
  - `strict_autonomy_required_override_denied`
  - `layered_health_gate:<reason>`
  - `execution_supervisor_launch_readiness_severity_gate:<reason>`
  - `execution_supervisor_probe_execution_gate:<reason>`
  - `execution_supervisor_worker_health_canary_gate:<reason>`
- This normalization applies to handover/history/gate-os/gtc preflight posture
  projections so operator and release-facing severity remains consistent with the
  fail-closed semantics already enforced by `continuity_now` / verify-gate coupling.
- Canonical severity-family evaluation now lives in
  `ops/openclaw/continuity/continuity_policy.py::is_severe_verify_gate_preflight_blocker`
  and wrapper surfaces import that helper to reduce future cross-surface drift.

Explicit canary execution evidence integration note (V3-H bounded):
- `probationary` worker restoration now requires explicit canary execution evidence,
  not only `canary_status=pass`.
- For dispatch qualification to move a probationary worker beyond `canary_required`,
  canary evidence must include:
  - `canary_status=pass`,
  - a non-future, fresh canary timestamp,
  - a non-empty canary artifact path present on disk.
- If canary evidence is missing/stale/invalid, qualification remains blocked with
  deterministic reasons (`dispatch_target_worker_canary_timestamp_missing`,
  `dispatch_target_worker_canary_stale`,
  `dispatch_target_worker_canary_artifact_missing`, etc.).
- Probe gates remain required after canary evidence passes; this tightens
  probationary transition sequencing without enabling any launch mutation path.

Probe due-now/overdue operator surfacing note (V3-I bounded):
- Mission-control now projects `launch_readiness.probe_execution_plan` urgency signals from
  dispatch qualification/intent into headline, truth-strip, warnings, and action cards.
- Operator-visible counters are surfaced explicitly:
  - `probe_execution_due_now_worker_count`
  - `probe_execution_overdue_worker_count`
  - `probe_execution_pending_worker_count`
- When probe urgency is present, mission-control emits bounded action cards for inspection/refresh:
  - `inspect_execution_supervisor_probe_execution_plan`
  - `refresh_execution_supervisor_probe_execution_plan`
- This remains projection-only (no launch mutation path changes); fail-closed plan-only posture is
  unchanged.

Verify-gate probe-overdue coupling note (V3-J bounded):
- `verify_gate_status.sh` now evaluates `launch_readiness.probe_execution_plan` urgency as a
  first-class preflight gate surface, with bounded source fallback:
  1) dispatch qualification (`launch_readiness.probe_execution_plan` / top-level), then
  2) standalone `state/continuity/latest/execution_supervisor_probe_execution_plan_latest.json`
     when dispatch projection is missing or stale.
- Fresh overdue evidence (`overdue_worker_count >= OPENCLAW_VERIFY_GATE_STATUS_PROBE_OVERDUE_BLOCKER_MIN`,
  default `1`) activates a bounded blocker with predicted reason:
  `execution_supervisor_probe_execution_gate:probe_execution_overdue`.
- `continuity_now.sh` now propagates this under
  `verify.gate_preflight.launch_readiness_probe_execution_gate` and adds deterministic reason tokens:
  - blocker: `execution_supervisor_probe_execution_overdue_gate_active`
  - warnings: `execution_supervisor_probe_execution_due_now`,
    `execution_supervisor_probe_execution_overdue`
- This is still projection/actionability hardening only; launch mutation posture remains fail-closed
  and plan-only.

Verify-gate probe actionability surfacing note (V3-K bounded):
- `launch_readiness_probe_execution_gate` now carries explicit probe-plan operator pointers in
  verify preflight output:
  - `probe_execution_plan_path` / `probe_execution_plan_present`
  - `probe_execution_source` (`dispatch_qualification` or `standalone_probe_execution_plan`)
  - `dispatch_qualification_failure_reason` (when fallback is engaged)
  - `inspect_probe_execution_plan_command`
  - `first_actionable_command` / `action_priority`
- `continuity_now.sh` propagates these fields under
  `verify.gate_preflight.launch_readiness_probe_execution_gate`.
- `operator_mission_control.sh` now consumes this preflight surface directly and emits
  targeted action cards when probe urgency/failure is present:
  - `inspect_verify_gate_launch_readiness_probe_execution`
  - `refresh_verify_gate_launch_readiness_probe_execution`
- This remains bounded fail-closed observability/actionability hardening only;
  no launch mutation path is introduced.

Probe action-priority persistence note (V3-M bounded):
- `execution_supervisor_probe_execution_plan` now carries explicit
  `action_priority` (`p1` for overdue urgency, `p2` for due-now urgency, otherwise `null`).
- Dispatch qualification `launch_readiness.probe_execution_plan` and standalone
  probe-plan projections persist this priority so urgency survives partial projection
  degradation.
- `verify_gate_status.sh` now consumes persisted `action_priority` when present and
  keeps `first_actionable_command` anchored to probe-plan inspection even when
  due/overdue counters are unavailable.
- `operator_mission_control.sh` now surfaces
  `execution_supervisor_launch_readiness_probe_execution_action_priority` and uses
  persisted priority to keep probe inspection action cards visible/ordered during
  bounded projection loss.
- This remains projection-only fail-closed hardening; no launch mutation path is enabled.

Demotion/recovery lifecycle persistence note (V3-N bounded):
- `launch_readiness.demotion_restore_posture` now carries persisted lifecycle chronology
  derived from canary/probe schedule state:
  - `oldest_restore_pending_since` / `oldest_restore_pending_worker` / `oldest_restore_pending_age_sec`
  - `oldest_demoted_at` / `oldest_demoted_worker` / `oldest_demoted_age_sec`
  - `latest_restored_at` / `latest_restored_worker` / `latest_restored_age_sec`
  - bounded `action_priority` (`p1` when restore-pending cohort includes overdue probes,
    `p2` for restore-pending or demoted cohorts without overdue probes).
- `verify_gate_status.sh` now carries these demotion posture signals inside
  `launch_readiness_probe_execution_gate` and can fall back to
  `demotion_restore_posture.action_priority` when probe-plan urgency fields are missing/stale.
- When this fallback is used, `first_actionable_command` is anchored to
  dispatch-qualification inspection to preserve fail-closed operator guidance.
- This remains bounded observability/actionability hardening only; autonomous launch
  mutation remains disabled.

Demotion lifecycle operator-surface carry-forward note (V3-P bounded):
- `verify_gate_status.sh` now projects the full bounded demotion lifecycle cohort counters
  and chronology under `launch_readiness_probe_execution_gate`:
  - `demotion_demoted_worker_count` / `demotion_restored_worker_count`
  - `demotion_oldest_demoted_at` / `demotion_oldest_demoted_worker` / `demotion_oldest_demoted_age_sec`
  - `demotion_latest_restored_at` / `demotion_latest_restored_worker` / `demotion_latest_restored_age_sec`
- `continuity_now.sh` now preserves these demotion lifecycle fields in
  `verify.gate_preflight.launch_readiness_probe_execution_gate` for downstream
  mission-control consumption.
- `operator_mission_control.sh` now surfaces demotion lifecycle posture in headline/truth-strip,
  emits `inspect_execution_supervisor_demotion_restore_posture` +
  `refresh_execution_supervisor_demotion_restore_projection` action cards,
  and can fall back to verify-preflight demotion posture when dispatch artifacts are absent.
- No launch mutation behavior is introduced; this is fail-closed observability/actionability
  hardening only.

Probe priority provenance + demotion-safe routing note (V3-O bounded):
- `continuity_now.sh` now preserves
  `verify.gate_preflight.launch_readiness_probe_execution_gate.action_priority_source`
  so downstream surfaces can distinguish probe-plan urgency from demotion fallback urgency.
- `operator_mission_control.sh` now carries
  `execution_supervisor_launch_readiness_probe_execution_action_priority_source`
  alongside probe action priority.
- When probe-plan urgency fields are absent but demotion posture carries bounded
  urgency (`launch_readiness_demotion_action_priority`), mission-control now adopts
  that priority as a fallback with source `demotion_restore_posture`.
- Probe-plan inspection action cards are now suppressed for demotion-derived
  fallback urgency, so remediation routes to demotion posture inspection instead
  of a missing/empty probe-plan artifact.
- This remains bounded projection/actionability hardening only; no launch mutation
  path is introduced.

Verify-gate worker-health canary freshness coupling note (V3-L bounded):
- `verify_gate_status.sh` now evaluates worker-health canary source posture from
  dispatch qualification (`source.worker_health_*`) as a first-class preflight gate surface.
- When worker-health gate + artifact are required and canary evidence is
  missing/unreadable/invalid/stale/future (with fresh dispatch-qualification evidence),
  preflight predicts a blocker:
  `execution_supervisor_worker_health_canary_gate:<reason>`.
- `continuity_now.sh` now propagates this under
  `verify.gate_preflight.launch_readiness_worker_health_canary_gate` and emits deterministic
  reason tokens:
  - blocker: `execution_supervisor_worker_health_canary_gate_active`
  - warnings: `execution_supervisor_worker_health_canary_stale`,
    `execution_supervisor_worker_health_canary_missing`, and timestamp/validity variants.
- `operator_mission_control.sh` now prioritizes
  `refresh_execution_supervisor_worker_health_canary_evidence` when this preflight gate is active,
  preserving fail-closed remediation actionability without enabling launch mutation.

Worker-health canary actionability persistence note (V3-L1 bounded):
- `verify_gate_status.sh` now carries explicit canary preflight actionability fields:
  - `dispatch_qualification_failure_reason`
  - `worker_health_canary_source` (`dispatch_qualification` or `standalone_worker_health_canary`)
  - `first_actionable_command`, `action_priority`
- When dispatch qualification is missing/stale or omits canary payload details,
  preflight can now fall back to standalone
  `state/continuity/latest/execution_supervisor_worker_health_canary_latest.json`
  so canary urgency survives bounded projection degradation.
- `continuity_now.sh` now preserves these fields under
  `verify.gate_preflight.launch_readiness_worker_health_canary_gate` and maps
  dispatch-failure reasons to deterministic warning tokens.
- `operator_mission_control.sh` now honors persisted canary preflight actionability
  (priority + first actionable command), keeping worker-health canary refresh/inspect
  action cards visible even when dispatch qualification is unavailable.

Dispatch resource-preflight + uncertainty coupling note (V3-P bounded):
- `continuity_current.sh` now performs mandatory host resource preflight checks during
  dispatch qualification for ready candidates (memory headroom, disk free headroom,
  and load-per-core pressure), with fail-closed candidate blocking when insufficient
  or telemetry is unavailable.
- Dispatch qualification now carries:
  - per-candidate `resource_preflight` evidence
  - top-level `resource_preflight` summary
  - top-level `uncertainty_signal` (`confidence_score`, quantile band, reasons,
    `requires_operator_review`)
- `verify_gate_status.sh` now projects these into
  `launch_readiness_worker_health_canary_gate` as operator-facing preflight signals:
  - `dispatch_resource_preflight_blocked` (active blocker)
  - `dispatch_resource_preflight_degraded`
  - `dispatch_uncertainty_operator_review_required`
- `continuity_now.sh` now preserves these fields/warnings and routes replay hints to
  dispatch-qualification inspection when resource/uncertainty conditions drive the
  preflight signal.
- This remains bounded preflight + projection hardening only; launch mutation remains
  disabled (`mode=plan_only_fail_closed`, `launch_mutation_allowed=false`).

Failover runtime evidence preflight coupling note (V3-N bounded):
- `verify_gate_status.sh` now evaluates
  `state/continuity/latest/failover_stress_runtime_evidence.json` as a first-class
  preflight gate surface under `failover_stress_runtime_evidence_gate`.
- The gate fail-closes on missing/unreadable/invalid/stale evidence, non-pass
  `overall_verdict` / `publish_chain_verdict`, publish-assertion failures, or
  explicit repeatability mismatch (`repeatability.status == "mismatch"`).
- Active failover runtime preflight blockers now predict with reason family:
  `failover_stress_runtime_evidence_gate:<reason>`.
- `continuity_now.sh` now propagates this gate under
  `verify.gate_preflight.failover_stress_runtime_evidence_gate`, with deterministic
  blocker/warning tokens:
  - blocker: `failover_stress_runtime_evidence_gate_active`
  - warnings: `failover_stress_runtime_evidence_stale`,
    `failover_stress_runtime_repeatability_mismatch`, plus
    timestamp/validity/verdict variants.
- Preflight replay/actionability now includes inspect/refresh commands for
  failover runtime evidence (`continuity.sh failover-stress-runtime-evidence --json`).

Worker-health canary evidence producer note (post V3-H bounded):
- `ops/openclaw/continuity/execution_supervisor_worker_health_canary.py` is now the
  canonical producer for:
  - `state/continuity/latest/execution_supervisor_worker_health_canary_latest.json`
  - `state/continuity/history/execution_supervisor_worker_health_canary_history.jsonl`
- The producer consumes codex dispatch-health posture and route-worker discovery
  from latest dispatch-qualification when available.
- If dispatch-health is unavailable, producer fallback is fail-closed:
  discovered/default workers are emitted as `probationary` with
  `canary_status=required` and `probe_status=required`.
- Operator mission-control now emits a refresh action when
  `worker_health_gate_required=true` but worker-health canary source is missing.

Executable probe scheduling evidence note (V3-I bounded):
- Dispatch qualification now emits a dedicated probe-execution plan artifact:
  - `state/continuity/latest/execution_supervisor_probe_execution_plan_latest.json`
  - `state/continuity/history/execution_supervisor_probe_execution_plan_history.jsonl`
- This artifact is plan-only fail-closed (`mode=plan_only_fail_closed`,
  `launch_mutation_allowed=false`) and converts canary/probe-required workers into
  deterministic evidence tasks with explicit required fields:
  - `capture_canary_execution_evidence` (`canary_status`, `canary_checked_at`, `canary_artifact_path`)
  - `capture_probe_execution_evidence` (`probe_status`, `probe_checked_at`, `probe_artifact_path`)
- Each planned worker row carries an expected artifact-path hint,
  current evidence posture, due/schedule timing, and linked target/blocked tasks,
  so operator follow-through can be audited from canonical artifacts rather than
  inferred from transcript fragments.

Stable probe urgency cohort signature note (V3-O bounded):
- `execution_supervisor_probe_execution_plan` now persists stable urgency cohort
  fingerprints for due-now and overdue probe cohorts:
  - `due_now_cohort_workers[]`, `due_now_cohort_signature`,
    `due_now_cohort_signature_first_seen_at`,
    `due_now_cohort_signature_consecutive_ticks`
  - `overdue_cohort_workers[]`, `overdue_cohort_signature`,
    `overdue_cohort_signature_first_seen_at`,
    `overdue_cohort_signature_consecutive_ticks`
- Cohort signatures are deterministic (`sha256(sorted(workers))[:16]`) and are
  persisted tick-over-tick from prior probe-plan artifacts to distinguish
  stable urgency from churn when raw counts are unchanged.
- Dispatch qualification `launch_readiness.probe_execution_plan` and
  verify-gate probe-execution preflight now carry these fields for fail-closed,
  projection-only operator triage continuity.
- This is bounded observability hardening only; launch mutation remains disabled
  (`mode=plan_only_fail_closed`, `launch_mutation_allowed=false`).
