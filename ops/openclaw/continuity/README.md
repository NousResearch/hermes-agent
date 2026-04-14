# Continuity OS (local-first) — quick operator guide

This directory is the deterministic continuity substrate for the control-plane workspace.

## What it does
- Captures host/runtime truth snapshots.
- Writes append-only checkpoints (`json + md`).
- Maintains latest pointers for fast, safe resume.
- Provides verify-before-mutate gate.
- Projects successor-safe `continuity/current.json` + `handover/latest.{json,md}` surfaces.
- Exposes operator mission-control headline/actions export.

## Canonical command sequence (safe resume)
1. `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/snapshot_ground_truth.sh`
2. `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/verify_then_resume.sh`
   - includes baseline gates by default: continuity DB integrity (`--strict`), architecture contract validation, swarm operability checks, slot-fill protocol contract checks (`check_slot_fill_protocol.sh --json`), and strict GTC latest-surface schema validation (`gtc_latest_schema_check.sh --strict --json`).

Preflight strict-autonomy mode/source before verify:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh verify-gate-status --json`

Restore-drill freshness refresher (bounded dry-run rollback evidence writer):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh restore-drill-refresh --json`

Optional execute of first planned action:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/verify_then_resume.sh --execute`

Mutating dispatcher commands now require explicit coherence-aware action tokens:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh --action-token <current.action_token> reconcile`
- Fetch token from `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json` (`.action_token`).
- Anchor-only tokens are rejected by default when coherence metadata exists; break-glass override: `--allow-legacy-anchor` (or `OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY=1`).
- Explicit unanchored override still exists for emergency/manual workflows only: `--allow-unanchored-mutate`.

Compact operator status surface:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/continuity_now.sh`
- JSON form: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/continuity_now.sh --json`
- Includes `incident_replay.recommended_commands` with one-command `gtc_incident_replay` / `queue_arbitrator trace` / `history` pointers for routine incident slicing.
- Surfaces sustained stale degraded pending backlog signal (`warning_reasons += degraded_pending_backlog_stale_sustained`) with explicit `queue-sync` + trace replay commands.
- Includes `parity` weekly freshness status (`parity:weekly_harness`) and flags due cadence as a warning, not a hard blocker.

Successor-safe surfaces (read-only first):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh blocker-registry --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh handover --refresh --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh mission-control --refresh --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh gate-os --refresh --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh execution-frontier --refresh --json`

Doctrine drift registry writer (deterministic upsert contract; direct mutator path requires action token):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/doctrine_drift_registry_write.sh --action-token <current.action_token> --reason <reason_key> --severity warn --evidence <ref> --json`

Continuity audit rollup/history surface:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh`
- JSON form: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh history`

Queue arbitration + replay trace surface:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh ready-list --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh claim --agent <name> --actor-role <planner|executor|validator|sre_watchdog|librarian|outer_gate> --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh trace --task-id autopilot:quality_gate --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh handoffs --json`
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_arbitrator.sh remediate --expire-overdue-locks --release-terminal-locks --requeue-resolved-blocked --requeue-orphaned-running --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh queue-arb ...`
- transition command now enforces both transition policy matrix + `role_required` boundary by default; use `--allow-any-transition` only for explicit manual recovery.
- runtime notes:
  - `ops/autopilot/bin/hl_autopilot_tick.sh` claims queue tasks through `queue_arbitrator.sh claim` and emits deterministic transitions on completion/retry/block paths.
    - delegated `kind: agent` completions are ingress-gated and persisted (`state/contracts/reports/...` under repo root), with queue reasons:
      `autopilot_delegated_contract_retry_backoff`, `autopilot_delegated_contract_retry_exhausted`, `autopilot_delegated_contract_invalid`,
      plus provider-aware branches `autopilot_delegated_provider_retry_backoff`, `autopilot_delegated_provider_retry_exhausted`, `autopilot_delegated_provider_blocked_nonretryable`.
    - non-zero runtime exits now classify provider failures (`autopilot.provider_failure_summary.v1`) and route bounded retries/exhaustion via
      `autopilot_provider_failure_retry_backoff`, `autopilot_provider_failure_retry_exhausted`, `autopilot_provider_failure_blocked_nonretryable`.
      Classifier now extracts nested Codex stream error payloads (including `error.code/type` + usage tokens), promotes `server_error/unknown_error`
      transient mapping, and allows a one-shot immediate requeue for zero-output transient stream failures before normal backoff.
      Provider-failure summaries are now strict-schema validated before queue transition/handoff persistence.
      Queue-sync projection defaults to drop-on-invalid for legacy safety, with optional fail-close mode for strict governance (`--invalid-provider-summary-mode fail_close`).
    - queue transition handoff packets now carry structured delegated gate summaries under `gate_metadata.gate_summary` (classification, retry profile/signature, decision path, queue reason), enabling remediation without log scraping.
    - queue claim deferrals now honor structured retry hints from arbitrator skip metadata (for example cooldown-driven `retry_after_sec`) instead of always using one static retry delay.
    - queue claim/transition role mismatches now auto-retry once using arbitrator-provided expected role (`role_required_mismatch`) and persist the recovered actor-role override in autopilot state (`queue_role_overrides`) to reduce operator nudges from role-handoff drift.
  - `ops/openclaw/architecture/run_competitive_parity_harness.sh` (dispatcher: `continuity.sh parity-run`) is a non-autopilot producer path that now uses the same claim/transition lock discipline (`task_id=parity:weekly_harness`).
  - `ops/openclaw/continuity/normalize_event_sources.sh` now also runs through queue claim/transition discipline (`task_id=continuity:normalize_event_sources`, role `sre_watchdog`).

Queue replay/projection verification (deterministic journal audit):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_replay_verify.sh --json`
- strict gate mode: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/queue_replay_verify.sh --strict --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh queue-replay --strict --json`
- writes:
  - `state/continuity/latest/queue_replay_projection.json`
  - `state/continuity/latest/queue_replay_verify.json`
- status policy:
  - active replay integrity mismatches (`active_status_mismatches`) and role mismatches are fail-close (`status=fail`, strict mode exits non-zero).
  - legacy/historical replay artifacts (`legacy_status_mismatches`, for example terminal queue rows with active projected replay state) remain visible but non-blocking (`status=warn`).
  - hard discontinuities still surface as `warn`.
  - idempotent stale-from-status rows are surfaced under `soft_discontinuities` (`soft_discontinuity_task_count`) without tripping warn/fail.
  - normalized historical replay debt is surfaced under `historical_discontinuities` (`historical_discontinuity_task_count`), including terminal reopen rows missing a reset transition and pre-S5-A stale review requeue residue.

Audited lock-break workflow (operator break-glass):
- preview: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/lock_break.sh --task-id <task_id> --reason "<why>" --operator <name> --action-token <current.action_token> --json`
- apply: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/lock_break.sh --task-id <task_id> --reason "<why>" --operator <name> --action-token <current.action_token> --apply --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh lock-break ...`
- writes:
  - `state/continuity/latest/lock_break_last.json`
  - `state/continuity/lock_break/<audit_id>.json`

Continuity DB integrity check:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/db_integrity_check.sh --json`
- strict gate mode: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/db_integrity_check.sh --strict`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh db-check --json`
- now includes queue-contract provenance checks (source/task namespace alignment), role boundary checks (`role_required` presence/enum/review invariants), handoff packet integrity/linkage checks, strict provider-failure handoff summary schema validation (`autopilot.provider_failure_summary.v1`), terminal evidence/artifact trace hygiene, and weekly parity freshness signal (warn-only).

Swarm runtime doctor (operator preflight):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/swarm_runtime_check.sh --strict --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh swarm-check --strict --json`
- now includes Ground-Truth Connectors v2 readiness (`state/gtc-v2/latest/gateboard.json`) as a hard mutation gate.

Event source namespace normalization:
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/normalize_event_sources.sh --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh normalize-events --json`
- now queue-disciplined as a `continuity_ops` producer task with role-aware claim/transition.

Deterministic web capture wrapper (phase-2 start, low-noise):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_web_capture_macro.sh --mode auto`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh web-capture --dry-run --json`
- wrapper is cadence-gated + per-domain guard state (`state/continuity/latest/web_capture_domain_<domain>.json`), queue-integrated (`task_id=continuity:web_capture:<macro_slug>`), and emits blocker events only (bot-wall requires sustained-window threshold before escalation).
- login-wall now opens deterministic operator contracts (`state/continuity/latest/web_capture_login_contract_<domain>.{json,md}`) for Browser Relay/manual-auth continuation.
- runtime quality gates now fail-fast on practical artifact integrity signals (`blank_screenshot_detected`, `dom_capture_missing`) before marking run `DONE`.

Deterministic web capture scheduler (governed runtime surface):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_web_capture_scheduler.sh --dry-run --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh web-capture-scheduler --dry-run --json`
- emits governed scheduler state: `state/continuity/latest/web_capture_scheduler_state.json`
  (`schema_version=openclaw.web_capture.scheduler_state.v1`, contract schema: `ops/openclaw/architecture/schemas/web_capture_scheduler_state.schema.json`).
- scheduler freshness/contract validity are surfaced in `continuity_now`, `operator_mission_control`, `gate_os_snapshot`, and `db_integrity_check`.

Librarian/Curator runtime scaffold:
- ingest curation queue: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh librarian ingest --json`
- lint supersession + queue hygiene: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh librarian lint --json`
- promote canonical artifact: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh librarian promote --candidate-id <id> --reason "<why>" --operator <name> --json`
- rebuild canonical index: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh librarian build-index --json`
- retrieval hygiene check: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh librarian hygiene --json`
  - faster pass: add `--skip-retrieval-eval` when only schema/secret hygiene is needed.
- writes under: `state/continuity/librarian/{curation_queue.json,promotions.jsonl,canonical_index.json,canonical_index.md,retrieval_hygiene.json}`

Research Case governance MVP (governed extraction → understanding → implementation):
- init case scaffold: `python3 /home/yeqiuqiu/clawd-architect/scripts/research_case_pipeline.py init --case-id <id> --title "..." --intent "..." --source <path>`
- record synthesis + head update: `python3 /home/yeqiuqiu/clawd-architect/scripts/research_case_pipeline.py record-synth --case-id <id> --synth-id <synth_id> --takeaway "..."`
- promote candidate + gatecheck + promotion-contract gate:
  - `python3 /home/yeqiuqiu/clawd-architect/scripts/research_case_pipeline.py promote --case-id <id> --candidate-id <cand_id> --requirement "..."`
  - emits `promotion_candidate.json`, `promotion_gate_decision.json`, append-only `promotion_gate_decisions.jsonl`, and fail-closed gate outcome in case `gatecheck.json`
- lint/status/publish latest: `python3 /home/yeqiuqiu/clawd-architect/scripts/research_case_pipeline.py lint --case-id <id> --strict --json` and `python3 /home/yeqiuqiu/clawd-architect/scripts/research_case_pipeline.py status --case-id <id> --publish --json`
- writes under: `memory/research_cases/<id>/` with checkpoint/handover mirrors + `state/continuity/latest/research_case_*.json` registry surfaces.

Lane-governance runtime entrypoints (contractized operators):
- promotion gate runner: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh promotion-gate --candidate <promotion_candidate.json> --json`
- model qualification/rollout gate runner: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-gate --packet <model_rollout_candidate.json> --json`
- model rollout health snapshot producer: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-health --json`
- model rollout cost-governance snapshot producer: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-cost --json`
- long-window route-policy soak/lint snapshot: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-route-policy-lint --window-hours 168 --json`
- ring-soak automation snapshot: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-soak --json`
- consolidated rollout dashboard snapshot: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-dashboard --json`
  - now surfaces live effective routing (`routing.latest/effective`) from session-topology decisions so operators can see current route class/model and blocker guidance directly in the dashboard headline.
- model rollout ledger/controller: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh model-rollout-controller --json`
- deterministic route-policy topology router (session/task/risk -> route class/model): `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-route --topology <session_topology_contract.json> --request <session_route_request.json> --qualification-decision <model_rollout_gate_decision.json> --transport-decision <session_transport_decision.json> --json`
  - strict transport conformance is default; bounded legacy bypass is explicit via `--legacy-allow-missing-transport-decision`
- deterministic transport-topology router (chat/thread -> lane/agent/session key): `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh session-transport-route --topology <session_topology_transport_contract.json> --request <session_topology_transport_route_request.json> --json`
- knowledge review/approval/promotion queue runtime: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh promotion-queue list --json`
- markdown conversion quality gate: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh markdown-gate --packet <markdown_conversion_gate_packet.json> --json`
- source material classification layer: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh material-classify --packet <source_material_classification_packet.json> --json`
- production knowledge ingestion layer: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh knowledge-ingest evaluate --packet <production_knowledge_ingestion_packet.json> --json`
- doc/PDF intake closeout integration gate: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh doc-intake-closeout --packet <document_intake_batch_integration.json> --json`
- release evidence ladder gate: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh release-evidence-gate --bundle <release_evidence_bundle.json> --json`
- repo-review fold-in/closeout verifier gate: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh repo-review-closeout --target-row <ROW_ID> --report-path <foldin_report.md> --json`
- legacy bounded queue helper (kept for compatibility): `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh knowledge-queue status --json`
- lane crossover ingress guard: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh lane-crossover-guard --packet <lane_crossover_packet.json> --to-lane-id <lane> --to-lane-epoch <epoch> --allow-from <lane>=<epoch>`
- cross-lane bridge ingest validator: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh bridge-ingest --bridge <bridge_object.json> --to-lane-id <lane> --to-lane-epoch <epoch> --allow-from <lane>=<epoch>`

Ground-Truth Connectors v2 sync (append-only evidence + latest gateboard):
- `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_v2_sync.sh --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh gtc-sync --json`
- emits deterministic evidence streams under `state/gtc-v2/evidence/**` and writes latest readiness pointers under `state/gtc-v2/latest/`.
- now also projects event lifecycle state to `state/gtc-v2/latest/event_projection.json` (`schema_version=gtc.event.v2`) and ships replay hints at `state/gtc-v2/latest/incident_replay.json`.
- strict latest-surface schema gate: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_latest_schema_check.sh --strict --json`
- dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh gtc-schema-check --strict --json`
- deterministic incident replay bundle (evidence chain + artifact pack from GTC only):
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_incident_replay.sh --incident-index 1 --json`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_incident_replay.sh --route-key "watchdog.web_capture|wrapper_exit_bybit_derivatives_capture" --json`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/gtc_incident_replay.sh --incident-index 1 --checkpoint-scope full --json` (break-glass broad checkpoint expansion)
  - dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh gtc-replay --incident-index 1 --json`
  - default checkpoint expansion is incident-scoped (`--checkpoint-scope incident`) to avoid replay noise from unrelated rows sharing the same checkpoint id.
  - writes deterministic bundles under `state/gtc-v2/incident_replay/incident-<incident_id>.{json,md}` (use `--no-write` for read-only stdout).

- incident-scoped replay examples:
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --since-checkpoint latest:drift_reconcile`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --since-checkpoint chk_20260308T141157Z_13167a --until latest`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --source-preset control-plane --trigger any --include-suppressed`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --source-preset autopilot --hours 24`
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/history.sh --tasks autopilot:% --actor-role executor --hours 24`

Drift-only reconcile (low-risk refresh path):
- canonical dispatcher command: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh --action-token <current.action_token> reconcile --json`
- fetch token from: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json` (`.action_token`)
- optional cooldown tuning: `--min-interval-sec <n>` (default `OPENCLAW_CONTINUITY_RECONCILE_MIN_INTERVAL_SEC` or `1800`)
  to avoid high-frequency `drift_reconcile` checkpoint churn; during cooldown it refreshes bridge/handover without writing a new checkpoint.
- `continuity_now.sh` now computes pointer/env/ground-truth capture alignment from live latest artifacts (not only cached bridge payload),
  and downgrades `ground_truth_capture_drift` to cooldown warnings while drift-reconcile cooldown is still active:
  - `ground_truth_capture_drift_cooldown`
  - `ground_truth_capture_drift_cooldown_policy_lag` (when latest ground-truth moved ahead of the reconcile checkpoint during cooldown)
  - reconcile contract flag: `reconcile.cooldown_policy_lag_warning_active` (canonical bool for the policy-lag warning posture)
  - backward-compatible alias (deprecated): `reconcile.cooldown_warning_suppressed_policy_lag`

## Scripts
- `init_db.sh` — initializes `state/continuity/continuity_os.sqlite` and schema.
- `capture_env_snapshot.sh` — writes env snapshot derived from latest ground-truth snapshot.
- `write_checkpoint.sh` — writes new append-only checkpoint pair + updates latest pointers.
- `render_context_handover_compat.sh` — renders `reports/handover_context_latest.md` from checkpoint truth.
- `sync_latest_artifacts.sh` — verifies pointer/hash alignment and writes `runtime_truth_bridge.json`.
- `verify_then_resume.sh` — executes baseline invariants + checkpoint verification commands before mutation (baseline can be bypassed with `--skip-baseline-checks` only when explicitly needed).
  - strict autonomy regression gate is default-on globally; verify now runs the autonomy regression cluster unless explicitly disabled.
  - explicit policy toggle: set `OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REGRESSIONS=0` (or legacy `OPENCLAW_STRICT_AUTONOMY_REGRESSIONS=0`) to disable strict autonomy in controlled contexts; set to `1` to force-enable explicitly.
  - per-run override: `--no-strict-autonomy-regressions` disables strict mode for one invocation (unless required policy is active), while `--strict-autonomy-regressions` force-enables it.
  - fail-closed policy mode: set `OPENCLAW_VERIFY_GATE_STRICT_AUTONOMY_REQUIRED=1` to force strict autonomy gating on every wrapper verify run and reject explicit `--no-strict-autonomy-regressions` bypass attempts.
  - runtime storm guard (default enabled) now rate-limits repeated wrapper-triggered verify executions with a shared token window and READY-report reuse path:
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_ENABLED` (default `1`)
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_WINDOW_SEC` (default `60`)
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_MAX_RUNS` (default `4`)
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_REUSE_READY_MAX_AGE_SEC` (default `120`)
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_OVER_BUDGET_MISSING_REPORT_GRACE_RUNS` (default `1`; max over-budget grace runs when verify report is missing/unknown before fail-closed blocking)
    - `OPENCLAW_VERIFY_GATE_STORM_GUARD_STATE_PATH` (default `state/continuity/latest/verify_gate_storm_guard_state.json` under `OPENCLAW_ROOT`)
    - budget accounting tracks actual verify executions (run decisions) so repeated `reuse_ready` / `block` invocations do not extend cooldown indefinitely.
    - when over budget, a fresh `READY` verify report is reused (skip re-run); non-READY/stale READY still fail-closed immediately; missing/unknown report gets bounded grace and then fails closed with `verify_gate_storm_guard_*` blocker reason.
  - verify report now records strict-autonomy provenance at `strict_autonomy_regressions.source` and `strict_autonomy_regressions.effective_source` (`wrapper`-effective source when the shared verify-gate wrapper injected `--strict-autonomy-regressions`), plus `strict_autonomy_regressions.wrapper_effective` hints for operator auditability.
  - strict mode now fail-closes if the autonomy cluster summary does not explicitly include `gtc_latest_schema_failclose`, `gtc_publish_manifest_auth_dual_mode`, `gtc_incident_replay_verify_gate_posture`, `gtc_publish_transaction_regressions`, `queue_cooldown_authority_regressions`, `no_nudge_reminder_runtime_hardening`, `swarm_operability_contract_regressions`, and `slot_fill_protocol_contract_regressions` as selected+reported checks, preserving expected harness command lineage (`check_gtc_latest_schema_failclose_regressions.py`, `check_gtc_publish_manifest_auth_regressions.py`, `check_gtc_incident_replay_regressions.py`, `check_gtc_publish_transaction_regressions.py`, `check_queue_cooldown_authority_regressions.py`, `check_no_nudge_reminder_runtime_regressions.py`, `check_swarm_operability_regressions.py`, `check_slot_fill_protocol_regressions.py`) and required scenario contracts under `strict_autonomy_regressions.required_checks`.
  - strict mode also requires deterministic summary provenance for those required checks: `summary_schema_version` + `required_check_provenance` (`schema_version` + `check_id` + `contract_fingerprint` + `contract_inputs`), where the fingerprint is a canonical sha256 over immutable contract inputs (check id/harness/source/scenario set/min-count), and `contract_inputs` must exactly match the expected machine-readable contract.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic verify-report timestamps during replay/preflight harness runs.
- `verify_gate_status.sh` — preflight status helper for wrapper-equivalent strict-autonomy effective mode/source, override/required context, and predicted verify-gate blocker reason.
  - now also emits `routing_preflight` (latest routing decision freshness + effective route class/model + blocker/actionable command hints).
  - dispatcher shortcut: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh verify-gate-status --json`
- `restore_drill_refresh.sh` / `restore_drill_refresh.py` — deterministic restore-drill freshness refresher (weekly cadence via watchdog), selecting a bounded READY checkpoint, running `verify_then_resume.sh --run-rollback`, writing `restore_drill_latest.json`, and emitting `reports/restore_drill_auto_*.md` evidence.
- `continuity_now.sh` — renders compact continuity status from checkpoint + verify + ground-truth + bridge (includes compact reconcile history rollup + queue transition traceability summary).
  - refresh storm guard (default enabled) now rate-limits repeated `--refresh` invocations to prevent filesystem churn / sync-validation storms on the live refresh path (`snapshot_ground_truth` + refresh hooks):
    - `OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_ENABLED` (default `1`)
    - `OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_WINDOW_SEC` (default `60`)
    - `OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_MAX_RUNS` (default `4`)
    - `OPENCLAW_CONTINUITY_REFRESH_STORM_GUARD_STATE_PATH` (default `state/continuity/latest/refresh_storm_guard_state.json` under `OPENCLAW_ROOT`)
    - over budget (prospective run would exceed `max_runs` within `window_sec`) skips refresh execution for that call and surfaces warning `refresh_hook_storm_guard_active` with structured state under `refresh_storm_guard`.
  - stale-wave auto-remediation state contract is now schema-bound (`ops/openclaw/architecture/schemas/queue_stale_wave_auto_remediation.schema.json`) and published under `queue.stale_wave_auto_remediation_contract.{contract_schema_path,projection_schema_valid,state_schema_valid,schema_validation_errors}` for fail-close drift detection.
  - now surfaces compact verify-gate preflight status in-place under `verify.gate_preflight` (strict-autonomy enabled/source, required + override state, predicted blocker) so operators can assess strict-autonomy gate posture without running a separate command.
  - now computes/writes `state/continuity/latest/coherence_stamp.json` (anchor+policy+evaluator+connector+scheduler tuple hash), publishes `coherence_bundle_latest.json` with shared generation metadata (`build_generation_id`, `valid_until`), and fail-closes readiness on policy signature drift or expired critical connector leases.
  - when verify status is `READY` and top-level not-ready posture is drift-only (`ground_truth_capture_drift` / `connector_freshness_drift`), continuity_now now reclassifies those reasons to `reconcile_only_reasons` + warning-tier surfacing (`ground_truth_capture_drift_reconcile_only`, `connector_freshness_drift_reconcile_only`) so strict output no longer reports contradictory verify-ready vs continuity-not-ready posture while preserving stale evidence.
  - suppresses derivative `gtc_gateboard_blocked` when the live gateboard is blocked only by `verify_status_not_ready:*`, `verify_last.json` has already advanced to `READY`, and the verify report timestamp is newer than `gateboard.generated_at`; the stale publication lag is surfaced as warning `gtc_gateboard_verify_status_lag` plus `coherence.gtc_gateboard_derivative_suppression_reason=gtc_verify_status_stale_after_verify_ready` instead of a false blocker, and `gtc` is projected to effective truth (`verify_status=READY`, empty `blocking_reasons`) while preserving raw gateboard residue in `gtc.{verify_status_raw,blocking_reasons_raw,mutate_allowed_raw,status_raw}`.
  - suppresses derivative `connector_freshness_drift` as a top-level not-ready reason when coherence reports expiry-only connector drift while GTC gateboard is already hard-blocked by a non-connector blocker; connector evidence is still surfaced under `coherence.connector_*` plus warning `connector_freshness_drift_suppressed_by_gtc_gateboard`.
  - suppresses derivative `layered_health_gate_unready` from top-level not-ready reasons when verify preflight shows layered-health failure isolated to runtime-truth lanes (`A2_RUNTIME_CONTINUITY`, `C1_OPERATOR_SURFACE`) with no active rollout/failover blockers, **or** when verify is already blocked on `a6_observability_failed` and the only failed A6 component is `layered_health_snapshot`; suppression is exposed under `coherence.layered_health_derivative_*` and warnings `layered_health_gate_unready_suppressed_derivative` / `layered_health_gate_unready_suppressed_by_verify_blocker`.
  - suppresses stale-latch `verify_blocker` when `verify_last` remains `BLOCKER/a6_observability_failed` but verify-gate preflight predicts `ready_to_run=true` with no active blocker reason (and no active rollout/failover blockers), surfacing warning `verify_blocker_latch_residue_suppressed` plus coherence annotations under `coherence.verify_blocker_latch_residue_*` instead of a false top-level blocker.
  - `layered_health_snapshot.sh` now consumes that `coherence.layered_health_derivative_suppressed` marker (instead of blanket suppression), honors continuity-projected `rollout_blocker_reasons`, and adds fail-closed restore coupling (`a6_restore_evidence_not_pass`) whenever `SLO-4_RESTORE_DRILL_FRESHNESS` is not pass.
  - publishes rollout-coupled coupling normalization in-place via `rollout_blocker_reasons` + `rollout_blocked` (fail-closed blocker posture) and `rollout_warning_reasons` (warning-only rollout residue) so downstream health surfaces can distinguish READY/allowed posture from rollout warning debt without silently promoting blocker semantics.
  - policy freshness epoch is now monotonic against published coherence surfaces (`current/coherence_stamp/coherence_bundle_latest`) so stale/missing `policy_freshness_state` cannot silently roll epoch backward.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic generated/age windows during replay and regression harness runs.
- `continuity_current.sh` — computes successor-safe `state/continuity/current.json` + derived pointer surfaces (`state/continuity/pointers/latest.json`, canonical read pointer `state/continuity/latest/continuity_read_pointer.json`).
  - now embeds shared coherence metadata (`coherence.tuple_hash`, policy signature, connector freshness reasons, bundle generation/ttl), emits canonical `.action_token`, and auto-refreshes when coherence/policy/connector freshness drifts without anchor movement.
  - generation-pinned read contract is published in the canonical read pointer (`continuity_read_contract`: current sha/generated_at + coherence generation/ttl) so downstream consumers can fail-close on mixed-generation reads.
  - stale-wave auto-remediation failures are now projected as live-only warnings: when failure evidence exists but there is no active stale-wave signal, `queue_stale_wave_auto_remediation_failed` is suppressed as historical residue and annotation fields (`failure_present`, `failure_signal_live`, `failure_signal_state`, `failure_residue_*`, `failure_observed_*`) are emitted for forensic context; dispatch context mirrors this classification under `queue_stale_wave_auto_failure_signal_state` + `queue_stale_wave_auto_failure_observed_*`.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic cache-TTL/coherence-expiry evaluation in replay/regression flows.
- `doctrine_drift_registry_write.sh` — deterministic doctrine drift incident upsert writer (reason-derived default incident ids, evidence de-duplication, schema-bound registry contract at `ops/openclaw/architecture/schemas/doctrine_drift_registry.schema.json`).
- `handover_latest.sh` — emits `state/handover/latest.{json,md}` and reports stale drift vs canonical truth anchor.
  - now mirrors compact verify-gate preflight posture (`verify_gate_preflight`) into handover JSON/MD and elevates severe predicted blockers (for example `strict_autonomy_required_override_denied`) into explicit `blockers` entries with verify-gate follow-up action.
  - enforces generation-pinned read contract parity against `state/continuity/latest/continuity_read_pointer.json` (current sha/generated_at + coherence generation) and fail-closes resume/reset safety on mismatch/stale pointer evidence.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic handover `generated_at` during replay/regression runs.
- `operator_mission_control.sh` — single-pane mission-control truth strip + action export (`state/continuity/latest/operator_mission_control.json`).
  - now carries verify-gate preflight posture in both `truth_strip` and `headline` (`verify_gate_*`) for at-a-glance strict-autonomy required/override/blocker context.
  - now surfaces effective routing diagnostics (`effective_routing` truth row + `headline.routing_preflight_*` + optional `inspect/unblock_effective_routing` actions when routing is blocked/stale).
  - now includes generation-pointer contract parity (`generation_pointer_contract`) and fail-closes headline readiness/mutation gate when mixed-generation continuity reads are detected.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic headline timestamps, cooldown windows, and overdue-lock classification in replay/regression runs.
- `gate_os_snapshot.sh` — unified GateOS result snapshot using `clawd.gate_result.v1` (`state/continuity/latest/gate_os_latest.json`).
  - now includes `continuity.verify.preflight` gate result sourced from `verify.gate_preflight` (`continuity_now`), with severe predicted blockers (for example `strict_autonomy_required_override_denied`) promoted to `hard_fail` semantics.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic gate run timestamps and freshness-age evaluation.
- `history.sh` — on-demand continuity audit rollup/history from continuity DB + checkpoints + reconcile event activity, plus `work_queue`/`task_transitions` replay filters (`--task/--tasks`, `--actor-role/--actor-roles`).
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic default range endpoints, age windows, and stale-lock cutoff evaluation in replay/preflight runs.
- `reconcile.sh` — low-risk drift reconcile path (writes fresh checkpoint when not-ready is drift-only; includes min-interval guard + sync-only fallback during cooldown, plus low-noise reconcile telemetry persisted only on emitted events).
  - reconcile telemetry keys now follow policy taxonomy `continuity.reconcile.event_policy.v2`:
    - policy refusal (`policy.refused_non_drift_only`) => `warn`
    - cooldown-only outcomes (`cooldown.*`) => `info` unless infra failure
    - infra/command failures (`*.failed`) => `critical`
- `queue_sync_from_autopilot_json.sh` — mirrors autopilot JSON into `work_queue` + `task_transitions` + `task_dependencies` + `task_file_targets` + `task_artifacts` with run/artifact evidence refs, `role_required` assignment, queue cooldown projection (`next_after_ts -> cooldown_until`), cross-role handoff packet persistence, and strict provider-failure gate summary validation with controlled handling mode (`drop` default, optional fail-close via `--invalid-provider-summary-mode fail_close` or `OPENCLAW_QUEUE_SYNC_INVALID_PROVIDER_SUMMARY_MODE=fail_close`). When queue infra has recovered, it also backfills explicit degraded-local execution reconciliation annotations (`autopilot_degraded_local_execution_reconciled`) from autopilot degraded-run ledger state for first-class forensics, and projects sustained stale pending degraded backlog signal/recovery state under `queue_infra_degraded.degraded_pending_stale_signal` (with event routing keys `degraded_pending_backlog_stale_sustained` / `degraded_pending_backlog_recovered`).
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` via shared `fixed_now.py` helper for deterministic queue-sync timestamps/reconciliation windows during replay/regression runs.
- `queue_arbitrator.sh` — deterministic claim/transition helper over dependency-ready tasks + file locks + replay trace/handoff views, with runtime `role_required` enforcement, queue cooldown-aware ready/claim gating (`cooldown_until` respected), strict provider-failure gate summary schema validation on `--gate-summary-json`, and guided remediation utilities.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` via shared `fixed_now.py` helper for deterministic ready/claim/transition clock authority during replay/regression runs.
- `repair_provider_failure_summaries.sh` — dry-run/apply repair helper for malformed persisted `autopilot.provider_failure_summary.v1` handoff payloads (`task_handoff_packets.gate_metadata_json`), replacing invalid `gate_summary` blobs with auditable `gate_summary_repair` metadata.
- `queue_replay_verify.sh` — deterministic queue journal replay/projection verifier (status + role projection mismatch detection).
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic projection/report `generated_at` timestamps during replay/regression runs.
- `lock_break.sh` — audited lock-break workflow with action-token envelope (legacy truth-anchor alias retained), operator identity, and replay-safe audit artifacts.
- `librarian_curator.sh` — librarian/curator scaffold for ingest, lint, promote, build-index, and retrieval hygiene checks.
- `truth_anchor_guard.sh` — mutation precondition validator used by mutating command router paths.
  - enforces coherence-aware action tokens by default (tuple hash + policy signature + generation + valid-until); anchor-only tokens require explicit `--allow-legacy-anchor` break-glass override.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` via shared `fixed_now.py` helper for deterministic action-token expiry evaluation in replay/regression harness runs.
- `check_action_token_regressions.py` — local regression harness for action-token durability edges (expiry, mismatch matrix, stale policy/coherence without anchor movement).
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_action_token_regressions.py`
- `check_delegation_contract_regressions.py` — focused delegated completion ingress + gate contract regression harness (env-probe invalid ingress, partial packet retry routing, provider transient/non-retryable classification, path/exit-code verification).
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_delegation_contract_regressions.py`
- `check_autopilot_delegated_ingress_regressions.py` — focused autopilot delegated ingestion wiring harness (live agent-step ingress decision persistence, policy-aware bounded retry routing for retry-class outcomes, non-retry invalid escalation behavior, and guardrail coverage that provider-error raw text does not bleed into user-facing gate summary fields).
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autopilot_delegated_ingress_regressions.py`
- `check_delegated_gate_summary_queue_regressions.py` — validates queue transition + queue-sync projection of structured delegated gate summaries into handoff packet metadata (`gate_metadata.gate_summary`) and failure signatures, including invalid provider-summary drop-vs-fail-close semantics.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_delegated_gate_summary_queue_regressions.py`
- `check_autopilot_tick_provider_failure_queue_regressions.py` — deterministic integration harness for non-zero tick provider exit classification (`autopilot.provider_failure_summary.v1`) through queue transition + handoff persistence.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autopilot_tick_provider_failure_queue_regressions.py`
- `check_queue_cooldown_authority_regressions.py` — validates autopilot backoff projection into queue cooldown metadata and cooldown-aware claim deferral/reauthorization behavior.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_queue_cooldown_authority_regressions.py`
- `check_codex_request_failed_sanitization.py` — verifies active OpenClaw helper-bundle error formatting no longer leaks raw `Codex request failed (unknown_error).` strings.
  - baseline leak probe: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_codex_request_failed_sanitization.py --expect-leak`
  - post-hotfix verify: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_codex_request_failed_sanitization.py`
- `apply_codex_request_failed_sanitization.py` — hotfix applier for OpenClaw helper bundles (patches all hashed `pi-embedded-helpers-*.js` variants under `dist/` + `dist/plugin-sdk/`).
  - preview only: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/apply_codex_request_failed_sanitization.py --dry-run`
  - apply (root required on system install): `sudo python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/apply_codex_request_failed_sanitization.py`
- `tests/test_autopilot_entrypoints_fixed_now_parity.py` — deterministic helper-present vs helper-absent parity harness for autopilot entrypoints (`hl_autopilot_tick.sh` deferred-wake scheduling + `hl_autopilot_ctl.sh status` due-window projection) under `OPENCLAW_AUTOPILOT_FIXED_NOW_TS`.
  - run: `./.venv/bin/pytest -q /home/yeqiuqiu/clawd-architect/tests/test_autopilot_entrypoints_fixed_now_parity.py`
- `check_autonomy_continuity_regressions.py` — one-command regression cluster runner for autonomy/fixed-now continuity guarantees (action-token guard, verify gate strict autonomy + wrapper policy toggles, GTC latest schema gate fail-close negative regressions, publish-manifest authenticity dual-mode trust regression, publish-transaction lock/crash-recovery fail-close semantics, incident replay verify-gate posture projection integrity, swarm operability contract fail-close/warn-boundary regressions, slot-fill protocol contract regressions, autopilot entrypoint parity, continuity refresh authority, operator surfaces fixed-now, degraded pending stale signal chain).
  - full cluster run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autonomy_continuity_regressions.py --json`
  - minimal smoke checklist: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autonomy_continuity_regressions.py --profile smoke --json`
  - critical-path benchmark subset: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autonomy_continuity_regressions.py --profile critical-path --json`
  - hard better/worse benchmark answer against replayable baseline fixture:
    - `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_autonomy_continuity_regressions.py --profile critical-path --benchmark-baseline /home/yeqiuqiu/clawd-architect/tests/fixtures/a6/autonomy_regression_critical_path_baseline_v1.json --benchmark-output /home/yeqiuqiu/clawd-architect/state/continuity/latest/autonomy_regression_critical_path_benchmark_scorecard_latest.json --json`
    - baseline runs now append rolling benchmark history to `state/continuity/history/autonomy_regression_critical_path_benchmark_history.jsonl` and emit `benchmark_trend` in summary JSON (use `--benchmark-history-path ''` to disable or `--benchmark-trend-window <N>` to tune window size).
  - dispatcher shortcuts:
    - full: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh autonomy-regressions --json`
    - smoke: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh autonomy-smoke --json`
- `db_integrity_check.sh` — continuity SQLite integrity + invariant checker (critical/warn classification), including queue provenance contract/freshness + role-boundary guards.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic age/freshness windows (handoff coverage, parity freshness, scheduler staleness) during replay/regression checks.
- `swarm_runtime_check.sh` — one-command runtime doctor combining operability, slot-fill protocol contract status, DB invariants, continuity-now, GTC latest schema gate, and queue snapshot.
  - honors `OPENCLAW_AUTOPILOT_FIXED_NOW_TS` for deterministic runtime-check `generated_at` timestamps during replay/regression runs.
- `normalize_event_sources.sh` — canonicalizes legacy/non-namespaced `continuity_events.source` values.
- `gtc_v2_sync.sh` — Ground-Truth Connectors v2 sync (connector evidence index + append-only JSONL + latest gateboard/continuity surfaces + `gtc.event.v2` projection + incident replay hints).
  - incident replay latest surface now carries compact verify-gate preflight posture (`incident_replay.verify_gate_preflight`) so replay operators can see strict-autonomy mode/blocker context in the same artifact.
  - latest artifacts now share `build_generation_id`/`valid_until` and publish `state/gtc-v2/latest/publish_manifest.json` as the commit marker for near-atomic reads.
  - latest/surfaces publishes are now staged under `state/gtc-v2/.staging/*` and promoted only after schema-gate success, so schema failures do not overwrite live latest pointers.
  - commit lock now rejects stale writers even on bootstrap (`base_generation_id` vs live generation compare under lock), and also rejects promotion when continuity coherence/policy epoch drifts during the same run (`error_class=stale_coherence`, exit `7`).
  - publish now fail-closes on latest-surface schema violations (exit `3`) using `gtc_latest_schema_check.sh`; break-glass override exists via `--skip-schema-gate`.
  - publish manifest authenticity now supports dual mode with explicit scheme/profile: default `ed25519-sha256` signatures (`gtc.publish_manifest.ed25519.v1`) over canonical payload fields, anchored by `publish_anchor.manifest_auth_root`; legacy `hmac-sha256` (`gtc.publish_manifest.hmac.v1`) remains supported via `OPENCLAW_GTC_PUBLISH_MANIFEST_AUTH_MODE=hmac`.
  - Ed25519 keys default to `state/continuity/secrets/gtc_publish_manifest_ed25519_{private,public}.pem` (env overrides: `OPENCLAW_GTC_PUBLISH_MANIFEST_ED25519_{PRIVATE_KEY_PEM,PUBLIC_KEY_PEM,PRIVATE_KEY_FILE,PUBLIC_KEY_FILE,KEY_ID}`); legacy HMAC env/file overrides remain unchanged (`OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY{,_ID}` / `OPENCLAW_GTC_PUBLISH_MANIFEST_HMAC_KEY_FILE`).
  - publish now writes an explicit transaction journal/state machine under `state/gtc-v2/publish_journal/` (`latest_transaction.json` + `transactions.jsonl`) with step markers (`staging_ready`, `schema_gate_passed|skipped`, `lock_acquired`, `latest_promoted`, `surfaces_promoted`, `promotion_cleanup_complete`, `verified`) and terminal states (`committed|aborted|failed`).
  - startup now performs deterministic recovery for non-terminal publish transactions (`prepared|promoting|verifying`): it acquires the publish lock, reconciles mid-promotion crash windows using staged backup markers, and forces terminalization before new publish attempts.
- `gtc_latest_schema_check.sh` — strict schema validation over generated GTC latest surfaces (`continuity_current`, `gateboard`, `event_projection`, `incident_replay`, `publish_anchor`, `publish_manifest`, and per-connector latest JSON), plus publish-manifest authenticity verification (`publish_manifest_authenticity`) for both `hmac-sha256` and `ed25519-sha256` modes; `latest_paths` provenance/linkage + digest checks run only when authenticity is trusted (expected target mapping + out-of-tree rejection).
- `gtc_incident_replay.sh` — deterministic incident replay builder from GTC `event_projection/open_incidents` + evidence index; reconstructs route/task/checkpoint evidence chain and artifact pack (including typed queue task artifact roles/manifests), with incident-scoped checkpoint filtering by default (`--checkpoint-scope full` for legacy broad expansion).
  - replay bundle JSON/markdown now mirrors compact verify-gate preflight posture (`verify_gate_preflight`) from `state/continuity/latest/continuity_now_latest.json` for trust-context continuity across replay narratives.
- `check_gtc_latest_schema_regressions.py` — local corruption/negative regression harness for intentionally broken GTC latest surfaces (including publish-manifest authenticity tamper + path provenance mismatch/escape cases); asserts strict fail-close (`--strict` non-zero) and non-strict surfaced failure payloads.
  - uses deterministic self-seeded latest fixtures (not `state/gtc-v2/latest`) so strict-gate regressions are stable across environments.
  - includes a fail-fast seed/schema drift sentinel (`seed_schema_fixture_contract`) that checks seed/schema contract coverage (required/const/enum/type/format) before running negative cases and returns actionable `seed_latest_surfaces()` update guidance when schema constraints evolve.
  - includes explicit dual-mode publish-auth coverage in one gate: compatibility `hmac-sha256` valid/tamper plus default `ed25519-sha256` valid/tamper.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_gtc_latest_schema_regressions.py`
- `check_gtc_publish_manifest_auth_regressions.py` — narrow canonical trust harness for publish-manifest authenticity dual-mode verification (compatibility `hmac-sha256` valid + tamper fail-close, default `ed25519-sha256` valid + tamper fail-close) with deterministic self-seeded fixtures reused from the latest-schema regression surface.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_gtc_publish_manifest_auth_regressions.py`
- `check_gtc_publish_transaction_regressions.py` — focused publish lock/crash/recovery harness for staged promotion robustness, including mid-promotion crash-window recovery semantics.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_gtc_publish_transaction_regressions.py`
- `check_gtc_queue_artifact_manifest_regressions.py` — validates queue-task GTC projection of typed task artifact manifests (`task_artifact:<type>` roles + metadata/sha carry-through) into refs/artifact linkage.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_gtc_queue_artifact_manifest_regressions.py`
- `check_gtc_incident_replay_regressions.py` — validates deterministic replay utility reconstruction path (`route/open incident -> evidence chain -> artifact pack`), incident-scoped checkpoint filtering vs full-scope break-glass behavior, typed artifact-role carry-through, and verify-gate preflight posture propagation into incident replay surfaces/bundles.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_gtc_incident_replay_regressions.py`
- `check_swarm_operability_regressions.py` — validates swarm operability contract enforcement boundaries (`check_swarm_operability.sh`) across healthy baseline, missing-role fail-close, malformed-role fail-close, and warning-only runbook snippet drift semantics.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_swarm_operability_regressions.py`
- `check_slot_fill_protocol_regressions.py` — validates slot-fill protocol governance boundaries (`check_slot_fill_protocol.sh`) across healthy baseline, required snippet fail-close, execution-mode/tuple fail-close (`EXECUTE_NOW|PLAN_ONLY` + `execution_mode|worker_lane|model_selection`), warning-only workflow reference drift, and dispatcher route continuity.
  - run: `python3 /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity/check_slot_fill_protocol_regressions.py`
- `../continuity.sh` — dispatcher (`now|current|execution-status|execution-frontier|handover|blocker-registry|mission-control|gate-os|history|reconcile|verify|verify-gate-status|sync|checkpoint|queue-sync|queue-arb|queue-replay|repo-review-closeout|failover-replay-evidence|lock-break|librarian|promotion-gate|model-rollout-gate|model-rollout-health|model-rollout-cost|model-rollout-controller|session-route|session-transport-route|knowledge-queue(legacy)|promotion-queue(canonical)|markdown-gate|material-classify|knowledge-ingest|doc-intake-closeout|release-evidence-gate|lane-crossover-guard|bridge-ingest|db-check|swarm-check|autonomy-regressions|autonomy-smoke|parity-run|web-capture|gtc-sync|gtc-schema-check|gtc-replay|normalize-events`).
- `../architecture/run_competitive_parity_harness.sh` — low-noise parity scorecard runner (queue claim/transition + schema validation + regression/blocker surfacing only).
- `../run_web_capture_macro.sh` — low-noise deterministic web-capture wrapper (cadence + per-domain backoff/login guard + blocker-only event routing).
- `../run_web_capture_scheduler.sh` — deterministic multi-domain fairness scheduler with governed contract-bearing state output.
- `web_capture_domain_guard.sh` — per-domain runtime guard (cooldown/login state + sustained bot-wall escalation + operator contract generation).

## Key state paths
- Checkpoints: `state/continuity/checkpoints/chk_*.json|md`
- Latest pointer: `state/continuity/latest/latest_pointer.json`
- Latest json/md symlinks: `state/continuity/latest/handover_latest.{json,md}`
- Successor current: `state/continuity/current.json`
- Canonical blocker registry (derived): `state/continuity/latest/blocker_registry.json`
- Successor pointer (derived): `state/continuity/pointers/latest.json`
- Canonical continuity read pointer (generation-pinned contract): `state/continuity/latest/continuity_read_pointer.json`
- Successor handover: `state/handover/latest.{json,md}`
- Mission control export: `state/continuity/latest/operator_mission_control.json`
- GateOS snapshot: `state/continuity/latest/gate_os_latest.json`
- Execution status + frontier ledger: `state/continuity/latest/{execution_program_status.json,execution_frontier_ledger.json}`
- Web domain guard state: `state/continuity/latest/web_capture_domain_*.json`
- Web scheduler state: `state/continuity/latest/web_capture_scheduler_state.json`
- Web login operator contracts: `state/continuity/latest/web_capture_login_contract_*.{json,md}`
- Queue replay projection/report: `state/continuity/latest/{queue_replay_projection.json,queue_replay_verify.json}`
- Lock-break latest + audit archive: `state/continuity/latest/lock_break_last.json`, `state/continuity/lock_break/*.json`
- Librarian scaffold outputs: `state/continuity/librarian/{curation_queue.json,promotions.jsonl,canonical_index.json,canonical_index.md,retrieval_hygiene.json}`
- Verify report: `state/continuity/latest/verify_last.json`
- Coherence stamp + policy epoch state: `state/continuity/latest/{coherence_stamp.json,coherence_bundle_latest.json,policy_freshness_state.json}`
- Drift bridge: `state/continuity/latest/runtime_truth_bridge.json`
- Ground truth latest: `state/ground_truth/latest.json`
- GTC v2 evidence streams: `state/gtc-v2/evidence/<connector_type>/<connector_id>/YYYY-MM-DD.jsonl`
- GTC v2 latest readiness pointers: `state/gtc-v2/latest/{continuity_current.json,gateboard.json,event_projection.json,incident_replay.json,publish_manifest.json}`
- GTC deterministic incident bundles: `state/gtc-v2/incident_replay/incident-<incident_id>.{json,md}`
- GTC v2 publish transaction journal: `state/gtc-v2/publish_journal/{latest_transaction.json,transactions.jsonl}`

## Architecture references
- `docs/ops/continuity_queue_state_model_v1.md`
- `docs/ops/latest5_pdf_architecture_first_program_2026-03-08.md`
- `docs/ops/latest8_pdf_architecture_first_program_2026-03-09.md`
- `docs/ops/batch2_pdf_architecture_first_program_2026-03-09.md`
- `ops/openclaw/architecture/README.md`

## Guardrails
- Do not overwrite historical checkpoint files.
- Prefer script outputs over manual markdown edits.
- If verify returns `BLOCKER`, resolve verification failures before any mutating command.
