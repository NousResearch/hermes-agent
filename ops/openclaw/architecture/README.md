# OpenClaw Architecture Substrate (Latest-8 PDF program)

Purpose: keep the control-plane architecture explicit, deterministic, and successor-operable.

This directory is architecture-first scaffolding for:
1. Swarm role contracts (planner/executor/validator/SRE/librarian)
2. Queue/dependency/file-lock state model
3. Deterministic web interaction (fetch-vs-browser routing + macro artifacts)
4. Deterministic UI design copilot pipeline (EDD gates + canonical doc taxonomy)
5. Trading terminal design-language + competitive parity harness contracts
6. Ground-Truth Connectors v2 evidence/index/latest contract (append-only + successor-operable)

## Canonical contract files
- `swarm_role_contracts.v1.yaml`
- `web_interaction_idd.v1.yaml`
- `ui_design_edd.v1.yaml`
- `trading_terminal_design_language.v1.yaml`
- `competitive_parity_harness.v1.yaml`
- `ground_truth_connectors.v2.yaml`

## Schemas
- `schemas/web_capture_macro.schema.json`
- `schemas/design_component_spec_frontmatter.schema.json`
- `schemas/competitive_scorecard.schema.json`
- `schemas/gtc_evidence.schema.json`
- `schemas/gtc_latest.schema.json`
- `schemas/gtc_event.schema.json`
- `schemas/gtc_gateboard.schema.json`
- `schemas/gtc_incident_replay.schema.json`
- `schemas/gtc_publish_anchor.schema.json`
- `schemas/gtc_publish_manifest.schema.json`
- `schemas/gtc_connector_latest.schema.json`
- `schemas/web_capture_bundle_index.schema.json`
- `schemas/continuity_current.schema.json`
- `schemas/handover_latest.schema.json`
- `schemas/operator_mission_control.schema.json`
- `schemas/operator_triage_console.schema.json`
- `schemas/gate_result.schema.json`
- `schemas/web_capture_scheduler_state.schema.json`
- `schemas/queue_stale_wave_auto_remediation.schema.json`

## Queue + lock substrate
- Runtime DB initializer: `ops/openclaw/continuity/init_db.sh`
- Runtime sync: `ops/openclaw/continuity/queue_sync_from_autopilot_json.sh`
- Runtime arbitration/replay: `ops/openclaw/continuity/queue_arbitrator.sh`
- Runtime integrity checker: `ops/openclaw/continuity/db_integrity_check.sh`
- Ground-truth connector sync: `ops/openclaw/continuity/gtc_v2_sync.sh`

## Contract validation
- `ops/openclaw/architecture/validate_contracts.sh --json`
- `ops/openclaw/architecture/check_swarm_operability.sh --json`
- `ops/openclaw/architecture/validate_component_spec.sh --spec <path> --json`
- `ops/web_capture/validate_macro.sh --macro <path> --json`

## Web interaction runtime entrypoints (phase-2 start)
- deterministic macro runner:
  - `bash /home/yeqiuqiu/clawd-architect/ops/web_capture/run_macro.sh --macro <path> --mode auto --json`
  - enforces artifact quality gates (`blank_screenshot_detected`, `dom_capture_missing`) before returning success.
- low-noise wrapper (cadence + per-domain backoff/login guard + blocker routing + queue discipline):
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_web_capture_macro.sh --mode auto`
  - dispatcher: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh web-capture ...`
  - queue task id contract: `continuity:web_capture:<macro_slug>`
  - domain guard state: `state/continuity/latest/web_capture_domain_<domain>.json`
  - login-wall operator contract: `state/continuity/latest/web_capture_login_contract_<domain>.{json,md}`
- fairness scheduler (governed state surface):
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_web_capture_scheduler.sh --dry-run --json`
  - dispatcher: `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh web-capture-scheduler --dry-run --json`
  - scheduler state contract: `state/continuity/latest/web_capture_scheduler_state.json`
  - schema: `ops/openclaw/architecture/schemas/web_capture_scheduler_state.schema.json`

## Competitive parity operational runner (low-noise)
- Manual/on-demand:
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/architecture/run_competitive_parity_harness.sh --json`
- Through continuity dispatcher:
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh parity-run --json`
- Periodic wrapper (silent unless blocker):
  - `bash /home/yeqiuqiu/clawd-architect/ops/openclaw/run_competitive_parity_harness.sh`
    - cadence-gated to weekly by default (`OPENCLAW_COMPETITIVE_PARITY_MIN_INTERVAL_SEC=604800`),
      with `--force` available for manual override.

Runner behavior:
- Claims queue task `parity:weekly_harness` via `queue_arbitrator.sh claim` (non-autopilot producer path).
- Validates discovered/provided scorecards against `schemas/competitive_scorecard.schema.json`.
- Writes scorecard/dashboard artifacts under `state/architecture/competitive_parity/`.
- Periodic wrapper records cadence state at `state/cron_watchdog/competitive_parity_schedule_state.json`.
- Emits blocker/regression events only (no routine success spam).

## Non-goals
- This directory does not enable broad autonomous crawling or product-code mutation by itself.
- Activation remains gated by explicit runbooks, command invocation, and continuity checks.
