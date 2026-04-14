# Latest-5 PDF Architecture-First Program (Control-plane)

Date: 2026-03-08
Scope: `/home/yeqiuqiu/clawd-architect` only (control-plane, continuity, workflow, governed knowledge)

## Why this exists
Operator correction: build architecture substrate first, then capability utilization.

This document codifies the substrate layers extracted from the 5-PDF pack:
- Native swarm architecture
- deterministic web interaction copilot
- deterministic UI design copilot pipeline
- trading terminal design language
- trading terminal competitive analysis

Canonical source pack references:
- `memory/inbound_pdfs/latest5_architecture_pack_2026-03-08/INDEX.md`
- `memory/inbound_pdfs/latest5_architecture_pack_2026-03-08/MANIFEST.json`
- `reports/latest5_architecture_pack_canonical_synthesis_2026-03-08.md`

## Substrate layers (in order)

### Layer 1 — Swarm role contracts (planner/executor/validator/SRE/librarian)
Canonical files:
- `ops/openclaw/architecture/swarm_role_contracts.v1.yaml`
- `docs/ops/swarm_operating_contract_runbook_v1.md`

Contract highlights:
- explicit mandates + forbidden actions by role
- role-specific allowed write surfaces
- required handoff packet fields
- no-merge-without-validator policy

### Layer 2 — Queue/dependency/file-lock state model
Runtime substrate:
- `ops/openclaw/continuity/init_db.sh`
- `ops/openclaw/continuity/queue_sync_from_autopilot_json.sh`
- `ops/openclaw/continuity/queue_arbitrator.sh`
- `ops/openclaw/continuity/db_integrity_check.sh`

Implemented primitives:
- `task_dependencies` table for blocking DAG edges
- `task_file_targets` table for declared mutation scopes
- `file_locks` table with ACTIVE/RELEASED/EXPIRED states
- `task_artifacts` table for deterministic evidence references
- arbitration commands: ready-list/claim/transition/trace/locks
- runtime adoption:
  - `ops/autopilot/bin/hl_autopilot_tick.sh` uses lock-aware `queue_arbitrator.sh claim` before step launch and emits queue transitions on complete/retry/block paths.
  - `ops/openclaw/architecture/run_competitive_parity_harness.sh` provides an additional non-autopilot producer using the same queue claim/transition lock discipline (`task_id=parity:weekly_harness`).

### Layer 3 — Deterministic web interaction architecture (IDD)
Canonical files:
- `ops/openclaw/architecture/web_interaction_idd.v1.yaml`
- `ops/openclaw/architecture/schemas/web_capture_macro.schema.json`
- `ops/web_capture/README.md`
- `ops/web_capture/macros/bybit_derivatives_capture.yaml`

Contract highlights:
- fetch-vs-browser routing policy
- bot/login/region gating detection contract
- ARIA-first selector policy with fallback hierarchy
- artifact bundle requirements and reliability thresholds

### Layer 4 — Deterministic UI design copilot architecture (EDD)
Canonical files:
- `ops/openclaw/architecture/ui_design_edd.v1.yaml`
- `ops/openclaw/architecture/schemas/design_component_spec_frontmatter.schema.json`
- `ops/openclaw/architecture/templates/component_spec_template.md`
- `obsvault_yq_terminal/04_Gates/Design_Copilot_Gate_Matrix_v1.md`

Contract highlights:
- taxonomy for tokens/components/contracts/guidelines/teardowns
- strict component spec frontmatter schema
- ordered gate stack (G1..G6)
- zero-tolerance a11y + runtime error policy

### Layer 5 — Trading terminal design-language + competitive parity contracts
Canonical files:
- `ops/openclaw/architecture/trading_terminal_design_language.v1.yaml`
- `ops/openclaw/architecture/competitive_parity_harness.v1.yaml`
- `ops/openclaw/architecture/schemas/competitive_scorecard.schema.json`
- `ops/openclaw/architecture/templates/competitive_scorecard_template.json`
- `ops/openclaw/architecture/run_competitive_parity_harness.sh`

Contract highlights:
- 8pt/4pt spatial system + explicit density modes
- typography split (UI sans, data mono tabular)
- 3-tier token architecture and semantic discipline
- parity scorecard dimensions + acceptance tests

## Execution order policy
1. Contract/spec/schema changes first.
2. Queue + traceability substrate second.
3. Only then broaden active capability utilization.

## Successor checklist
- Validate DB integrity:
  - `bash ops/openclaw/continuity/db_integrity_check.sh --strict --json`
- Sync queue state:
  - `bash ops/openclaw/continuity/queue_sync_from_autopilot_json.sh`
- Inspect queue trace before interventions:
  - `bash ops/openclaw/continuity/queue_arbitrator.sh trace --task-id autopilot:apply_fixes --json`
- Treat architecture contracts in `ops/openclaw/architecture/` as canonical source for future automation expansion.
- Validate executable swarm operability contract:
  - `bash ops/openclaw/architecture/check_swarm_operability.sh --json`
- Validate scaffold/runtime entrypoints before broader expansion:
  - `bash ops/openclaw/architecture/validate_component_spec.sh --json`
  - `bash ops/web_capture/validate_macro.sh --json`
  - `bash ops/web_capture/run_macro.sh --macro ops/web_capture/macros/example_domain_capture.yaml --mode auto --json`
  - `bash ops/openclaw/run_web_capture_macro.sh --macro ops/web_capture/macros/example_domain_capture.yaml --dry-run --json`
