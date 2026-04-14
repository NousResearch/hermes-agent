# Human-First Observability Ergonomics v1

Date: 2026-03-21
Status: active (Wave 8 C1 Operator Cockpit UX)
Parent doctrine: `docs/ops/unified_operating_doctrine_v1.md`

## 0) Purpose
To close the loop between system failure reports (JSON, log trails, stack traces) and human action. Every failing state must provide a path forward, eliminating the need to search the repo for "how to fix this".

## 1) Ergonomics Principles

### 1. Zero Guessing
Raw JSON dumps are for the machine. The operator must not have to parse JSON fields mentally to figure out the problem. A failing `verify_then_resume.sh` state must be translated into human-readable text.

### 2. Remediation Hints
Every known failure mode must be mapped to a playbook or a specific command.
- If `SLO-1_VERIFY_FRESHNESS` is out of budget, the action is: `Run: bash ops/openclaw/continuity/verify_then_resume.sh`.
- If `A1_CONTROL_PLANE` layer `alive` fails, the action is: `Run: openclaw gateway start`.
- If `A6_OBSERVABILITY_FAILED` triggers, the action is: `Review: docs/ops/incident_playbooks/blindness_recovery.md`.

### 3. Clickable Actions
If the target platform supports inline buttons (e.g., Telegram), the remediation hint must ideally be exposed as a callback button or command link to streamline operator resolution. If CLI, it must be copy-pasteable as a full command (no generic placeholders like `<path/to/script>`).

## 2) Mapping Implementation
The `cockpit_summary.sh` or `cockpit_alert_router.py` will contain a static mapping dictionary linking `error_id` or `layer_failed` to these `remediation_hint`s. These hints are rendered under the `Immediate Action (Remediation)` section of the Cockpit Action Card.
