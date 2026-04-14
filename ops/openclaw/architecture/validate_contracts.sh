#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
ARCH_DIR="$ROOT/ops/openclaw/architecture"
JSON_OUT=0

usage() {
  cat <<'EOF'
Usage: validate_contracts.sh [options]

Validate architecture contract pack presence + parseability + schema conformance.

Options:
  --json    JSON output
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

python3 - "$ARCH_DIR" "$ROOT" "$JSON_OUT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List

arch_dir = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
json_out = bool(int(sys.argv[3]))

try:
    import yaml
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"pyyaml_missing:{exc}"}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

try:
    import jsonschema
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"jsonschema_missing:{exc}"}, ensure_ascii=False, indent=2))
    raise SystemExit(1)

contracts = [
    "swarm_role_contracts.v1.yaml",
    "web_interaction_idd.v1.yaml",
    "ui_design_edd.v1.yaml",
    "trading_terminal_design_language.v1.yaml",
    "competitive_parity_harness.v1.yaml",
    "ground_truth_connectors.v2.yaml",
]
schemas = [
    "schemas/web_capture_macro.schema.json",
    "schemas/design_component_spec_frontmatter.schema.json",
    "schemas/competitive_scorecard.schema.json",
    "schemas/gtc_evidence.schema.json",
    "schemas/gtc_latest.schema.json",
    "schemas/gtc_event.schema.json",
    "schemas/gtc_gateboard.schema.json",
    "schemas/gtc_connector_latest.schema.json",
    "schemas/gtc_incident_replay.schema.json",
    "schemas/gtc_publish_manifest.schema.json",
    "schemas/handover_latest.schema.json",
    "schemas/operator_mission_control.schema.json",
    "schemas/operator_triage_console.schema.json",
    "schemas/blocker_registry.schema.json",
    "schemas/web_capture_scheduler_state.schema.json",
    "schemas/queue_stale_wave_auto_remediation.schema.json",
    "schemas/no_nudge_continuity_cron_guard_summary.schema.json",
    "schemas/swarm_operability_check.schema.json",
    "schemas/slot_fill_protocol_check.schema.json",
    "schemas/failover_fsm_state_snapshot.schema.json",
    "schemas/failover_fsm_reset_readiness_report.schema.json",
    "schemas/failover_fsm_successor_resume_validation_report.schema.json",
    "schemas/successor_safe_handover_proof.schema.json",
    "schemas/successor_safe_handover_proof_status.schema.json",
    "schemas/load_shedding_signal_snapshot.schema.json",
    "schemas/load_shedding_decision.schema.json",
    "schemas/load_shedding_escape_worker_report.schema.json",
    "schemas/load_shedding_watchdog_probe.schema.json",
]

wave4_contract_docs = [
    "docs/ops/lane_topology_authority_contract_v1.md",
    "docs/ops/lane_boundary_contract_v1.md",
    "docs/ops/controlled_cross_lane_bridge_contract_v1.md",
    "docs/ops/orchestrator_api_contract_v1.md",
    "docs/ops/doctrine_object_contract_v1.md",
    "docs/ops/promotion_protocol_contract_v1.md",
    "docs/ops/anti_drift_waiver_contract_v1.md",
    "docs/ops/source_of_truth_map_guard_policy_contract_v1.md",
]

wave4_schemas = [
    "docs/ops/schemas/lane_topology_authority_contract.schema.json",
    "docs/ops/schemas/lane_crossover_packet.schema.json",
    "docs/ops/schemas/cross_lane_bridge_object.schema.json",
    "docs/ops/schemas/orchestrator_snapshot_resolve.schema.json",
    "docs/ops/schemas/orchestrator_plan.schema.json",
    "docs/ops/schemas/orchestrator_run.schema.json",
    "docs/ops/schemas/orchestrator_event_stream.schema.json",
    "docs/ops/schemas/orchestrator_replay_resync.schema.json",
    "docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json",
    "docs/ops/schemas/doctrine_object.schema.json",
    "docs/ops/schemas/promotion_candidate.schema.json",
    "docs/ops/schemas/mutation_attestation.schema.json",
    "docs/ops/schemas/lane_action_intent.schema.json",
    "docs/ops/schemas/b7_candidate_opportunity_surface.v1.schema.json",
    "docs/ops/schemas/proposal_archive_packet.v1.schema.json",
    "docs/ops/schemas/test_gap_packet.v1.schema.json",
    "docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json",
    "docs/ops/schemas/anti_drift_waiver_register.schema.json",
    "docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json",
]

wave4_template_schema_checks = [
    {
        "template": "docs/ops/templates/lane_topology_authority_contract.template.json",
        "schema": "docs/ops/schemas/lane_topology_authority_contract.schema.json",
    },
    {
        "template": "docs/ops/templates/mutation_attestation.template.json",
        "schema": "docs/ops/schemas/mutation_attestation.schema.json",
    },
    {
        "template": "docs/ops/templates/lane_action_intent.template.json",
        "schema": "docs/ops/schemas/lane_action_intent.schema.json",
    },
    {
        "template": "docs/ops/templates/b7_candidate_opportunity_surface.v1.template.json",
        "schema": "docs/ops/schemas/b7_candidate_opportunity_surface.v1.schema.json",
    },
    {
        "template": "docs/ops/templates/proposal_archive_packet.v1.template.json",
        "schema": "docs/ops/schemas/proposal_archive_packet.v1.schema.json",
    },
    {
        "template": "docs/ops/templates/test_gap_packet.v1.template.json",
        "schema": "docs/ops/schemas/test_gap_packet.v1.schema.json",
    },
    {
        "template": "docs/ops/templates/lane_crossover_signal.template.json",
        "schema": "docs/ops/schemas/lane_crossover_packet.schema.json",
    },
    {
        "template": "docs/ops/templates/lane_crossover_ticket.template.json",
        "schema": "docs/ops/schemas/lane_crossover_packet.schema.json",
    },
    {
        "template": "docs/ops/templates/lane_crossover_deep_review.template.json",
        "schema": "docs/ops/schemas/lane_crossover_packet.schema.json",
    },
    {
        "template": "docs/ops/templates/cross_lane_bridge_object.template.json",
        "schema": "docs/ops/schemas/cross_lane_bridge_object.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_snapshot_resolve.template.json",
        "schema": "docs/ops/schemas/orchestrator_snapshot_resolve.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_plan.template.json",
        "schema": "docs/ops/schemas/orchestrator_plan.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_run.template.json",
        "schema": "docs/ops/schemas/orchestrator_run.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_event_stream.template.json",
        "schema": "docs/ops/schemas/orchestrator_event_stream.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_replay_resync.template.json",
        "schema": "docs/ops/schemas/orchestrator_replay_resync.schema.json",
    },
    {
        "template": "docs/ops/templates/orchestrator_contract_bridge_packet.template.json",
        "schema": "docs/ops/schemas/orchestrator_contract_bridge_packet.schema.json",
    },
    {
        "template": "docs/ops/templates/doctrine_object.template.json",
        "schema": "docs/ops/schemas/doctrine_object.schema.json",
    },
    {
        "template": "docs/ops/templates/promotion_candidate.template.json",
        "schema": "docs/ops/schemas/promotion_candidate.schema.json",
    },
    {
        "template": "docs/ops/templates/anti_drift_waiver_register.template.json",
        "schema": "docs/ops/schemas/anti_drift_waiver_register.schema.json",
    },
    {
        "template": "docs/ops/templates/source_of_truth_map_guard_policy_pack.template.json",
        "schema": "docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json",
    },
]

checks: List[Dict[str, Any]] = []
errors: List[str] = []


def to_json_safe(value):
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return value


def add(file: str, ok: bool, error: str = "", extra: Dict[str, Any] = None):
    row: Dict[str, Any] = {"file": file, "ok": bool(ok)}
    if error:
        row["error"] = error
        errors.append(f"invalid:{file}:{error}")
    if extra:
        row.update(extra)
    checks.append(row)


def validate_jsonschema(payload: Any, schema_rel: str, *, file_rel: str) -> None:
    schema_obj = all_schema_docs.get(schema_rel)
    if not isinstance(schema_obj, dict):
        raise ValueError(f"missing_schema_doc:{schema_rel}")
    jsonschema.validate(payload, schema_obj)
    add(
        file_rel,
        True,
        extra={
            "validation": "jsonschema",
            "schema": schema_rel,
        },
    )


runtime_env = {**os.environ, "OPENCLAW_ROOT": str(root)}


schema_docs: Dict[str, Dict[str, Any]] = {}
wave4_schema_docs: Dict[str, Dict[str, Any]] = {}
all_schema_docs: Dict[str, Dict[str, Any]] = {}

for rel in contracts:
    p = arch_dir / rel
    if not p.exists():
        checks.append({"file": rel, "ok": False, "error": "missing"})
        errors.append(f"missing:{rel}")
        continue
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top_level_not_object")
        for key in ("version", "id", "status"):
            if key not in data:
                raise ValueError(f"missing_key:{key}")
        checks.append({"file": rel, "ok": True})
    except Exception as exc:
        checks.append({"file": rel, "ok": False, "error": str(exc)})
        errors.append(f"invalid:{rel}:{exc}")

for rel in schemas:
    p = arch_dir / rel
    if not p.exists():
        checks.append({"file": rel, "ok": False, "error": "missing"})
        errors.append(f"missing:{rel}")
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top_level_not_object")
        for key in ("$schema", "$id", "type"):
            if key not in data:
                raise ValueError(f"missing_key:{key}")
        schema_docs[rel] = data
        checks.append({"file": rel, "ok": True})
    except Exception as exc:
        checks.append({"file": rel, "ok": False, "error": str(exc)})
        errors.append(f"invalid:{rel}:{exc}")

for rel in wave4_contract_docs:
    p = root / rel
    if not p.exists():
        checks.append({"file": rel, "ok": False, "error": "missing"})
        errors.append(f"missing:{rel}")
        continue
    try:
        text = p.read_text(encoding="utf-8")
        first_non_empty = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_non_empty:
            raise ValueError("empty_markdown")
        if not first_non_empty.startswith("#"):
            raise ValueError("missing_heading")
        checks.append({"file": rel, "ok": True, "validation": "markdown_contract"})
    except Exception as exc:
        checks.append({"file": rel, "ok": False, "error": str(exc), "validation": "markdown_contract"})
        errors.append(f"invalid:{rel}:{exc}")

for rel in wave4_schemas:
    p = root / rel
    if not p.exists():
        checks.append({"file": rel, "ok": False, "error": "missing"})
        errors.append(f"missing:{rel}")
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top_level_not_object")
        for key in ("$schema", "$id", "type"):
            if key not in data:
                raise ValueError(f"missing_key:{key}")
        wave4_schema_docs[rel] = data
        checks.append({"file": rel, "ok": True, "validation": "schema_doc"})
    except Exception as exc:
        checks.append({"file": rel, "ok": False, "error": str(exc), "validation": "schema_doc"})
        errors.append(f"invalid:{rel}:{exc}")

all_schema_docs.update(schema_docs)
all_schema_docs.update(wave4_schema_docs)

dependency_policy_pack_rel = "state/continuity/latest/core_roadmap_dependency_unblock_policy_pack_v1.json"
dependency_policy_pack_schema_rel = "docs/ops/schemas/core_roadmap_dependency_unblock_policy_pack.schema.json"
dependency_policy_pack_path = root / dependency_policy_pack_rel
if not dependency_policy_pack_path.exists():
    checks.append({"file": dependency_policy_pack_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{dependency_policy_pack_rel}")
else:
    try:
        dependency_policy_pack_obj = json.loads(dependency_policy_pack_path.read_text(encoding="utf-8"))
        validate_jsonschema(
            dependency_policy_pack_obj,
            dependency_policy_pack_schema_rel,
            file_rel=dependency_policy_pack_rel,
        )
    except Exception as exc:
        checks.append(
            {
                "file": dependency_policy_pack_rel,
                "ok": False,
                "error": str(exc),
                "validation": "jsonschema",
                "schema": dependency_policy_pack_schema_rel,
            }
        )
        errors.append(f"invalid:{dependency_policy_pack_rel}:{exc}")

anti_drift_waiver_rel = "state/continuity/latest/anti_drift_waiver_register.json"
anti_drift_waiver_schema_rel = "docs/ops/schemas/anti_drift_waiver_register.schema.json"
anti_drift_waiver_path = root / anti_drift_waiver_rel
if not anti_drift_waiver_path.exists():
    checks.append({"file": anti_drift_waiver_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{anti_drift_waiver_rel}")
else:
    try:
        anti_drift_waiver_obj = json.loads(anti_drift_waiver_path.read_text(encoding="utf-8"))
        validate_jsonschema(
            anti_drift_waiver_obj,
            anti_drift_waiver_schema_rel,
            file_rel=anti_drift_waiver_rel,
        )
    except Exception as exc:
        checks.append(
            {
                "file": anti_drift_waiver_rel,
                "ok": False,
                "error": str(exc),
                "validation": "jsonschema",
                "schema": anti_drift_waiver_schema_rel,
            }
        )
        errors.append(f"invalid:{anti_drift_waiver_rel}:{exc}")

source_of_truth_map_guard_policy_rel = "state/continuity/latest/source_of_truth_map_guard_policy_v1.json"
source_of_truth_map_guard_policy_schema_rel = "docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json"
source_of_truth_map_guard_policy_path = root / source_of_truth_map_guard_policy_rel
if not source_of_truth_map_guard_policy_path.exists():
    checks.append({"file": source_of_truth_map_guard_policy_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{source_of_truth_map_guard_policy_rel}")
else:
    try:
        source_of_truth_map_guard_policy_obj = json.loads(
            source_of_truth_map_guard_policy_path.read_text(encoding="utf-8")
        )
        validate_jsonschema(
            source_of_truth_map_guard_policy_obj,
            source_of_truth_map_guard_policy_schema_rel,
            file_rel=source_of_truth_map_guard_policy_rel,
        )
    except Exception as exc:
        checks.append(
            {
                "file": source_of_truth_map_guard_policy_rel,
                "ok": False,
                "error": str(exc),
                "validation": "jsonschema",
                "schema": source_of_truth_map_guard_policy_schema_rel,
            }
        )
        errors.append(f"invalid:{source_of_truth_map_guard_policy_rel}:{exc}")

for check in wave4_template_schema_checks:
    template_rel = str(check.get("template") or "").strip()
    schema_rel = str(check.get("schema") or "").strip()
    if not template_rel or not schema_rel:
        continue

    template_path = root / template_rel
    if not template_path.exists():
        checks.append({"file": template_rel, "ok": False, "error": "missing"})
        errors.append(f"missing:{template_rel}")
        continue

    try:
        template_obj = json.loads(template_path.read_text(encoding="utf-8"))
        validate_jsonschema(template_obj, schema_rel, file_rel=template_rel)
    except Exception as exc:
        checks.append(
            {
                "file": template_rel,
                "ok": False,
                "error": str(exc),
                "validation": "jsonschema",
                "schema": schema_rel,
            }
        )
        errors.append(f"invalid:{template_rel}:{exc}")

# Canonical web macro schema validation.
macro_rel = "ops/web_capture/macros/bybit_derivatives_capture.yaml"
macro_path = root / macro_rel
if not macro_path.exists():
    checks.append({"file": macro_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{macro_rel}")
else:
    try:
        macro_obj = to_json_safe(yaml.safe_load(macro_path.read_text(encoding="utf-8")))
        schema_obj = schema_docs.get("schemas/web_capture_macro.schema.json")
        if not isinstance(schema_obj, dict):
            raise ValueError("missing_macro_schema_doc")
        jsonschema.validate(macro_obj, schema_obj)
        checks.append({"file": macro_rel, "ok": True, "validation": "jsonschema"})
    except Exception as exc:
        checks.append({"file": macro_rel, "ok": False, "error": str(exc), "validation": "jsonschema"})
        errors.append(f"invalid:{macro_rel}:{exc}")

# Runtime dry-run check for deterministic web-capture runner.
web_runner_rel = "ops/web_capture/run_macro.sh"
web_runner_path = root / web_runner_rel
if not web_runner_path.exists():
    checks.append({"file": web_runner_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{web_runner_rel}")
else:
    try:
        cp = subprocess.run(
            ["bash", str(web_runner_path), "--macro", str(macro_path), "--mode", "auto", "--dry-run", "--json"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        payload = json.loads(cp.stdout or "{}")
        ok = cp.returncode == 0 and bool(payload.get("ok"))
        checks.append(
            {
                "file": web_runner_rel,
                "ok": ok,
                "validation": "runtime_dry_run",
                "route": payload.get("route"),
                "status": payload.get("status"),
            }
        )
        if not ok:
            errors.append(f"invalid:{web_runner_rel}:dry_run_failed")
    except Exception as exc:
        checks.append({"file": web_runner_rel, "ok": False, "error": str(exc), "validation": "runtime_dry_run"})
        errors.append(f"invalid:{web_runner_rel}:{exc}")

# Scheduler runtime + state contract dry-run check.
scheduler_runner_rel = "ops/openclaw/run_web_capture_scheduler.sh"
scheduler_runner_path = root / scheduler_runner_rel
scheduler_probe_state_rel = "state/continuity/latest/web_capture_scheduler_contract_probe.json"
scheduler_probe_state_path = root / scheduler_probe_state_rel
if not scheduler_runner_path.exists():
    checks.append({"file": scheduler_runner_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{scheduler_runner_rel}")
else:
    try:
        cp = subprocess.run(
            [
                "bash",
                str(scheduler_runner_path),
                "--macro",
                str(macro_path),
                "--dry-run",
                "--state",
                str(scheduler_probe_state_path),
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=60,
        )
        payload = json.loads(cp.stdout or "{}")
        ok = cp.returncode == 0 and bool(payload.get("ok"))
        checks.append(
            {
                "file": scheduler_runner_rel,
                "ok": ok,
                "validation": "runtime_dry_run",
                "status": payload.get("status"),
                "state_path": payload.get("state_path") or scheduler_probe_state_rel,
            }
        )
        if not ok:
            errors.append(f"invalid:{scheduler_runner_rel}:dry_run_failed")
        else:
            schema_obj = schema_docs.get("schemas/web_capture_scheduler_state.schema.json")
            if not isinstance(schema_obj, dict):
                raise ValueError("missing_web_capture_scheduler_state_schema_doc")
            state_obj = json.loads(scheduler_probe_state_path.read_text(encoding="utf-8"))
            jsonschema.validate(state_obj, schema_obj)
            checks.append(
                {
                    "file": scheduler_probe_state_rel,
                    "ok": True,
                    "validation": "jsonschema",
                    "schema": "schemas/web_capture_scheduler_state.schema.json",
                }
            )
    except Exception as exc:
        checks.append({"file": scheduler_runner_rel, "ok": False, "error": str(exc), "validation": "runtime_dry_run"})
        errors.append(f"invalid:{scheduler_runner_rel}:{exc}")
    finally:
        try:
            if scheduler_probe_state_path.exists():
                scheduler_probe_state_path.unlink()
        except Exception:
            pass

# Runtime operator-surface contract checks.
handover_runner_rel = "ops/openclaw/continuity/handover_latest.sh"
handover_runner_path = root / handover_runner_rel
if not handover_runner_path.exists():
    checks.append({"file": handover_runner_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{handover_runner_rel}")
else:
    try:
        cp = subprocess.run(
            ["bash", str(handover_runner_path), "--json"],
            text=True,
            capture_output=True,
            timeout=90,
            cwd=str(root),
            env=runtime_env,
        )
        payload = json.loads(cp.stdout or "{}")
        handover_json_rel = str(payload.get("handover_json") or "").strip() if isinstance(payload, dict) else ""
        if not handover_json_rel:
            fallback_handover_path = root / "state" / "handover" / "latest.json"
            if fallback_handover_path.exists():
                handover_json_rel = "state/handover/latest.json"
        ok = bool(handover_json_rel)
        add(
            handover_runner_rel,
            ok,
            "" if ok else f"runtime_contract_probe_failed:returncode={cp.returncode}",
            extra={
                "validation": "runtime_dry_run",
                "probe_returncode": cp.returncode,
                "handover_json": handover_json_rel or (payload.get("handover_json") if isinstance(payload, dict) else None),
                "stale": payload.get("stale") if isinstance(payload, dict) else None,
                "used_fallback_existing_payload": bool((not isinstance(payload, dict)) or (not payload.get("handover_json"))) and bool(handover_json_rel),
            },
        )
        if ok:
            handover_payload_rel = str(payload.get("handover_json") or "state/handover/latest.json").strip() or "state/handover/latest.json"
            handover_payload_path = (root / handover_payload_rel).resolve()
            if not handover_payload_path.exists():
                raise ValueError(f"handover_payload_missing:{handover_payload_rel}")
            handover_payload = json.loads(handover_payload_path.read_text(encoding="utf-8"))
            try:
                validate_jsonschema(
                    handover_payload,
                    "schemas/handover_latest.schema.json",
                    file_rel=handover_payload_rel,
                )
            except Exception as exc:
                add(
                    handover_payload_rel,
                    False,
                    str(exc),
                    extra={
                        "validation": "jsonschema",
                        "schema": "schemas/handover_latest.schema.json",
                    },
                )
    except Exception as exc:
        add(handover_runner_rel, False, str(exc), extra={"validation": "runtime_dry_run"})

mission_control_runner_rel = "ops/openclaw/continuity/operator_mission_control.sh"
mission_control_runner_path = root / mission_control_runner_rel
mission_control_payload_rel = "state/continuity/latest/operator_mission_control.json"
if not mission_control_runner_path.exists():
    checks.append({"file": mission_control_runner_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{mission_control_runner_rel}")
else:
    try:
        cp = subprocess.run(
            ["bash", str(mission_control_runner_path), "--json"],
            text=True,
            capture_output=True,
            timeout=120,
            cwd=str(root),
            env=runtime_env,
        )
        payload = json.loads(cp.stdout or "{}")
        ok = cp.returncode == 0 and isinstance(payload, dict) and bool(payload.get("schema"))
        add(
            mission_control_runner_rel,
            ok,
            "" if ok else f"runtime_contract_probe_failed:returncode={cp.returncode}",
            extra={
                "validation": "runtime_dry_run",
                "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
            },
        )
        if ok:
            mission_control_payload_path = (root / mission_control_payload_rel).resolve()
            if not mission_control_payload_path.exists():
                raise ValueError(f"mission_control_payload_missing:{mission_control_payload_rel}")
            mission_control_payload = json.loads(mission_control_payload_path.read_text(encoding="utf-8"))
            try:
                validate_jsonschema(
                    mission_control_payload,
                    "schemas/operator_mission_control.schema.json",
                    file_rel=mission_control_payload_rel,
                )
            except Exception as exc:
                add(
                    mission_control_payload_rel,
                    False,
                    str(exc),
                    extra={
                        "validation": "jsonschema",
                        "schema": "schemas/operator_mission_control.schema.json",
                    },
                )
    except Exception as exc:
        add(mission_control_runner_rel, False, str(exc), extra={"validation": "runtime_dry_run"})

triage_console_runner_rel = "ops/openclaw/continuity/operator_triage_console.sh"
triage_console_runner_path = root / triage_console_runner_rel
triage_console_payload_rel = "state/continuity/latest/operator_triage_console.json"
if not triage_console_runner_path.exists():
    checks.append({"file": triage_console_runner_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{triage_console_runner_rel}")
else:
    try:
        cp = subprocess.run(
            ["bash", str(triage_console_runner_path), "--json"],
            text=True,
            capture_output=True,
            timeout=120,
            cwd=str(root),
            env=runtime_env,
        )
        payload = json.loads(cp.stdout or "{}")
        ok = cp.returncode == 0 and isinstance(payload, dict) and bool(payload.get("schema"))
        add(
            triage_console_runner_rel,
            ok,
            "" if ok else f"runtime_contract_probe_failed:returncode={cp.returncode}",
            extra={
                "validation": "runtime_dry_run",
                "generated_at": payload.get("generated_at") if isinstance(payload, dict) else None,
            },
        )
        if ok:
            triage_console_payload_path = (root / triage_console_payload_rel).resolve()
            if not triage_console_payload_path.exists():
                raise ValueError(f"triage_console_payload_missing:{triage_console_payload_rel}")
            triage_console_payload = json.loads(triage_console_payload_path.read_text(encoding="utf-8"))
            try:
                validate_jsonschema(
                    triage_console_payload,
                    "schemas/operator_triage_console.schema.json",
                    file_rel=triage_console_payload_rel,
                )
            except Exception as exc:
                add(
                    triage_console_payload_rel,
                    False,
                    str(exc),
                    extra={
                        "validation": "jsonschema",
                        "schema": "schemas/operator_triage_console.schema.json",
                    },
                )
    except Exception as exc:
        add(triage_console_runner_rel, False, str(exc), extra={"validation": "runtime_dry_run"})

# Component frontmatter template schema validation.
component_rel = "ops/openclaw/architecture/templates/component_spec_template.md"
component_path = root / component_rel
if not component_path.exists():
    checks.append({"file": component_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{component_rel}")
else:
    try:
        text = component_path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError("missing_frontmatter_start")
        parts = text.split("\n---\n", 1)
        if len(parts) != 2:
            raise ValueError("missing_frontmatter_end")
        frontmatter = to_json_safe(yaml.safe_load(parts[0].replace("---\n", "", 1)))
        schema_obj = schema_docs.get("schemas/design_component_spec_frontmatter.schema.json")
        if not isinstance(schema_obj, dict):
            raise ValueError("missing_component_frontmatter_schema_doc")
        jsonschema.validate(frontmatter, schema_obj)
        checks.append({"file": component_rel, "ok": True, "validation": "frontmatter_jsonschema"})
    except Exception as exc:
        checks.append({"file": component_rel, "ok": False, "error": str(exc), "validation": "frontmatter_jsonschema"})
        errors.append(f"invalid:{component_rel}:{exc}")

# Competitive scorecard template schema validation.
scorecard_rel = "ops/openclaw/architecture/templates/competitive_scorecard_template.json"
scorecard_path = root / scorecard_rel
if not scorecard_path.exists():
    checks.append({"file": scorecard_rel, "ok": False, "error": "missing"})
    errors.append(f"missing:{scorecard_rel}")
else:
    try:
        score_obj = json.loads(scorecard_path.read_text(encoding="utf-8"))
        schema_obj = schema_docs.get("schemas/competitive_scorecard.schema.json")
        if not isinstance(schema_obj, dict):
            raise ValueError("missing_competitive_scorecard_schema_doc")
        jsonschema.validate(score_obj, schema_obj)
        checks.append({"file": scorecard_rel, "ok": True, "validation": "jsonschema"})
    except Exception as exc:
        checks.append({"file": scorecard_rel, "ok": False, "error": str(exc), "validation": "jsonschema"})
        errors.append(f"invalid:{scorecard_rel}:{exc}")

# Swarm operability executable checks.
swarm_check_script = root / "ops" / "openclaw" / "architecture" / "check_swarm_operability.sh"
if not swarm_check_script.exists():
    checks.append({"file": "ops/openclaw/architecture/check_swarm_operability.sh", "ok": False, "error": "missing"})
    errors.append("missing:ops/openclaw/architecture/check_swarm_operability.sh")
else:
    try:
        cp = subprocess.run(
            ["bash", str(swarm_check_script), "--json"],
            text=True,
            capture_output=True,
            timeout=30,
        )
        payload = json.loads(cp.stdout or "{}")
        ok = cp.returncode == 0 and bool(payload.get("ok"))
        checks.append(
            {
                "file": "ops/openclaw/architecture/check_swarm_operability.sh",
                "ok": ok,
                "validation": "executable_check",
                "critical_failures": payload.get("critical_failures"),
                "warn_failures": payload.get("warn_failures"),
            }
        )
        if not ok:
            errors.append(
                "invalid:ops/openclaw/architecture/check_swarm_operability.sh:failed"
            )
    except Exception as exc:
        checks.append(
            {
                "file": "ops/openclaw/architecture/check_swarm_operability.sh",
                "ok": False,
                "error": str(exc),
                "validation": "executable_check",
            }
        )
        errors.append(f"invalid:ops/openclaw/architecture/check_swarm_operability.sh:{exc}")

out = {
    "ok": len(errors) == 0,
    "contract_count": len(contracts),
    "schema_count": len(schemas),
    "wave4_contract_doc_count": len(wave4_contract_docs),
    "wave4_schema_count": len(wave4_schemas),
    "wave4_template_count": len(wave4_template_schema_checks),
    "errors": errors,
    "checks": checks,
}

if json_out:
    print(json.dumps(out, ensure_ascii=False, indent=2))
else:
    print("ARCHITECTURE CONTRACT CHECK")
    print(f"- ok: {out['ok']}")
    print(f"- checks: {len(checks)}")
    print(f"- errors: {len(errors)}")

if not out["ok"]:
    raise SystemExit(1)
PY
