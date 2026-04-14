#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

JSON_MODE=0
if [ "${1:-}" = "--json" ]; then
  JSON_MODE=1
  shift
fi
if [ "$#" -ne 0 ]; then
  echo "usage: $0 [--json]" >&2
  exit 64
fi

ROUTING_PREFLIGHT_REFRESH_ROOT="$ROOT" ROUTING_PREFLIGHT_REFRESH_JSON_MODE="$JSON_MODE" python3 - <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_cmd(root: pathlib.Path, cmd: list[str], out_path: pathlib.Path, timeout: int = 600) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=os.environ.copy(),
    )
    out_text = str(cp.stdout or "")
    out_path.write_text(out_text, encoding="utf-8")
    parsed = None
    parse_error = None
    if out_text.strip():
        try:
            loaded = json.loads(out_text)
            if isinstance(loaded, dict):
                parsed = loaded
        except Exception as exc:  # pragma: no cover - diagnostics path
            parse_error = str(exc)
    return {
        "rc": int(cp.returncode),
        "stdout_path": str(out_path),
        "stderr_tail": str(cp.stderr or "").strip()[-400:],
        "stdout_parse_error": parse_error,
        "json": parsed,
    }


root = pathlib.Path(os.environ["ROUTING_PREFLIGHT_REFRESH_ROOT"]).resolve()
json_mode = str(os.environ.get("ROUTING_PREFLIGHT_REFRESH_JSON_MODE", "0")).strip() == "1"
latest = root / "state" / "continuity" / "latest"
latest.mkdir(parents=True, exist_ok=True)

run_id = "routing-preflight-refresh-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

transport_request_path = latest / "routing_preflight_refresh_transport_request.json"
transport_decision_path = latest / "routing_preflight_refresh_transport_decision.json"
model_candidate_path = latest / "routing_preflight_refresh_model_rollout_candidate.json"
model_gate_decision_path = latest / "routing_preflight_refresh_model_rollout_gate_decision.json"
route_request_path = latest / "routing_preflight_refresh_route_request.json"
route_decision_path = latest / "routing_preflight_refresh_route_decision.json"
route_decision_operator_surface_check_path = latest / "routing_preflight_refresh_route_decision_operator_surface_check.json"
replay_evidence_path = latest / "routing_preflight_refresh_replay_evidence.json"
summary_path = latest / "routing_preflight_refresh_latest.json"

summary: Dict[str, Any] = {
    "schema": "clawd.routing_preflight_refresh.v1",
    "generated_at": iso_now(),
    "run_id": run_id,
    "status": "blocked",
    "failure_reason": None,
    "steps": {},
    "artifacts": {
        "transport_request": str(transport_request_path),
        "transport_decision": str(transport_decision_path),
        "model_rollout_candidate": str(model_candidate_path),
        "model_rollout_gate_decision": str(model_gate_decision_path),
        "route_request": str(route_request_path),
        "route_decision": str(route_decision_path),
        "route_decision_operator_surface_check": str(route_decision_operator_surface_check_path),
        "replay_evidence": str(replay_evidence_path),
    },
    "route": {},
}

try:
    transport_request = {
        "channel": "telegram",
        "chat": {
            "scope": "group",
            "id": "-1001234567890",
            "is_forum": False,
        },
        "event": {
            "kind": "message",
            "text": "routing preflight freshness probe",
        },
    }
    transport_request_path.write_text(json.dumps(transport_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    candidate_template_path = root / "docs" / "ops" / "templates" / "model_rollout_candidate.template.json"
    source_hash_path = root / "docs" / "ops" / "model_routing_no_llm_matrix_v1.md"
    schema_path = root / "docs" / "ops" / "schemas" / "model_rollout_candidate.schema.json"

    candidate = json.loads(candidate_template_path.read_text(encoding="utf-8"))
    now_iso = iso_now()
    if isinstance(candidate, dict):
        candidate["evaluated_at"] = now_iso
        if isinstance(candidate.get("scorecard"), dict):
            candidate["scorecard"]["scored_at"] = now_iso
            if isinstance((candidate["scorecard"].get("cost")), dict):
                candidate["scorecard"]["cost"]["provider_evidence_updated_at"] = now_iso
        if isinstance(candidate.get("source_refs"), list):
            source_hash = hashlib.sha256(source_hash_path.read_bytes()).hexdigest()
            for row in candidate["source_refs"]:
                if not isinstance(row, dict):
                    continue
                ref_path = str(row.get("path") or "").strip()
                if ref_path == "docs/ops/model_routing_no_llm_matrix_v1.md":
                    row["content_hash"] = f"sha256:{source_hash}"
    model_candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    replay_step = run_cmd(
        root,
        ["bash", "ops/openclaw/continuity.sh", "failover-replay-evidence", "--json"],
        replay_evidence_path,
        timeout=900,
    )
    summary["steps"]["failover_replay_evidence"] = {
        "rc": replay_step.get("rc"),
        "stderr_tail": replay_step.get("stderr_tail"),
        "json_parse_error": replay_step.get("stdout_parse_error"),
        "artifact": str(replay_evidence_path),
    }
    if replay_step.get("rc") != 0 or not isinstance(replay_step.get("json"), dict):
        summary["failure_reason"] = "failover_replay_evidence_refresh_failed"
        raise RuntimeError(summary["failure_reason"])

    gate_step = run_cmd(
        root,
        [
            "bash",
            "ops/openclaw/continuity.sh",
            "model-rollout-gate",
            "--packet",
            str(model_candidate_path),
            "--schema-path",
            str(schema_path),
            "--json",
        ],
        model_gate_decision_path,
    )
    gate_json = gate_step.get("json") if isinstance(gate_step.get("json"), dict) else {}
    summary["steps"]["model_rollout_gate"] = {
        "rc": gate_step.get("rc"),
        "stderr_tail": gate_step.get("stderr_tail"),
        "json_parse_error": gate_step.get("stdout_parse_error"),
        "decision": gate_json.get("decision"),
        "block_reason": gate_json.get("block_reason"),
        "artifact": str(model_gate_decision_path),
    }
    if gate_step.get("rc") != 0 or str(gate_json.get("decision") or "").upper() != "PASS":
        summary["failure_reason"] = "model_rollout_gate_not_pass"
        raise RuntimeError(summary["failure_reason"])

    transport_step = run_cmd(
        root,
        [
            "bash",
            "ops/openclaw/continuity.sh",
            "session-transport-route",
            "--topology",
            "docs/ops/templates/session_topology_transport_contract.template.json",
            "--request",
            str(transport_request_path),
            "--json",
        ],
        transport_decision_path,
    )
    transport_json = transport_step.get("json") if isinstance(transport_step.get("json"), dict) else {}
    summary["steps"]["session_transport_route"] = {
        "rc": transport_step.get("rc"),
        "stderr_tail": transport_step.get("stderr_tail"),
        "json_parse_error": transport_step.get("stdout_parse_error"),
        "decision": transport_json.get("decision"),
        "block_reason": transport_json.get("block_reason"),
        "artifact": str(transport_decision_path),
    }
    if transport_step.get("rc") != 0 or str(transport_json.get("decision") or "").upper() != "PASS":
        summary["failure_reason"] = "session_transport_route_not_pass"
        raise RuntimeError(summary["failure_reason"])

    route_binding = ((transport_json.get("route") or {}).get("routing_basis") or {})
    lane = ((transport_json.get("route") or {}).get("lane") or {})
    session = ((transport_json.get("route") or {}).get("session") or {})
    transport_info = ((transport_json.get("route") or {}).get("transport") or {})
    transport_key = str(route_binding.get("transport_key") or "").strip()
    lane_name = str(lane.get("name") or "").strip()
    agent_id = str(lane.get("agent_id") or "").strip()
    session_key = str(session.get("session_key") or "").strip()
    if not transport_key or not lane_name or not agent_id or not session_key:
        summary["failure_reason"] = "transport_binding_incomplete"
        raise RuntimeError(summary["failure_reason"])

    route_request = {
        "session_kind": "worker_slice",
        "task_class": "research",
        "risk_tier": "low",
        "scope_shape": "single_surface",
        "verification_class": "self_check",
        "worker_topology": "single",
        "fold_in_target": "queue_continuity",
        "worker_lane": lane_name,
        "support_only": True,
        "invocation_prompt": "Produce a bounded continuity routing preflight summary.",
        "knowledge_retrieval": {
            "enabled": False,
            "required": False,
        },
        "event_backbone": {
            "correlation_id": run_id,
            "idempotency_key": run_id,
            "sequence": 1,
            "parent_event_id": None,
        },
        "workflow_state_machine": {
            "workflow_id": run_id,
            "expected_current_state": "INIT",
        },
        "transport_route": {
            "transport_key": transport_key,
            "lane_name": lane_name,
            "agent_id": agent_id,
            "session_key": session_key,
            "message_thread_id": transport_info.get("message_thread_id"),
        },
    }
    route_request_path.write_text(json.dumps(route_request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    route_step = run_cmd(
        root,
        [
            "bash",
            "ops/openclaw/continuity.sh",
            "session-route",
            "--legacy-allow-missing-worker-allocation-contract",
            "--topology",
            "docs/ops/templates/session_topology_contract.template.json",
            "--request",
            str(route_request_path),
            "--qualification-decision",
            str(model_gate_decision_path),
            "--transport-decision",
            str(transport_decision_path),
            "--json",
        ],
        route_decision_path,
    )
    route_json = route_step.get("json") if isinstance(route_step.get("json"), dict) else {}
    route_body = route_json.get("route") if isinstance(route_json.get("route"), dict) else {}
    summary["steps"]["session_route"] = {
        "rc": route_step.get("rc"),
        "stderr_tail": route_step.get("stderr_tail"),
        "json_parse_error": route_step.get("stdout_parse_error"),
        "decision": route_json.get("decision"),
        "block_reason": route_json.get("block_reason"),
        "artifact": str(route_decision_path),
    }
    summary["route"] = {
        "decision": route_json.get("decision"),
        "route_class": route_body.get("route_class"),
        "required_rollout_stage": route_body.get("required_rollout_stage"),
        "selected_model": route_body.get("selected_model"),
        "selected_rule_id": route_body.get("selected_rule_id"),
        "evaluated_at": route_json.get("evaluated_at"),
    }
    if route_step.get("rc") != 0 or str(route_json.get("decision") or "").upper() != "PASS":
        summary["failure_reason"] = "session_route_not_pass"
        raise RuntimeError(summary["failure_reason"])

    route_surface_check_step = run_cmd(
        root,
        [
            "python3",
            "ops/openclaw/continuity/check_routing_preflight_route_decision_operator_surface.py",
            "--json",
            "--route-decision",
            str(route_decision_path),
        ],
        route_decision_operator_surface_check_path,
    )
    route_surface_check_json = (
        route_surface_check_step.get("json") if isinstance(route_surface_check_step.get("json"), dict) else {}
    )
    summary["steps"]["route_decision_operator_surface_check"] = {
        "rc": route_surface_check_step.get("rc"),
        "stderr_tail": route_surface_check_step.get("stderr_tail"),
        "json_parse_error": route_surface_check_step.get("stdout_parse_error"),
        "ok": route_surface_check_json.get("ok"),
        "issue_count": route_surface_check_json.get("issue_count"),
        "artifact": str(route_decision_operator_surface_check_path),
    }
    if route_surface_check_step.get("rc") != 0 or route_surface_check_json.get("ok") is not True:
        summary["failure_reason"] = "route_decision_operator_surface_invalid"
        raise RuntimeError(summary["failure_reason"])

    summary["status"] = "pass"
except Exception:
    if not summary.get("failure_reason"):
        summary["failure_reason"] = "routing_preflight_refresh_failed"

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if json_mode:
    print(json.dumps(summary, ensure_ascii=False, indent=2))
else:
    if summary.get("status") == "pass":
        route = summary.get("route") if isinstance(summary.get("route"), dict) else {}
        print(
            "PASS routing_preflight_refresh "
            f"route_class={route.get('route_class') or 'unknown'} "
            f"selected_model={route.get('selected_model') or 'unknown'}"
        )
    else:
        print(f"BLOCK routing_preflight_refresh reason={summary.get('failure_reason') or 'unknown'}")

raise SystemExit(0 if summary.get("status") == "pass" else 2)
PY
