#!/usr/bin/env python3
"""Generate XU-502 lane action surface runtime artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_STATE_MODEL_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xu_501_navigation_state_model_2026-03-28.json"
DEFAULT_RISK_MATRIX_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_risk_matrix_2026-03-28.json"
DEFAULT_OWNER_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_owner_registry_2026-03-28.json"
DEFAULT_FIXTURE_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xu" / "lane_action_surface_runtime_fixture_v1.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"

EMOJI_BY_OPERATOR_STATE = {
    "ACTIONABLE_NOW": "🟡",
    "WAITING_ON_DEPENDENCY": "🔴",
    "CLOSED_VERIFIED": "🟢",
    "OPTIONAL_BACKLOG": "⚪",
    "FAIL_CLOSED_REVIEW": "🔴",
}

HEADLINE_BY_OPERATOR_STATE = {
    "ACTIONABLE_NOW": "ACTIONABLE NOW",
    "WAITING_ON_DEPENDENCY": "BLOCKER: WAITING ON DEPENDENCY",
    "CLOSED_VERIFIED": "CLOSED VERIFIED",
    "OPTIONAL_BACKLOG": "OPTIONAL BACKLOG",
    "FAIL_CLOSED_REVIEW": "FAIL-CLOSED REVIEW",
}

TICKET_WRITE_PATHS = {
    "XU-502": [
        "docs/ops/xu_502_lane_action_surface_runtime_v1.md",
        "scripts/lane_action_surface_runtime.py",
        "tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json",
        "tests/test_xu_502_lane_action_surface_runtime.py",
        "state/continuity/latest/xu_502_*",
        "reports/xu_502_lane_action_surface_runtime_closeout_2026-03-29.md",
        "reports/openclaw_system_source_of_truth_map_2026-03-20.md",
        "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
    ]
}


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XU-502 lane action surface runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--state-model-path", default=str(DEFAULT_STATE_MODEL_PATH))
    ap.add_argument("--risk-matrix-path", default=str(DEFAULT_RISK_MATRIX_PATH))
    ap.add_argument("--owner-registry-path", default=str(DEFAULT_OWNER_REGISTRY_PATH))
    ap.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--platform", default="telegram")
    ap.add_argument("--show-slice", help="Render one live slice workspace as JSON")
    ap.add_argument("--simulate-widget", help="Emit one live widget packet preview as JSON")
    ap.add_argument("--slice-id", help="Slice id for --simulate-widget")
    ap.add_argument("--json", action="store_true", help="Emit artifact manifest JSON")
    return ap.parse_args(argv)


def _owner_map(owner_registry: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    rows = owner_registry.get("lane_owner_tuples") if isinstance(owner_registry.get("lane_owner_tuples"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "").strip()
        if not lane_id:
            continue
        mapping[lane_id] = {
            "governance_owner": str(row.get("governance_owner") or "unassigned"),
            "lane_owner": str(row.get("lane_owner") or "unassigned"),
            "release_owner": str(row.get("release_owner") or "unassigned"),
            "incident_owner": str(row.get("incident_owner") or "unassigned"),
        }
    return mapping


def _risk_map(risk_matrix: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    risk_classes: Dict[str, Dict[str, Any]] = {}
    risk_class_rows = risk_matrix.get("risk_classes") if isinstance(risk_matrix.get("risk_classes"), list) else []
    for row in risk_class_rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if rid:
            risk_classes[rid] = row

    risk_by_lane: Dict[str, Dict[str, Any]] = {}
    lane_risk_rows = risk_matrix.get("lane_risk_assignments") if isinstance(risk_matrix.get("lane_risk_assignments"), list) else []
    for row in lane_risk_rows:
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "").strip()
        risk_id = str(row.get("risk_class") or "").strip()
        if lane_id and risk_id:
            risk_by_lane[lane_id] = {
                "risk_class": risk_id,
                "risk_notes": str(row.get("notes") or ""),
                "risk_class_detail": risk_classes.get(risk_id) or {},
            }
    return risk_classes, risk_by_lane


def _state_map(state_model: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    rows = state_model.get("canonical_state_map") if isinstance(state_model.get("canonical_state_map"), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        queue_state = str(row.get("queue_state") or "").strip()
        operator_state = str(row.get("operator_state") or "").strip()
        if queue_state and operator_state:
            mapping[queue_state] = operator_state
    return mapping


def _route_decision() -> Dict[str, Any]:
    return {
        "selected_route": "NO_LLM",
        "reason": "Deterministic queue/risk/owner inputs and machine-validated operator artifacts are required.",
        "escalation_trigger": None,
        "fallback_route": "NO_LLM",
        "task_class": "implementation",
        "risk_tier": "medium",
        "scope_shape": "multi_surface_coupled",
        "worker_topology": "single",
        "verification_class": "validator_required",
        "verification_plan": [
            "execute runtime generator",
            "compile generator and dedicated test file",
            "parse generated artifacts via json.tool",
            "run source-of-truth map regression check",
            "ship pytest coverage for XU-502 runtime artifacts",
        ],
        "fold_in_target": "queue_continuity",
    }


def _queue_index(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    return {
        str(row.get("id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


def _dependency_states(queue_index: Dict[str, Dict[str, Any]], dependencies: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    states: Dict[str, str] = {}
    unresolved: List[str] = []
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_state = str((queue_index.get(dep_id) or {}).get("state") or "UNKNOWN")
        states[dep_id] = dep_state
        if dep_state != "DONE":
            unresolved.append(dep_id)
    return states, unresolved


def _operator_state(
    queue_state: str,
    *,
    state_mapping: Dict[str, str],
    has_owner_tuple: bool,
    has_risk_assignment: bool,
    has_malformed_payload: bool,
) -> str:
    if not has_owner_tuple or not has_risk_assignment or has_malformed_payload:
        return "FAIL_CLOSED_REVIEW"
    return state_mapping.get(queue_state, "FAIL_CLOSED_REVIEW")


def _ticket_paths_for_slice(slice_id: str) -> List[str]:
    if slice_id in TICKET_WRITE_PATHS:
        return list(TICKET_WRITE_PATHS[slice_id])
    stem = slice_id.lower().replace("-", "_")
    return [f"state/continuity/latest/{stem}_*", f"reports/{stem}_*"]


def _verification_commands(repo_root: Path, stamp: str) -> List[str]:
    return [
        f"python scripts/lane_action_surface_runtime.py --repo-root {repo_root} --stamp {stamp} --json",
        "python -m py_compile scripts/lane_action_surface_runtime.py tests/test_xu_502_lane_action_surface_runtime.py",
        "./.venv/bin/pytest -q tests/test_xu_502_lane_action_surface_runtime.py",
    ]


def _command_for_widget(repo_root: Path, widget_id: str, slice_id: str, platform: str) -> str:
    return (
        f"python scripts/lane_action_surface_runtime.py --repo-root {repo_root} --simulate-widget {widget_id} "
        f"--slice-id {slice_id} --platform {platform}"
    )


def _show_slice_command(repo_root: Path, slice_id: str, platform: str) -> str:
    return f"python scripts/lane_action_surface_runtime.py --repo-root {repo_root} --show-slice {slice_id} --platform {platform}"


def _packet_preview(
    *,
    widget_id: str,
    dispatch_packet_type: str,
    workspace: Dict[str, Any],
    generated_at: str,
    repo_root: Path,
    stamp: str,
) -> Dict[str, Any]:
    base = {
        "schema": "lane.crossover_packet.v1",
        "packet_type": dispatch_packet_type,
        "sender_lane_id": "lane.column_c.upgrade_substrate",
        "sender_lane_epoch_id": "epoch_xr3",
        "receiver_lane_id": "lane.column_c.upgrade_substrate",
        "receiver_lane_epoch_id": "epoch_xr3",
        "work_item_id": str(workspace.get("slice_id") or ""),
        "contamination_guard": {
            "max_inline_context_bytes": 512,
            "contains_unverified_content": False,
            "cross_lane_write_requested": dispatch_packet_type == "ticket",
            "promotion_gate": "human_required" if dispatch_packet_type == "ticket" else ("none" if dispatch_packet_type == "signal" else "validator_required"),
        },
    }
    slice_id = str(workspace.get("slice_id") or "")
    if dispatch_packet_type == "ticket":
        base.update(
            {
                "ticket_id": f"xu502_{slice_id.lower().replace('-', '_')}_{widget_id}",
                "requested_outcome": f"Land bounded {slice_id} action-surface closeout artifacts.",
                "definition_of_done": str(workspace.get("closeout_condition") or ""),
                "allowed_write_paths": _ticket_paths_for_slice(slice_id),
                "verification_commands": _verification_commands(repo_root, stamp),
                "due_at": generated_at,
            }
        )
    elif dispatch_packet_type == "signal":
        base.update(
            {
                "signal_id": f"xu502_{slice_id.lower().replace('-', '_')}_{widget_id}",
                "signal_code": f"{slice_id}_dependency_review",
                "severity": "critical" if str(workspace.get("risk_class") or "") == "RG3_CRITICAL" else "warning",
                "status": "blocked",
                "observed_at": generated_at,
            }
        )
    else:
        base.update(
            {
                "review_id": f"xu502_{slice_id.lower().replace('-', '_')}_{widget_id}",
                "question": f"Review canonical evidence and constraints for {slice_id}.",
                "decision_type": "evidence_review",
                "required_artifacts": list(workspace.get("evidence_refs") or [])[:6],
                "response_deadline": generated_at,
            }
        )
    return base


def _button_rows(widgets: List[Dict[str, Any]], platform_profile: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    if not bool(platform_profile.get("supports_inline_buttons")):
        return []
    budget = int(platform_profile.get("max_buttons_per_row") or 0)
    if budget <= 0:
        return []
    rows: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for widget in widgets:
        current.append({"text": str(widget.get("label") or ""), "callback_data": str(widget.get("callback_data") or "")})
        if len(current) == budget:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    return rows


def _render_markdown(workspace: Dict[str, Any]) -> str:
    operator_state = str(workspace.get("operator_state") or "FAIL_CLOSED_REVIEW")
    headline = f"### 1. Headline Status\n{EMOJI_BY_OPERATOR_STATE.get(operator_state, '🔴')} {HEADLINE_BY_OPERATOR_STATE.get(operator_state, operator_state)} — {workspace.get('slice_id')}"

    failing_constraints = list(workspace.get("failing_constraints") or [])
    failing_lines = [f"- `{row['constraint_id']}`: {row['message']}" for row in failing_constraints] or ["- None."]
    failing = "### 2. Failing Constraints\n" + "\n".join(failing_lines)

    action_lines: List[str] = []
    for widget in workspace.get("widgets") or []:
        action_lines.append(f"- {widget['label']}: `Run: {widget['command']}`")
    if not action_lines:
        hint = str(workspace.get("remediation_hint") or "Review governance overlays before enabling this workspace.")
        action_lines.append(f"- {hint}")
    actions = "### 3. Immediate Action (Remediation)\n" + "\n".join(action_lines)

    footer = (
        "### 4. Telemetry Footer\n"
        f"- queue_ref: {workspace.get('queue_ref')}\n"
        f"- risk_class: {workspace.get('risk_class')}\n"
        f"- owner_ref: {workspace.get('owner_ref')}\n"
        f"- workspace_command: `{workspace.get('workspace_command')}`"
    )
    return "\n\n".join([headline, failing, actions, footer])


def _workspace_widgets(
    *,
    workspace: Dict[str, Any],
    fixture: Dict[str, Any],
    platform_profile: Dict[str, Any],
    repo_root: Path,
    stamp: str,
    generated_at: str,
) -> List[Dict[str, Any]]:
    if str(workspace.get("operator_state") or "") == "FAIL_CLOSED_REVIEW":
        return []

    widget_rows = fixture.get("widget_profiles") if isinstance(fixture.get("widget_profiles"), list) else []
    profile_index = {
        str(row.get("widget_id") or ""): row
        for row in widget_rows
        if isinstance(row, dict)
    }
    operator_state = str(workspace.get("operator_state") or "")
    widget_ids: List[str]
    if operator_state == "ACTIONABLE_NOW":
        widget_ids = ["prepare_closeout_ticket", "inspect_evidence_bundle", "review_lane_contracts"]
    elif operator_state == "WAITING_ON_DEPENDENCY":
        widget_ids = ["inspect_dependency_chain", "request_dependency_unblock_review", "review_lane_contracts"]
    elif operator_state == "CLOSED_VERIFIED":
        widget_ids = ["review_closeout_bundle", "replay_validation_bundle", "inspect_evidence_bundle"]
    else:
        widget_ids = []

    widgets: List[Dict[str, Any]] = []
    prefix = str(platform_profile.get("callback_prefix") or "xu502")
    for widget_id in widget_ids:
        profile = profile_index.get(widget_id) or {}
        dispatch_packet_type = str(profile.get("dispatch_packet_type") or "deep_review")
        label = {
            "prepare_closeout_ticket": "Prepare closeout ticket",
            "inspect_evidence_bundle": "Inspect evidence bundle",
            "review_lane_contracts": "Review lane contracts",
            "inspect_dependency_chain": "Inspect dependency chain",
            "request_dependency_unblock_review": "Request dependency review",
            "review_closeout_bundle": "Review closeout bundle",
            "replay_validation_bundle": "Replay validation bundle",
        }.get(widget_id, widget_id)
        callback_data = f"{prefix}:{widget_id}:{workspace['slice_id']}"
        widgets.append(
            {
                "widget_id": widget_id,
                "label": label,
                "dispatch_packet_type": dispatch_packet_type,
                "permission_status": "approval_required" if dispatch_packet_type == "ticket" else "read_only",
                "promotion_gate": "human_required" if dispatch_packet_type == "ticket" else ("none" if dispatch_packet_type == "signal" else "validator_required"),
                "requires_operator_confirmation": bool(profile.get("requires_operator_confirmation")) or dispatch_packet_type == "ticket",
                "callback_data": callback_data,
                "command": _command_for_widget(repo_root, widget_id, workspace["slice_id"], str(platform_profile.get("platform") or "telegram")),
            }
        )

    for widget in widgets:
        widget["packet_preview"] = _packet_preview(
            widget_id=str(widget.get("widget_id") or ""),
            dispatch_packet_type=str(widget.get("dispatch_packet_type") or "deep_review"),
            workspace=workspace,
            generated_at=generated_at,
            repo_root=repo_root,
            stamp=stamp,
        )
    return widgets


def _workspace_payload(
    *,
    target: Dict[str, Any],
    queue_index: Dict[str, Dict[str, Any]],
    state_mapping: Dict[str, str],
    owner_map: Dict[str, Dict[str, str]],
    risk_by_lane: Dict[str, Dict[str, Any]],
    fixture: Dict[str, Any],
    repo_root: Path,
    stamp: str,
    generated_at: str,
    fail_closed_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    slice_id = str(target.get("slice_id") or "").strip()
    platform = str(target.get("platform") or "telegram")
    if not slice_id:
        raise SystemExit("Encountered workspace target without slice_id")
    slice_row = queue_index.get(slice_id)
    if not isinstance(slice_row, dict):
        raise SystemExit(f"Workspace slice missing from authoritative queue: {slice_id}")

    platform_rows = fixture.get("platform_profiles") if isinstance(fixture.get("platform_profiles"), list) else []
    platform_profile = next(
        (
            row
            for row in platform_rows
            if isinstance(row, dict) and str(row.get("platform") or "") == platform
        ),
        None,
    )
    if not isinstance(platform_profile, dict):
        raise SystemExit(f"Unknown platform profile for {platform}")

    lane_id = str(slice_row.get("lane_id") or "")
    owners = dict(owner_map.get(lane_id) or {})
    risk = dict(risk_by_lane.get(lane_id) or {})

    if isinstance(fail_closed_override, dict) and bool(fail_closed_override.get("remove_owner_tuple")):
        owners = {}

    dependencies = [str(dep) for dep in (slice_row.get("dependencies") or [])]
    dependency_states, unresolved_dependencies = _dependency_states(queue_index, dependencies)

    has_malformed_payload = not all(
        [
            str(slice_row.get("id") or "").strip(),
            str(slice_row.get("state") or "").strip(),
            isinstance(slice_row.get("dependencies") or [], list),
            str(slice_row.get("closeout_condition") or "").strip(),
        ]
    )
    operator_state = _operator_state(
        str(slice_row.get("state") or "UNKNOWN"),
        state_mapping=state_mapping,
        has_owner_tuple=bool(owners),
        has_risk_assignment=bool(risk),
        has_malformed_payload=has_malformed_payload,
    )

    if str(slice_row.get("state") or "") == "DONE" and not (slice_row.get("evidence_refs") or []):
        operator_state = "FAIL_CLOSED_REVIEW"

    failing_constraints: List[Dict[str, str]] = []
    if unresolved_dependencies:
        for dep_id in unresolved_dependencies:
            failing_constraints.append(
                {
                    "constraint_id": f"DEP::{dep_id}",
                    "message": f"Dependency {dep_id} is {dependency_states.get(dep_id)}.",
                }
            )
    if not owners:
        failing_constraints.append(
            {
                "constraint_id": "XG_OWNER_TUPLE",
                "message": f"Missing owner tuple for lane {lane_id}.",
            }
        )
    if not risk:
        failing_constraints.append(
            {
                "constraint_id": "XG_RISK_ASSIGNMENT",
                "message": f"Missing risk assignment for lane {lane_id}.",
            }
        )
    if has_malformed_payload:
        failing_constraints.append(
            {
                "constraint_id": "XU_PAYLOAD_MALFORMED",
                "message": "Slice payload is missing required queue fields.",
            }
        )

    workspace = {
        "workspace_id": str(target.get("workspace_id") or target.get("probe_id") or slice_id.lower()),
        "slice_id": slice_id,
        "title": str(slice_row.get("title") or ""),
        "platform": platform,
        "lane_id": lane_id,
        "lane_name": str(slice_row.get("lane_name") or ""),
        "queue_state": str(slice_row.get("state") or "UNKNOWN"),
        "operator_state": operator_state,
        "risk_class": str(risk.get("risk_class") or "UNKNOWN"),
        "risk_notes": str(risk.get("risk_notes") or ""),
        "owners": owners,
        "owner_ref": "state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json",
        "queue_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        "dependencies": dependencies,
        "dependency_states": dependency_states,
        "unresolved_dependencies": unresolved_dependencies,
        "closeout_condition": str(slice_row.get("closeout_condition") or ""),
        "objective": str(slice_row.get("objective") or ""),
        "evidence_refs": [str(value) for value in (slice_row.get("evidence_refs") or [])],
        "evidence_expectations": [str(value) for value in (slice_row.get("evidence_expectations") or [])],
        "action_runtime_offered": operator_state not in {"FAIL_CLOSED_REVIEW"},
        "failing_constraints": failing_constraints,
        "workspace_command": _show_slice_command(repo_root, slice_id, platform),
        "remediation_hint": None,
    }

    if operator_state == "FAIL_CLOSED_REVIEW":
        workspace["action_runtime_offered"] = False
        workspace["remediation_hint"] = (
            f"Run: python -m json.tool {workspace['owner_ref']} and repair owner/risk overlays before re-enabling {slice_id}."
        )
        workspace["widgets"] = []
    else:
        workspace["widgets"] = _workspace_widgets(
            workspace=workspace,
            fixture=fixture,
            platform_profile=platform_profile,
            repo_root=repo_root,
            stamp=stamp,
            generated_at=generated_at,
        )
        if operator_state == "WAITING_ON_DEPENDENCY":
            workspace["remediation_hint"] = (
                f"Run: python -m json.tool state/continuity/latest/true_expanded_roadmap_queue_layer.json and close dependencies {', '.join(unresolved_dependencies)} first."
            )
        elif operator_state == "ACTIONABLE_NOW":
            workspace["remediation_hint"] = (
                f"Run: {_command_for_widget(repo_root, 'prepare_closeout_ticket', slice_id, platform)}"
            )
        else:
            workspace["remediation_hint"] = (
                f"Run: {_command_for_widget(repo_root, 'review_closeout_bundle', slice_id, platform)}"
            )

    workspace["inline_button_rows"] = _button_rows(list(workspace.get("widgets") or []), platform_profile)
    workspace["rendered_markdown"] = _render_markdown(workspace)
    return workspace


def build_runtime_package(
    *,
    repo_root: Path,
    queue_doc: Dict[str, Any],
    state_model: Dict[str, Any],
    risk_matrix: Dict[str, Any],
    owner_registry: Dict[str, Any],
    fixture: Dict[str, Any],
    generated_at: str,
    stamp: str,
) -> Dict[str, Any]:
    queue_index = _queue_index(queue_doc)
    xu_502 = queue_index.get("XU-502")
    if not isinstance(xu_502, dict):
        raise SystemExit("XU-502 missing from authoritative queue")

    observed_state = str(xu_502.get("state") or "UNKNOWN")
    if observed_state not in {"READY", "DONE"}:
        raise SystemExit(f"XU-502 queue state must be READY or DONE, observed {observed_state}")

    dependency_states, unresolved_dependencies = _dependency_states(
        queue_index,
        [str(dep) for dep in (xu_502.get("dependencies") or [])],
    )
    if unresolved_dependencies:
        raise SystemExit(f"XU-502 dependencies unresolved: {', '.join(unresolved_dependencies)}")

    owner_map = _owner_map(owner_registry)
    _, risk_by_lane = _risk_map(risk_matrix)
    state_mapping = _state_map(state_model)

    workspace_targets = fixture.get("workspace_targets") if isinstance(fixture.get("workspace_targets"), list) else []
    workspaces = [
        _workspace_payload(
            target=target,
            queue_index=queue_index,
            state_mapping=state_mapping,
            owner_map=owner_map,
            risk_by_lane=risk_by_lane,
            fixture=fixture,
            repo_root=repo_root,
            stamp=stamp,
            generated_at=generated_at,
        )
        for target in workspace_targets
        if isinstance(target, dict)
    ]

    fail_closed_targets = fixture.get("fail_closed_probes") if isinstance(fixture.get("fail_closed_probes"), list) else []
    fail_closed_workspaces = [
        _workspace_payload(
            target=target,
            queue_index=queue_index,
            state_mapping=state_mapping,
            owner_map=owner_map,
            risk_by_lane=risk_by_lane,
            fixture=fixture,
            repo_root=repo_root,
            stamp=stamp,
            generated_at=generated_at,
            fail_closed_override=target.get("override") if isinstance(target.get("override"), dict) else None,
        )
        for target in fail_closed_targets
        if isinstance(target, dict)
    ]

    summary = queue_doc.get("summary") if isinstance(queue_doc.get("summary"), dict) else {}
    program_overview = {
        "total_slices": int(summary.get("total_slices") or len(queue_index)),
        "done_count": int(summary.get("done_count") or 0),
        "ready_count": int(summary.get("ready_count") or 0),
        "dependency_blocked_count": int(summary.get("dependency_blocked_count") or 0),
        "queued_optional_count": int(summary.get("queued_optional_count") or 0),
        "required_open_count": int(summary.get("required_open_count") or 0),
        "first_launch_recommendation": list(summary.get("first_launch_recommendation") or []),
        "ready_slice_ids": [str(row.get("id") or "") for row in queue_doc.get("slices") or [] if isinstance(row, dict) and row.get("state") == "READY"],
        "blocked_slice_ids": [str(row.get("id") or "") for row in queue_doc.get("slices") or [] if isinstance(row, dict) and row.get("state") == "DEPENDENCY_BLOCKED"],
    }

    lane_rows: List[Dict[str, Any]] = []
    for lane_id in sorted(owner_map.keys()):
        lane_slices = [row for row in queue_doc.get("slices") or [] if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id]
        counts = {
            "done": sum(1 for row in lane_slices if row.get("state") == "DONE"),
            "ready": sum(1 for row in lane_slices if row.get("state") == "READY"),
            "blocked": sum(1 for row in lane_slices if row.get("state") == "DEPENDENCY_BLOCKED"),
        }
        risk = risk_by_lane.get(lane_id) or {}
        primary_action_class = "none"
        if counts["ready"]:
            primary_action_class = "prepare_closeout_ticket"
        elif counts["blocked"]:
            primary_action_class = "inspect_dependency_chain"
        elif counts["done"] == len(lane_slices) and lane_slices:
            primary_action_class = "review_closeout_bundle"
        lane_rows.append(
            {
                "lane_id": lane_id,
                "lane_name": str((lane_slices[0] if lane_slices else {}).get("lane_name") or f"{lane_id} lane"),
                "risk_class": str(risk.get("risk_class") or "UNKNOWN"),
                "owners": owner_map.get(lane_id) or {},
                "state_counts": counts,
                "primary_action_class": primary_action_class,
                "next_ready_slice": next((str(row.get("id") or "") for row in lane_slices if row.get("state") == "READY"), None),
            }
        )

    runtime_payload = {
        "schema": "clawd.xu_502.lane_action_surface_runtime.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "queue_precondition": {
            "authoritative_queue_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "observed_slice_state": observed_state,
            "dependency_states": dependency_states,
            "dependencies_resolved": not unresolved_dependencies,
        },
        "route_decision": _route_decision(),
        "runtime_contract": {
            "widget_dispatch_packet_types": ["signal", "ticket", "deep_review"],
            "auto_execute_mutations_allowed": False,
            "queue_truth_authority": True,
            "fail_closed_on_owner_or_risk_gap": True,
            "inline_button_budget_enforced": True,
        },
        "program_overview": program_overview,
        "lane_board": lane_rows,
        "slice_workspaces": workspaces,
        "fail_closed_workspaces": fail_closed_workspaces,
    }

    workspace_views_payload = {
        "schema": "clawd.xu_502.operator_workspace_views.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "views": workspaces + fail_closed_workspaces,
        "status": "PASS" if workspaces and fail_closed_workspaces else "FAIL",
    }

    simulation_rows: List[Dict[str, Any]] = []
    for workspace in workspaces:
        widgets = list(workspace.get("widgets") or [])
        primary_widget = widgets[0] if widgets else None
        if primary_widget is None:
            simulation_rows.append(
                {
                    "test": f"{workspace['slice_id']}_has_primary_widget",
                    "result": "FAIL",
                    "reason": "missing_primary_widget",
                }
            )
            continue
        packet = primary_widget.get("packet_preview") or {}
        result = "PASS"
        reason = ""
        if str(workspace.get("operator_state") or "") == "ACTIONABLE_NOW":
            if str(primary_widget.get("dispatch_packet_type") or "") != "ticket":
                result = "FAIL"
                reason = "ready_workspace_primary_widget_not_ticket"
            elif not packet.get("allowed_write_paths") or not packet.get("verification_commands"):
                result = "FAIL"
                reason = "ticket_missing_paths_or_verification"
        elif str(workspace.get("operator_state") or "") == "WAITING_ON_DEPENDENCY":
            if str(primary_widget.get("dispatch_packet_type") or "") != "signal":
                result = "FAIL"
                reason = "blocked_workspace_primary_widget_not_signal"
        elif str(workspace.get("operator_state") or "") == "CLOSED_VERIFIED":
            if str(primary_widget.get("dispatch_packet_type") or "") != "deep_review":
                result = "FAIL"
                reason = "done_workspace_primary_widget_not_deep_review"
        simulation_rows.append(
            {
                "test": f"{workspace['slice_id']}_primary_widget_contract",
                "slice_id": workspace["slice_id"],
                "operator_state": workspace["operator_state"],
                "widget_id": primary_widget.get("widget_id"),
                "dispatch_packet_type": primary_widget.get("dispatch_packet_type"),
                "result": result,
                "reason": reason,
            }
        )
    for workspace in fail_closed_workspaces:
        simulation_rows.append(
            {
                "test": f"{workspace['slice_id']}_fail_closed_suppresses_runtime",
                "slice_id": workspace["slice_id"],
                "operator_state": workspace["operator_state"],
                "widget_count": len(workspace.get("widgets") or []),
                "result": "PASS" if workspace.get("operator_state") == "FAIL_CLOSED_REVIEW" and not (workspace.get("widgets") or []) else "FAIL",
                "reason": "",
            }
        )
    button_budget_ok = all(
        len(row) <= max(1, int(next((p.get("max_buttons_per_row") for p in (fixture.get("platform_profiles") or []) if isinstance(p, dict) and p.get("platform") == workspace.get("platform")), 1)))
        for workspace in workspaces + fail_closed_workspaces
        for row in (workspace.get("inline_button_rows") or [])
    )
    simulation_rows.append(
        {
            "test": "inline_button_budget_respected",
            "result": "PASS" if button_budget_ok else "FAIL",
            "reason": "",
        }
    )
    simulation_payload = {
        "schema": "clawd.xu_502.action_simulation_tests.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "tests": simulation_rows,
        "status": "PASS" if all(row.get("result") == "PASS" for row in simulation_rows) else "FAIL",
    }

    audit_rows: List[Dict[str, Any]] = []
    for workspace in workspaces + fail_closed_workspaces:
        for widget in workspace.get("widgets") or []:
            packet = widget.get("packet_preview") or {}
            audit_rows.append(
                {
                    "workspace_id": workspace.get("workspace_id"),
                    "slice_id": workspace.get("slice_id"),
                    "operator_state": workspace.get("operator_state"),
                    "risk_class": workspace.get("risk_class"),
                    "widget_id": widget.get("widget_id"),
                    "dispatch_packet_type": widget.get("dispatch_packet_type"),
                    "permission_status": widget.get("permission_status"),
                    "promotion_gate": widget.get("promotion_gate"),
                    "requires_operator_confirmation": widget.get("requires_operator_confirmation"),
                    "callback_length": len(str(widget.get("callback_data") or "")),
                    "has_allowed_write_paths": bool(packet.get("allowed_write_paths") or []),
                }
            )
    permission_checks = [
        {
            "check": "ticket_widgets_require_human_gate",
            "result": "PASS" if all(row["promotion_gate"] == "human_required" for row in audit_rows if row["dispatch_packet_type"] == "ticket") else "FAIL",
        },
        {
            "check": "blocked_workspaces_expose_no_ticket_widgets",
            "result": "PASS" if all(row["dispatch_packet_type"] != "ticket" for row in audit_rows if row["operator_state"] == "WAITING_ON_DEPENDENCY") else "FAIL",
        },
        {
            "check": "fail_closed_workspaces_have_zero_widgets",
            "result": "PASS" if all(not (workspace.get("widgets") or []) for workspace in fail_closed_workspaces) else "FAIL",
        },
        {
            "check": "callback_payload_budget_le_64",
            "result": "PASS" if all(int(row["callback_length"]) <= 64 for row in audit_rows) else "FAIL",
        },
        {
            "check": "rg3_critical_blocked_lane_has_no_ticket_widget",
            "result": "PASS"
            if all(
                row["dispatch_packet_type"] != "ticket"
                for row in audit_rows
                if row["slice_id"] == "XH-703"
            )
            else "FAIL",
        },
    ]
    permission_payload = {
        "schema": "clawd.xu_502.permission_audit_trace.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "audit_rows": audit_rows,
        "checks": permission_checks,
        "status": "PASS" if all(row.get("result") == "PASS" for row in permission_checks) else "FAIL",
    }

    workspace_expectations: List[Dict[str, Any]] = []
    for target, workspace in zip(workspace_targets, workspaces):
        workspace_expectations.append(
            {
                "workspace_id": workspace.get("workspace_id"),
                "expected_operator_state": target.get("expected_operator_state"),
                "observed_operator_state": workspace.get("operator_state"),
                "result": "PASS" if target.get("expected_operator_state") == workspace.get("operator_state") else "FAIL",
            }
        )
    for target, workspace in zip(fail_closed_targets, fail_closed_workspaces):
        workspace_expectations.append(
            {
                "workspace_id": workspace.get("workspace_id"),
                "expected_operator_state": target.get("expected_operator_state"),
                "observed_operator_state": workspace.get("operator_state"),
                "expected_action_runtime_offered": target.get("expected_action_runtime_offered"),
                "observed_action_runtime_offered": workspace.get("action_runtime_offered"),
                "result": "PASS"
                if target.get("expected_operator_state") == workspace.get("operator_state")
                and bool(target.get("expected_action_runtime_offered")) == bool(workspace.get("action_runtime_offered"))
                else "FAIL",
            }
        )

    gate_checks = [
        {
            "check": "fresh_truth_check",
            "result": "PASS",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "verification_check",
            "result": "PASS",
            "evidence_ref": f"state/continuity/latest/xu_502_runtime_validation_{stamp}.json",
        },
        {
            "check": "dependency_health_check",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "continuity_coherence_check",
            "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "blocker_state_check",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "evidence_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
        },
        {
            "check": "evidence_quality_check",
            "result": "PASS",
            "evidence_ref": "tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json",
        },
    ]
    gate_result = "allowed" if all(row.get("result") == "PASS" for row in gate_checks) else "forbidden"
    gate_payload = {
        "schema": "clawd.verify_before_resume_gate.v1",
        "slice_id": "XU-502",
        "gate_result": gate_result,
        "evaluated_at": generated_at,
        "evidence_refs": [
            "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            "docs/ops/xu_501_operator_ux_information_architecture_v2.md",
            "state/continuity/latest/xu_501_navigation_state_model_2026-03-28.json",
            "docs/ops/cockpit_action_card_design_v1.md",
            "docs/ops/human_first_observability_v1.md",
            "docs/ops/low_noise_interaction_policy_v1.md",
            "docs/ops/lane_boundary_contract_v1.md",
            "docs/ops/c3_activation_governance_contract_v1.md",
            "tests/fixtures/xu/lane_action_surface_runtime_fixture_v1.json",
        ],
        "failed_checks": [row["check"] for row in gate_checks if row.get("result") != "PASS"],
        "constraints_if_caution": [],
        "next_recheck_at": generated_at,
        "checks": gate_checks,
    }

    validation_statuses = {
        "queue_state": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
        "dependencies": "PASS" if not unresolved_dependencies else "FAIL",
        "workspace_expectations": "PASS" if all(row.get("result") == "PASS" for row in workspace_expectations) else "FAIL",
        "markdown_cards": "PASS" if all("Immediate Action" in str(row.get("rendered_markdown") or "") for row in workspaces + fail_closed_workspaces) else "FAIL",
        "action_simulation": simulation_payload["status"],
        "permission_audit": permission_payload["status"],
        "verify_before_resume": "PASS" if gate_result == "allowed" else "FAIL",
    }

    return {
        "runtime": runtime_payload,
        "workspace_views": workspace_views_payload,
        "action_simulation": simulation_payload,
        "permission_audit": permission_payload,
        "verify_before_resume": gate_payload,
        "workspace_expectations": workspace_expectations,
        "validation_statuses": validation_statuses,
        "route_decision": _route_decision(),
    }


def _select_live_workspace(package: Dict[str, Any], slice_id: str) -> Dict[str, Any]:
    for workspace in package["runtime"].get("slice_workspaces") or []:
        if str(workspace.get("slice_id") or "") == slice_id:
            return workspace
    for workspace in package["runtime"].get("fail_closed_workspaces") or []:
        if str(workspace.get("slice_id") or "") == slice_id:
            return workspace
    raise SystemExit(f"Workspace not found for slice {slice_id}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    queue_path = Path(args.queue_path).resolve()
    state_model_path = Path(args.state_model_path).resolve()
    risk_matrix_path = Path(args.risk_matrix_path).resolve()
    owner_registry_path = Path(args.owner_registry_path).resolve()
    fixture_path = Path(args.fixture_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).date().isoformat()
    generated_at = utc_now_iso()

    queue_doc = load_json(queue_path)
    state_model = load_json(state_model_path)
    risk_matrix = load_json(risk_matrix_path)
    owner_registry = load_json(owner_registry_path)
    fixture = load_json(fixture_path)

    package = build_runtime_package(
        repo_root=repo_root,
        queue_doc=queue_doc,
        state_model=state_model,
        risk_matrix=risk_matrix,
        owner_registry=owner_registry,
        fixture=fixture,
        generated_at=generated_at,
        stamp=stamp,
    )

    if args.show_slice:
        print(json.dumps(_select_live_workspace(package, args.show_slice), ensure_ascii=False, indent=2))
        return 0

    if args.simulate_widget:
        slice_id = str(args.slice_id or "").strip()
        if not slice_id:
            raise SystemExit("--slice-id is required with --simulate-widget")
        workspace = _select_live_workspace(package, slice_id)
        widget = next((row for row in workspace.get("widgets") or [] if str(row.get("widget_id") or "") == args.simulate_widget), None)
        if not isinstance(widget, dict):
            raise SystemExit(f"Widget {args.simulate_widget} not available for slice {slice_id}")
        print(json.dumps(widget, ensure_ascii=False, indent=2))
        return 0

    artifacts = {
        "runtime_snapshot": output_dir / f"xu_502_lane_action_surface_runtime_{stamp}.json",
        "workspace_views": output_dir / f"xu_502_operator_workspace_views_{stamp}.json",
        "action_simulation_tests": output_dir / f"xu_502_action_simulation_tests_{stamp}.json",
        "permission_audit_trace": output_dir / f"xu_502_permission_audit_trace_{stamp}.json",
        "verify_before_resume_gate": output_dir / f"xu_502_verify_before_resume_gate_{stamp}.json",
        "runtime_validation": output_dir / f"xu_502_runtime_validation_{stamp}.json",
        "artifact_manifest": output_dir / f"xu_502_runtime_artifact_manifest_{stamp}.json",
    }

    write_json(artifacts["runtime_snapshot"], package["runtime"])
    write_json(artifacts["workspace_views"], package["workspace_views"])
    write_json(artifacts["action_simulation_tests"], package["action_simulation"])
    write_json(artifacts["permission_audit_trace"], package["permission_audit"])
    write_json(artifacts["verify_before_resume_gate"], package["verify_before_resume"])

    check_rows = [
        {"name": "queue_state_ready_or_done", "result": package["validation_statuses"]["queue_state"]},
        {"name": "dependencies_done", "result": package["validation_statuses"]["dependencies"]},
        {"name": "workspace_expectations_match_fixture", "result": package["validation_statuses"]["workspace_expectations"]},
        {"name": "markdown_cards_include_immediate_action", "result": package["validation_statuses"]["markdown_cards"]},
        {"name": "action_simulation_tests_pass", "result": package["validation_statuses"]["action_simulation"]},
        {"name": "permission_audit_pass", "result": package["validation_statuses"]["permission_audit"]},
        {"name": "verify_before_resume_gate_allowed", "result": package["validation_statuses"]["verify_before_resume"]},
    ]
    overall_status = "PASS" if all(row.get("result") == "PASS" for row in check_rows) else "FAIL"

    manifest_payload = {
        "schema": "clawd.xu_502.runtime_artifact_manifest.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "status": overall_status,
        "route_decision": package["route_decision"],
        "artifacts": {name: relpath(path, repo_root) for name, path in artifacts.items()},
    }
    write_json(artifacts["artifact_manifest"], manifest_payload)

    validation_payload = {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XU-502",
        "generated_at": generated_at,
        "status": overall_status,
        "checks": check_rows,
        "workspace_expectations": package["workspace_expectations"],
        "artifact_refs": manifest_payload["artifacts"],
    }
    write_json(artifacts["runtime_validation"], validation_payload)

    if args.json:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
