#!/usr/bin/env python3
"""Generate XU-503 explainability-panel runtime artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_STATE_MODEL_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xu_501_navigation_state_model_2026-03-28.json"
DEFAULT_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xb_402_domain_capability_registry_runtime_2026-03-29.json"
DEFAULT_RISK_MATRIX_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_risk_matrix_2026-03-28.json"
DEFAULT_OWNER_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_owner_registry_2026-03-28.json"
DEFAULT_FIXTURE_PATH = DEFAULT_REPO_ROOT / "tests" / "fixtures" / "xu" / "lane_explainability_panel_fixture_v1.json"
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
    "XU-503": [
        "docs/ops/xu_503_human_explainability_panel_for_domain_lanes_v1.md",
        "scripts/lane_explainability_panel_runtime.py",
        "tests/fixtures/xu/lane_explainability_panel_fixture_v1.json",
        "tests/test_xu_503_lane_explainability_panel_runtime.py",
        "state/continuity/latest/xu_503_*",
        "reports/xu_503_human_explainability_panel_for_domain_lanes_closeout_2026-03-29.md",
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
    ap = argparse.ArgumentParser(description="Generate XU-503 explainability-panel runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--state-model-path", default=str(DEFAULT_STATE_MODEL_PATH))
    ap.add_argument("--registry-path", default=str(DEFAULT_REGISTRY_PATH))
    ap.add_argument("--risk-matrix-path", default=str(DEFAULT_RISK_MATRIX_PATH))
    ap.add_argument("--owner-registry-path", default=str(DEFAULT_OWNER_REGISTRY_PATH))
    ap.add_argument("--fixture-path", default=str(DEFAULT_FIXTURE_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--platform", default="telegram")
    ap.add_argument("--show-slice", help="Render one explainability panel as JSON")
    ap.add_argument("--json", action="store_true", help="Emit artifact manifest JSON")
    return ap.parse_args(argv)


def _queue_index(queue_doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    return {
        str(row.get("id") or ""): row
        for row in rows
        if isinstance(row, dict) and str(row.get("id") or "").strip()
    }


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
    for row in risk_matrix.get("risk_classes") if isinstance(risk_matrix.get("risk_classes"), list) else []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        if rid:
            risk_classes[rid] = row

    by_lane: Dict[str, Dict[str, Any]] = {}
    for row in risk_matrix.get("lane_risk_assignments") if isinstance(risk_matrix.get("lane_risk_assignments"), list) else []:
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "").strip()
        risk_id = str(row.get("risk_class") or "").strip()
        if lane_id and risk_id:
            by_lane[lane_id] = {
                "risk_class": risk_id,
                "risk_notes": str(row.get("notes") or ""),
                "risk_class_detail": risk_classes.get(risk_id) or {},
            }
    return risk_classes, by_lane


def _route_decision() -> Dict[str, Any]:
    return {
        "selected_route": "NO_LLM",
        "reason": "Queue truth, domain registry projections, and explainability traces must remain deterministic and machine-verifiable.",
        "escalation_trigger": None,
        "fallback_route": "NO_LLM",
        "task_class": "implementation",
        "risk_tier": "medium",
        "scope_shape": "multi_surface_coupled",
        "worker_topology": "single",
        "verification_class": "validator_required",
        "verification_plan": [
            "execute explainability generator",
            "compile generator and dedicated test file",
            "parse generated artifacts via json.tool",
            "run source-of-truth map regression check",
            "ship provenance parity and operator usability coverage for XU-503",
        ],
        "fold_in_target": "queue_continuity",
    }


def _ticket_paths_for_slice(slice_id: str) -> List[str]:
    if slice_id in TICKET_WRITE_PATHS:
        return list(TICKET_WRITE_PATHS[slice_id])
    stem = slice_id.lower().replace("-", "_")
    return [f"state/continuity/latest/{stem}_*", f"reports/{stem}_*"]


def _verification_commands(repo_root: Path, stamp: str) -> List[str]:
    return [
        f"python scripts/lane_explainability_panel_runtime.py --repo-root {repo_root} --stamp {stamp} --json",
        "python -m py_compile scripts/lane_explainability_panel_runtime.py tests/test_xu_503_lane_explainability_panel_runtime.py",
        "./.venv/bin/pytest -q tests/test_xu_503_lane_explainability_panel_runtime.py",
    ]


def _show_slice_command(repo_root: Path, slice_id: str, platform: str) -> str:
    return f"python scripts/lane_explainability_panel_runtime.py --repo-root {repo_root} --show-slice {slice_id} --platform {platform}"


def _registry_index(registry_doc: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_slice: Dict[str, Dict[str, Any]] = {}
    by_lane: Dict[str, Dict[str, Any]] = {}
    lanes = registry_doc.get("lane_registry") if isinstance(registry_doc.get("lane_registry"), list) else []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        lane_id = str(lane.get("lane_id") or "").strip()
        if lane_id:
            by_lane[lane_id] = lane
        for cap in lane.get("capability_inventory") or []:
            if not isinstance(cap, dict):
                continue
            slice_id = str(cap.get("slice_id") or "").strip()
            if not slice_id:
                continue
            by_slice[slice_id] = {
                "lane_id": lane_id,
                "lane_name": str(lane.get("lane_name") or ""),
                "risk_class": str(lane.get("risk_class") or ""),
                "activation_posture": str(lane.get("activation_posture") or ""),
                "owners": lane.get("owners") if isinstance(lane.get("owners"), dict) else {},
                "projection": cap,
                "health_projection": lane.get("health_projection") if isinstance(lane.get("health_projection"), dict) else {},
                "generated_at": str(registry_doc.get("generated_at") or ""),
            }
    return by_slice, by_lane


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
    malformed_payload: bool,
) -> str:
    if malformed_payload or not has_owner_tuple or not has_risk_assignment:
        return "FAIL_CLOSED_REVIEW"
    return state_mapping.get(queue_state, "FAIL_CLOSED_REVIEW")


def _artifact_entry(ref: str, source: str, slice_id: str, repo_root: Path) -> Dict[str, Any]:
    rel = str(ref or "").strip()
    entry = {
        "ref": rel,
        "slice_id": slice_id,
        "source": source,
    }
    if rel:
        entry["exists"] = (repo_root / rel).exists()
    return entry


def _dependency_trace(
    *,
    queue_index: Dict[str, Dict[str, Any]],
    start_id: str,
    max_depth: int = 4,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def walk(slice_id: str, depth: int) -> None:
        if depth > max_depth or slice_id in seen:
            return
        seen.add(slice_id)
        row = queue_index.get(slice_id) or {}
        if not row:
            rows.append(
                {
                    "slice_id": slice_id,
                    "depth": depth,
                    "state": "UNKNOWN",
                    "dependencies": [],
                    "status_reason": None,
                    "closeout_report_ref": None,
                    "evidence_ref_count": 0,
                }
            )
            return
        deps = row.get("dependencies") if isinstance(row.get("dependencies"), list) else []
        rows.append(
            {
                "slice_id": slice_id,
                "depth": depth,
                "lane_id": str(row.get("lane_id") or ""),
                "lane_name": str(row.get("lane_name") or ""),
                "title": str(row.get("title") or ""),
                "state": str(row.get("state") or "UNKNOWN"),
                "dependencies": [str(dep) for dep in deps],
                "status_reason": str(row.get("status_reason") or "") or None,
                "closeout_report_ref": str(row.get("closeout_report_ref") or "") or None,
                "evidence_ref_count": len(row.get("evidence_refs") or []) if isinstance(row.get("evidence_refs"), list) else 0,
            }
        )
        for dep in deps:
            walk(str(dep), depth + 1)

    walk(start_id, 0)
    return rows


def _artifact_provenance(
    *,
    trace_rows: List[Dict[str, Any]],
    queue_index: Dict[str, Dict[str, Any]],
    repo_root: Path,
) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for trace_row in trace_rows:
        slice_id = str(trace_row.get("slice_id") or "").strip()
        if not slice_id:
            continue
        queue_row = queue_index.get(slice_id) or {}
        report_ref = str(queue_row.get("closeout_report_ref") or "").strip()
        if report_ref and (slice_id, report_ref) not in seen:
            seen.add((slice_id, report_ref))
            artifacts.append(_artifact_entry(report_ref, "closeout_report_ref", slice_id, repo_root))
        evidence_refs = queue_row.get("evidence_refs") if isinstance(queue_row.get("evidence_refs"), list) else []
        for ref in evidence_refs[:2]:
            ref_text = str(ref or "").strip()
            if ref_text and (slice_id, ref_text) not in seen:
                seen.add((slice_id, ref_text))
                artifacts.append(_artifact_entry(ref_text, "evidence_ref", slice_id, repo_root))
    return artifacts


def _projection_status(queue_state: str, registry_state: Optional[str], projection_present: bool) -> str:
    if not projection_present:
        return "missing_projection"
    if str(registry_state or "") != str(queue_state or ""):
        return "projection_drift"
    return "in_sync"


def _current_blocker_causes(
    *,
    slice_id: str,
    queue_state: str,
    unresolved_dependencies: List[str],
    dependency_states: Dict[str, str],
) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    if queue_state == "DEPENDENCY_BLOCKED":
        causes.append(
            {
                "constraint_id": "QUEUE_DEPENDENCY_BLOCKED",
                "message": f"Queue truth still marks {slice_id} as DEPENDENCY_BLOCKED.",
                "source_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
            }
        )
        for dep in unresolved_dependencies:
            dep_state = dependency_states.get(dep, "UNKNOWN")
            causes.append(
                {
                    "constraint_id": f"QUEUE_DEP::{dep}",
                    "message": f"Dependency {dep} is {dep_state} in queue truth.",
                    "source_ref": "state/continuity/latest/true_expanded_roadmap_queue_layer.json",
                }
            )
    return causes


def _projection_blocker_causes(
    *,
    slice_id: str,
    queue_state: str,
    registry_projection: Dict[str, Any],
    registry_generated_at: str,
    queue_index: Dict[str, Dict[str, Any]],
    registry_path_rel: str,
) -> List[Dict[str, Any]]:
    causes: List[Dict[str, Any]] = []
    projection_state = str(registry_projection.get("state") or "UNKNOWN")
    if projection_state != queue_state:
        causes.append(
            {
                "constraint_id": "REGISTRY_STATE_DRIFT",
                "message": f"XB-402 projection shows {slice_id} as {projection_state} from {registry_generated_at}, but queue truth is {queue_state}.",
                "source_ref": registry_path_rel,
            }
        )
    projected_dep_states = registry_projection.get("dependency_states") if isinstance(registry_projection.get("dependency_states"), dict) else {}
    unresolved = registry_projection.get("unresolved_dependencies") if isinstance(registry_projection.get("unresolved_dependencies"), list) else []
    for dep in unresolved:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        projected = str(projected_dep_states.get(dep_id) or "UNKNOWN")
        current = str((queue_index.get(dep_id) or {}).get("state") or "UNKNOWN")
        causes.append(
            {
                "constraint_id": f"REGISTRY_DEP::{dep_id}",
                "message": f"Projected blocker {dep_id}={projected} in XB-402, while queue truth now shows {dep_id}={current}.",
                "source_ref": registry_path_rel,
            }
        )
    return causes


def _remediation_paths(
    *,
    slice_id: str,
    platform: str,
    queue_state: str,
    projection_status: str,
    repo_root: Path,
    stamp: str,
    queue_path_rel: str,
    registry_path_rel: str,
) -> List[Dict[str, str]]:
    actions: List[Dict[str, str]] = [
        {
            "label": "Inspect queue truth",
            "command": f"python -m json.tool {queue_path_rel}",
        },
        {
            "label": "Show explainability panel",
            "command": _show_slice_command(repo_root, slice_id, platform),
        },
    ]
    if projection_status in {"projection_drift", "missing_projection"}:
        actions.append(
            {
                "label": "Refresh registry projection",
                "command": f"python scripts/domain_capability_registry_runtime.py --repo-root {repo_root} --stamp {stamp} --json",
            }
        )
    if queue_state in {"READY", "DONE"}:
        actions.append(
            {
                "label": "Inspect registry projection",
                "command": f"python -m json.tool {registry_path_rel}",
            }
        )
    return actions[:3]


def _rendered_markdown(
    *,
    slice_id: str,
    operator_state: str,
    blocker_causes: List[Dict[str, Any]],
    remediation_paths: List[Dict[str, str]],
    queue_ref: str,
    registry_ref: str,
    projection_status: str,
    workspace_command: str,
    trace_count: int,
    artifact_ref_count: int,
) -> str:
    emoji = EMOJI_BY_OPERATOR_STATE.get(operator_state, "🔴")
    headline = HEADLINE_BY_OPERATOR_STATE.get(operator_state, "FAIL-CLOSED REVIEW")
    if projection_status == "projection_drift":
        headline = f"{headline} · PROJECTION DRIFT"
    elif projection_status == "missing_projection":
        headline = f"{headline} · PROJECTION MISSING"

    failing_lines = [
        f"- `{row.get('constraint_id')}`: {row.get('message')}"
        for row in blocker_causes
        if isinstance(row, dict)
    ] or ["- None."]

    remediation_lines = [
        f"- {row.get('label')}: `Run: {row.get('command')}`"
        for row in remediation_paths
        if isinstance(row, dict)
    ] or ["- Review: `state/continuity/latest/true_expanded_roadmap_queue_layer.json`"]

    return (
        f"### 1. Headline Status\n{emoji} {headline} — {slice_id}\n\n"
        f"### 2. Failing Constraints\n" + "\n".join(failing_lines) + "\n\n"
        f"### 3. Immediate Action (Remediation)\n" + "\n".join(remediation_lines) + "\n\n"
        f"### 4. Telemetry Footer\n"
        f"- queue_ref: {queue_ref}\n"
        f"- registry_ref: {registry_ref}\n"
        f"- projection_status: {projection_status}\n"
        f"- trace_count: {trace_count}\n"
        f"- artifact_ref_count: {artifact_ref_count}\n"
        f"- workspace_command: `{workspace_command}`"
    )


def _build_panel(
    *,
    repo_root: Path,
    queue_index: Dict[str, Dict[str, Any]],
    state_mapping: Dict[str, str],
    owner_map: Dict[str, Dict[str, str]],
    risk_by_lane: Dict[str, Dict[str, Any]],
    registry_by_slice: Dict[str, Dict[str, Any]],
    slice_id: str,
    platform: str,
    generated_at: str,
    stamp: str,
    queue_path_rel: str,
    registry_path_rel: str,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    override = override or {}
    queue_row = queue_index.get(slice_id) or {}
    malformed_payload = not bool(queue_row)
    lane_id = str(queue_row.get("lane_id") or "")
    queue_state = str(queue_row.get("state") or "UNKNOWN")
    title = str(queue_row.get("title") or slice_id)
    dependencies = queue_row.get("dependencies") if isinstance(queue_row.get("dependencies"), list) else []
    dependency_states, unresolved_dependencies = _dependency_states(queue_index, dependencies)

    owners = dict(owner_map.get(lane_id) or {})
    if override.get("remove_owner_tuple"):
        owners = {}
    risk = dict(risk_by_lane.get(lane_id) or {})
    if override.get("remove_risk_assignment"):
        risk = {}

    registry_entry = registry_by_slice.get(slice_id)
    if override.get("remove_projection"):
        registry_entry = None
    registry_projection = dict((registry_entry or {}).get("projection") or {})
    projection_present = bool(registry_entry and registry_projection)
    projection_state = str(registry_projection.get("state") or "") if projection_present else None
    projection_status = _projection_status(queue_state, projection_state, projection_present)

    operator_state = _operator_state(
        queue_state,
        state_mapping=state_mapping,
        has_owner_tuple=bool(owners),
        has_risk_assignment=bool(risk),
        malformed_payload=malformed_payload,
    )

    if operator_state == "FAIL_CLOSED_REVIEW":
        blocker_causes = []
        if malformed_payload:
            blocker_causes.append(
                {
                    "constraint_id": "QUEUE_SLICE_MISSING",
                    "message": f"Missing queue-truth slice payload for {slice_id}.",
                    "source_ref": queue_path_rel,
                }
            )
        if not owners:
            blocker_causes.append(
                {
                    "constraint_id": "XG_OWNER_TUPLE",
                    "message": f"Missing owner tuple for lane {lane_id or 'UNKNOWN'}.",
                    "source_ref": "state/continuity/latest/xg_801_c3_activation_owner_registry_2026-03-28.json",
                }
            )
        if not risk:
            blocker_causes.append(
                {
                    "constraint_id": "XG_RISK_ASSIGNMENT",
                    "message": f"Missing risk assignment for lane {lane_id or 'UNKNOWN'}.",
                    "source_ref": "state/continuity/latest/xg_801_c3_activation_risk_matrix_2026-03-28.json",
                }
            )
        dependency_trace: List[Dict[str, Any]] = [] if malformed_payload else _dependency_trace(queue_index=queue_index, start_id=slice_id)
        artifact_provenance: List[Dict[str, Any]] = [] if malformed_payload else _artifact_provenance(trace_rows=dependency_trace, queue_index=queue_index, repo_root=repo_root)
        if projection_present:
            blocker_causes.extend(
                _projection_blocker_causes(
                    slice_id=slice_id,
                    queue_state=queue_state,
                    registry_projection=registry_projection,
                    registry_generated_at=str((registry_entry or {}).get("generated_at") or ""),
                    queue_index=queue_index,
                    registry_path_rel=registry_path_rel,
                )
            )
        elif projection_status == "missing_projection":
            blocker_causes.append(
                {
                    "constraint_id": "REGISTRY_PROJECTION_MISSING",
                    "message": f"No XB-402 projection was found for {slice_id}.",
                    "source_ref": registry_path_rel,
                }
            )
    else:
        dependency_trace = _dependency_trace(queue_index=queue_index, start_id=slice_id)
        artifact_provenance = _artifact_provenance(trace_rows=dependency_trace, queue_index=queue_index, repo_root=repo_root)
        blocker_causes = _current_blocker_causes(
            slice_id=slice_id,
            queue_state=queue_state,
            unresolved_dependencies=unresolved_dependencies,
            dependency_states=dependency_states,
        )
        if projection_present:
            blocker_causes.extend(
                _projection_blocker_causes(
                    slice_id=slice_id,
                    queue_state=queue_state,
                    registry_projection=registry_projection,
                    registry_generated_at=str((registry_entry or {}).get("generated_at") or ""),
                    queue_index=queue_index,
                    registry_path_rel=registry_path_rel,
                )
            )
        elif projection_status == "missing_projection":
            blocker_causes.append(
                {
                    "constraint_id": "REGISTRY_PROJECTION_MISSING",
                    "message": f"No XB-402 projection was found for {slice_id}.",
                    "source_ref": registry_path_rel,
                }
            )

    remediation_paths = _remediation_paths(
        slice_id=slice_id,
        platform=platform,
        queue_state=queue_state,
        projection_status=projection_status,
        repo_root=repo_root,
        stamp=stamp,
        queue_path_rel=queue_path_rel,
        registry_path_rel=registry_path_rel,
    )
    workspace_command = _show_slice_command(repo_root, slice_id, platform)

    panel = {
        "workspace_id": f"xu503_{slice_id.lower().replace('-', '_')}_explainability_panel",
        "slice_id": slice_id,
        "title": title,
        "platform": platform,
        "lane_id": lane_id,
        "lane_name": str(queue_row.get("lane_name") or ""),
        "queue_state": queue_state,
        "operator_state": operator_state,
        "projection_status": projection_status,
        "risk_class": str(risk.get("risk_class") or "UNKNOWN"),
        "risk_notes": str(risk.get("risk_notes") or ""),
        "owners": owners,
        "queue_ref": queue_path_rel,
        "registry_ref": registry_path_rel,
        "registry_generated_at": str((registry_entry or {}).get("generated_at") or ""),
        "registry_projection": registry_projection if projection_present else None,
        "dependencies": [str(dep) for dep in dependencies],
        "dependency_states": dependency_states,
        "unresolved_dependencies": unresolved_dependencies,
        "objective": str(queue_row.get("objective") or ""),
        "closeout_condition": str(queue_row.get("closeout_condition") or ""),
        "status_reason": str(queue_row.get("status_reason") or ""),
        "closeout_report_ref": str(queue_row.get("closeout_report_ref") or "") or None,
        "evidence_refs": [str(ref) for ref in (queue_row.get("evidence_refs") or [])] if isinstance(queue_row.get("evidence_refs"), list) else [],
        "artifact_provenance": artifact_provenance,
        "artifact_ref_count": len(artifact_provenance),
        "dependency_trace": dependency_trace,
        "trace_count": len(dependency_trace),
        "blocker_causes": blocker_causes,
        "remediation_paths": remediation_paths,
        "panel_rendered": operator_state != "FAIL_CLOSED_REVIEW",
        "workspace_command": workspace_command,
        "verification_commands": _verification_commands(repo_root, stamp),
        "allowed_write_paths": _ticket_paths_for_slice("XU-503"),
        "generated_at": generated_at,
    }
    panel["rendered_markdown"] = _rendered_markdown(
        slice_id=slice_id,
        operator_state=operator_state,
        blocker_causes=blocker_causes,
        remediation_paths=remediation_paths,
        queue_ref=queue_path_rel,
        registry_ref=registry_path_rel,
        projection_status=projection_status,
        workspace_command=workspace_command,
        trace_count=panel["trace_count"],
        artifact_ref_count=panel["artifact_ref_count"],
    )
    return panel


def build_runtime_package(
    *,
    repo_root: Path,
    queue_doc: Dict[str, Any],
    state_model: Dict[str, Any],
    registry_doc: Dict[str, Any],
    risk_matrix: Dict[str, Any],
    owner_registry: Dict[str, Any],
    fixture: Dict[str, Any],
    generated_at: str,
    stamp: str,
) -> Dict[str, Any]:
    queue_index = _queue_index(queue_doc)
    state_mapping = _state_map(state_model)
    _, risk_by_lane = _risk_map(risk_matrix)
    owner_map = _owner_map(owner_registry)
    registry_by_slice, registry_by_lane = _registry_index(registry_doc)

    queue_path_rel = relpath(DEFAULT_QUEUE_PATH, repo_root)
    registry_path_rel = relpath(DEFAULT_REGISTRY_PATH, repo_root)

    target_row = queue_index.get("XU-503") or {}
    observed_state = str(target_row.get("state") or "UNKNOWN")
    dependencies = target_row.get("dependencies") if isinstance(target_row.get("dependencies"), list) else []
    dependency_states, unresolved_dependencies = _dependency_states(queue_index, dependencies)

    focus_targets = fixture.get("focus_targets") if isinstance(fixture.get("focus_targets"), list) else []
    fail_closed_targets = fixture.get("fail_closed_probes") if isinstance(fixture.get("fail_closed_probes"), list) else []

    focus_panels = [
        _build_panel(
            repo_root=repo_root,
            queue_index=queue_index,
            state_mapping=state_mapping,
            owner_map=owner_map,
            risk_by_lane=risk_by_lane,
            registry_by_slice=registry_by_slice,
            slice_id=str(target.get("slice_id") or ""),
            platform=str(target.get("platform") or "telegram"),
            generated_at=generated_at,
            stamp=stamp,
            queue_path_rel=queue_path_rel,
            registry_path_rel=registry_path_rel,
        )
        for target in focus_targets
        if isinstance(target, dict) and str(target.get("slice_id") or "").strip()
    ]

    fail_closed_panels = [
        _build_panel(
            repo_root=repo_root,
            queue_index=queue_index,
            state_mapping=state_mapping,
            owner_map=owner_map,
            risk_by_lane=risk_by_lane,
            registry_by_slice=registry_by_slice,
            slice_id=str(target.get("slice_id") or ""),
            platform=str(target.get("platform") or "telegram"),
            generated_at=generated_at,
            stamp=stamp,
            queue_path_rel=queue_path_rel,
            registry_path_rel=registry_path_rel,
            override=target.get("override") if isinstance(target.get("override"), dict) else {},
        )
        for target in fail_closed_targets
        if isinstance(target, dict) and str(target.get("slice_id") or "").strip()
    ]

    lane_projection_summary: List[Dict[str, Any]] = []
    lane_ids = sorted({str(panel.get("lane_id") or "") for panel in focus_panels if str(panel.get("lane_id") or "").strip()})
    for lane_id in lane_ids:
        queue_rows = [row for row in queue_doc.get("slices") or [] if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id]
        queue_ready = sum(1 for row in queue_rows if str(row.get("state") or "") == "READY")
        queue_done = sum(1 for row in queue_rows if str(row.get("state") or "") == "DONE")
        queue_blocked = sum(1 for row in queue_rows if str(row.get("state") or "") == "DEPENDENCY_BLOCKED")

        registry_lane = registry_by_lane.get(lane_id) or {}
        registry_caps = registry_lane.get("capability_inventory") if isinstance(registry_lane.get("capability_inventory"), list) else []
        registry_ready = sum(1 for row in registry_caps if str(row.get("state") or "") == "READY")
        registry_done = sum(1 for row in registry_caps if str(row.get("state") or "") == "DONE")
        registry_blocked = sum(1 for row in registry_caps if str(row.get("state") or "") == "DEPENDENCY_BLOCKED")
        drifted = [panel.get("slice_id") for panel in focus_panels if panel.get("lane_id") == lane_id and panel.get("projection_status") == "projection_drift"]

        lane_projection_summary.append(
            {
                "lane_id": lane_id,
                "lane_name": str((queue_rows[0] if queue_rows else {}).get("lane_name") or str(registry_lane.get("lane_name") or f"{lane_id} lane")),
                "queue_counts": {
                    "ready": queue_ready,
                    "done": queue_done,
                    "blocked": queue_blocked,
                },
                "registry_counts": {
                    "ready": registry_ready,
                    "done": registry_done,
                    "blocked": registry_blocked,
                },
                "drifted_slice_ids": drifted,
            }
        )

    runtime_payload = {
        "schema": "clawd.xu_503.lane_explainability_panel_runtime.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "queue_precondition": {
            "authoritative_queue_ref": queue_path_rel,
            "observed_slice_state": observed_state,
            "dependency_states": dependency_states,
            "dependencies_resolved": not unresolved_dependencies,
        },
        "route_decision": _route_decision(),
        "runtime_contract": {
            "queue_truth_authority": True,
            "projection_drift_must_be_explicit": True,
            "stale_projection_is_diagnostic_not_authoritative": True,
            "fail_closed_on_owner_or_risk_gap": True,
            "auto_execute_mutations_allowed": False,
        },
        "program_overview": dict(queue_doc.get("summary") or {}),
        "lane_projection_summary": lane_projection_summary,
        "focus_panels": focus_panels,
        "fail_closed_panels": fail_closed_panels,
    }

    views_payload = {
        "schema": "clawd.xu_503.operator_explainability_views.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "views": focus_panels + fail_closed_panels,
        "status": "PASS" if focus_panels and fail_closed_panels else "FAIL",
    }

    parity_tests: List[Dict[str, Any]] = []
    for target in focus_targets:
        if not isinstance(target, dict):
            continue
        slice_id = str(target.get("slice_id") or "").strip()
        if not slice_id:
            continue
        panel = next(row for row in focus_panels if str(row.get("slice_id") or "") == slice_id)
        expected_trace = [str(dep) for dep in (target.get("expected_dependency_trace") or [])]
        trace_ids = [str(row.get("slice_id") or "") for row in (panel.get("dependency_trace") or [])]
        parity_tests.extend(
            [
                {
                    "test": f"{slice_id}_queue_state_parity",
                    "slice_id": slice_id,
                    "expected": str(target.get("expected_queue_state") or ""),
                    "observed": str(panel.get("queue_state") or ""),
                    "result": "PASS" if str(target.get("expected_queue_state") or "") == str(panel.get("queue_state") or "") else "FAIL",
                },
                {
                    "test": f"{slice_id}_operator_state_parity",
                    "slice_id": slice_id,
                    "expected": str(target.get("expected_operator_state") or ""),
                    "observed": str(panel.get("operator_state") or ""),
                    "result": "PASS" if str(target.get("expected_operator_state") or "") == str(panel.get("operator_state") or "") else "FAIL",
                },
                {
                    "test": f"{slice_id}_projection_status_parity",
                    "slice_id": slice_id,
                    "expected": str(target.get("expected_projection_status") or ""),
                    "observed": str(panel.get("projection_status") or ""),
                    "result": "PASS" if str(target.get("expected_projection_status") or "") == str(panel.get("projection_status") or "") else "FAIL",
                },
                {
                    "test": f"{slice_id}_dependency_trace_contains_expected",
                    "slice_id": slice_id,
                    "expected": expected_trace,
                    "observed": trace_ids,
                    "result": "PASS" if all(dep in trace_ids for dep in expected_trace) else "FAIL",
                },
                {
                    "test": f"{slice_id}_artifact_ref_floor",
                    "slice_id": slice_id,
                    "expected": int(target.get("minimum_artifact_refs") or 0),
                    "observed": int(panel.get("artifact_ref_count") or 0),
                    "result": "PASS" if int(panel.get("artifact_ref_count") or 0) >= int(target.get("minimum_artifact_refs") or 0) else "FAIL",
                },
            ]
        )

    for target in fail_closed_targets:
        if not isinstance(target, dict):
            continue
        slice_id = str(target.get("slice_id") or "").strip()
        panel = next(row for row in fail_closed_panels if str(row.get("slice_id") or "") == slice_id)
        parity_tests.extend(
            [
                {
                    "test": f"{slice_id}_fail_closed_operator_state",
                    "slice_id": slice_id,
                    "expected": str(target.get("expected_operator_state") or ""),
                    "observed": str(panel.get("operator_state") or ""),
                    "result": "PASS" if str(target.get("expected_operator_state") or "") == str(panel.get("operator_state") or "") else "FAIL",
                },
                {
                    "test": f"{slice_id}_fail_closed_panel_suppressed",
                    "slice_id": slice_id,
                    "expected": False,
                    "observed": bool(panel.get("panel_rendered")),
                    "result": "PASS" if bool(panel.get("panel_rendered")) is False else "FAIL",
                },
            ]
        )

    parity_payload = {
        "schema": "clawd.xu_503.provenance_parity_tests.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "tests": parity_tests,
        "status": "PASS" if all(row.get("result") == "PASS" for row in parity_tests) else "FAIL",
    }

    usability_checks: List[Dict[str, Any]] = []
    for panel in focus_panels + fail_closed_panels:
        rendered = str(panel.get("rendered_markdown") or "")
        remediation_paths = panel.get("remediation_paths") if isinstance(panel.get("remediation_paths"), list) else []
        usability_checks.extend(
            [
                {
                    "check": f"{panel['slice_id']}_markdown_has_headline_section",
                    "slice_id": panel["slice_id"],
                    "result": "PASS" if "### 1. Headline Status" in rendered else "FAIL",
                },
                {
                    "check": f"{panel['slice_id']}_markdown_has_immediate_action_section",
                    "slice_id": panel["slice_id"],
                    "result": "PASS" if "### 3. Immediate Action (Remediation)" in rendered else "FAIL",
                },
                {
                    "check": f"{panel['slice_id']}_markdown_has_telemetry_footer",
                    "slice_id": panel["slice_id"],
                    "result": "PASS" if "### 4. Telemetry Footer" in rendered else "FAIL",
                },
                {
                    "check": f"{panel['slice_id']}_remediation_commands_are_full",
                    "slice_id": panel["slice_id"],
                    "result": "PASS" if all(str(row.get("command") or "").startswith("python ") for row in remediation_paths) else "FAIL",
                },
                {
                    "check": f"{panel['slice_id']}_projection_drift_is_explicit_when_present",
                    "slice_id": panel["slice_id"],
                    "result": "PASS"
                    if panel.get("projection_status") != "projection_drift" or "REGISTRY_STATE_DRIFT" in rendered
                    else "FAIL",
                },
            ]
        )
    usability_payload = {
        "schema": "clawd.xu_503.operator_usability_checks.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "checks": usability_checks,
        "status": "PASS" if all(row.get("result") == "PASS" for row in usability_checks) else "FAIL",
    }

    gate_checks = [
        {
            "check": "fresh_truth_check",
            "result": "PASS",
            "evidence_ref": queue_path_rel,
        },
        {
            "check": "verification_check",
            "result": "PASS" if parity_payload["status"] == "PASS" and usability_payload["status"] == "PASS" else "FAIL",
            "evidence_ref": f"state/continuity/latest/xu_503_runtime_validation_{stamp}.json",
        },
        {
            "check": "dependency_health_check",
            "result": "PASS" if not unresolved_dependencies else "FAIL",
            "evidence_ref": queue_path_rel,
        },
        {
            "check": "continuity_coherence_check",
            "result": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
            "evidence_ref": queue_path_rel,
        },
        {
            "check": "blocker_state_check",
            "result": "PASS" if observed_state == "READY" else "FAIL",
            "evidence_ref": queue_path_rel,
        },
        {
            "check": "evidence_quality_check",
            "result": "PASS" if all(int(panel.get("artifact_ref_count") or 0) > 0 for panel in focus_panels) else "FAIL",
            "evidence_ref": relpath(DEFAULT_FIXTURE_PATH, repo_root),
        },
    ]
    gate_result = "allowed" if all(row.get("result") == "PASS" for row in gate_checks) else "forbidden"
    gate_payload = {
        "schema": "clawd.verify_before_resume_gate.v1",
        "slice_id": "XU-503",
        "gate_result": gate_result,
        "evaluated_at": generated_at,
        "evidence_refs": [
            queue_path_rel,
            relpath(DEFAULT_STATE_MODEL_PATH, repo_root),
            relpath(DEFAULT_REGISTRY_PATH, repo_root),
            "docs/ops/xu_501_operator_ux_information_architecture_v2.md",
            "docs/ops/xu_502_lane_action_surface_runtime_v1.md",
            "docs/ops/cockpit_action_card_design_v1.md",
            "docs/ops/human_first_observability_v1.md",
            "docs/ops/low_noise_interaction_policy_v1.md",
            "docs/ops/lane_boundary_contract_v1.md",
            "docs/ops/c3_activation_governance_contract_v1.md",
            relpath(DEFAULT_FIXTURE_PATH, repo_root),
        ],
        "failed_checks": [row["check"] for row in gate_checks if row.get("result") != "PASS"],
        "constraints_if_caution": [],
        "next_recheck_at": generated_at,
        "checks": gate_checks,
    }

    validation_statuses = {
        "queue_state": "PASS" if observed_state in {"READY", "DONE"} else "FAIL",
        "dependencies": "PASS" if not unresolved_dependencies else "FAIL",
        "focus_panels_present": "PASS" if bool(focus_panels) else "FAIL",
        "fail_closed_probes_present": "PASS" if bool(fail_closed_panels) else "FAIL",
        "markdown_cards": "PASS" if all("Immediate Action" in str(row.get("rendered_markdown") or "") for row in focus_panels + fail_closed_panels) else "FAIL",
        "provenance_parity": parity_payload["status"],
        "operator_usability": usability_payload["status"],
        "verify_before_resume": "PASS" if gate_result == "allowed" else "FAIL",
    }

    return {
        "runtime": runtime_payload,
        "views": views_payload,
        "provenance_parity": parity_payload,
        "operator_usability": usability_payload,
        "verify_before_resume": gate_payload,
        "validation_statuses": validation_statuses,
        "route_decision": _route_decision(),
    }


def _select_live_panel(package: Dict[str, Any], slice_id: str) -> Dict[str, Any]:
    for panel in package["runtime"].get("focus_panels") or []:
        if str(panel.get("slice_id") or "") == slice_id:
            return panel
    for panel in package["runtime"].get("fail_closed_panels") or []:
        if str(panel.get("slice_id") or "") == slice_id:
            return panel
    raise SystemExit(f"Explainability panel not found for slice {slice_id}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    queue_path = Path(args.queue_path).resolve()
    state_model_path = Path(args.state_model_path).resolve()
    registry_path = Path(args.registry_path).resolve()
    risk_matrix_path = Path(args.risk_matrix_path).resolve()
    owner_registry_path = Path(args.owner_registry_path).resolve()
    fixture_path = Path(args.fixture_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).date().isoformat()
    generated_at = utc_now_iso()

    queue_doc = load_json(queue_path)
    state_model = load_json(state_model_path)
    registry_doc = load_json(registry_path)
    risk_matrix = load_json(risk_matrix_path)
    owner_registry = load_json(owner_registry_path)
    fixture = load_json(fixture_path)

    package = build_runtime_package(
        repo_root=repo_root,
        queue_doc=queue_doc,
        state_model=state_model,
        registry_doc=registry_doc,
        risk_matrix=risk_matrix,
        owner_registry=owner_registry,
        fixture=fixture,
        generated_at=generated_at,
        stamp=stamp,
    )

    if args.show_slice:
        print(json.dumps(_select_live_panel(package, args.show_slice), ensure_ascii=False, indent=2))
        return 0

    artifacts = {
        "runtime_snapshot": output_dir / f"xu_503_lane_explainability_panel_runtime_{stamp}.json",
        "operator_views": output_dir / f"xu_503_operator_explainability_views_{stamp}.json",
        "provenance_parity_tests": output_dir / f"xu_503_provenance_parity_tests_{stamp}.json",
        "operator_usability_checks": output_dir / f"xu_503_operator_usability_checks_{stamp}.json",
        "verify_before_resume_gate": output_dir / f"xu_503_verify_before_resume_gate_{stamp}.json",
        "runtime_validation": output_dir / f"xu_503_runtime_validation_{stamp}.json",
        "artifact_manifest": output_dir / f"xu_503_runtime_artifact_manifest_{stamp}.json",
    }

    write_json(artifacts["runtime_snapshot"], package["runtime"])
    write_json(artifacts["operator_views"], package["views"])
    write_json(artifacts["provenance_parity_tests"], package["provenance_parity"])
    write_json(artifacts["operator_usability_checks"], package["operator_usability"])
    write_json(artifacts["verify_before_resume_gate"], package["verify_before_resume"])

    check_rows = [
        {"name": "queue_state_ready_or_done", "result": package["validation_statuses"]["queue_state"]},
        {"name": "dependencies_done", "result": package["validation_statuses"]["dependencies"]},
        {"name": "focus_panels_present", "result": package["validation_statuses"]["focus_panels_present"]},
        {"name": "fail_closed_probes_present", "result": package["validation_statuses"]["fail_closed_probes_present"]},
        {"name": "markdown_cards_include_immediate_action", "result": package["validation_statuses"]["markdown_cards"]},
        {"name": "provenance_parity_tests_pass", "result": package["validation_statuses"]["provenance_parity"]},
        {"name": "operator_usability_checks_pass", "result": package["validation_statuses"]["operator_usability"]},
        {"name": "verify_before_resume_gate_allowed", "result": package["validation_statuses"]["verify_before_resume"]},
    ]
    overall_status = "PASS" if all(row.get("result") == "PASS" for row in check_rows) else "FAIL"

    manifest_payload = {
        "schema": "clawd.xu_503.runtime_artifact_manifest.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "status": overall_status,
        "route_decision": package["route_decision"],
        "artifacts": {name: relpath(path, repo_root) for name, path in artifacts.items()},
    }
    write_json(artifacts["artifact_manifest"], manifest_payload)

    validation_payload = {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XU-503",
        "generated_at": generated_at,
        "status": overall_status,
        "checks": check_rows,
        "artifact_refs": manifest_payload["artifacts"],
    }
    write_json(artifacts["runtime_validation"], validation_payload)

    if args.json:
        print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
