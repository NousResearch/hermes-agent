#!/usr/bin/env python3
"""Build domain capability registry runtime artifacts (XB-402)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_QUEUE_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "true_expanded_roadmap_queue_layer.json"
DEFAULT_RISK_MATRIX_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_risk_matrix_2026-03-28.json"
DEFAULT_OWNER_REGISTRY_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "xg_801_c3_activation_owner_registry_2026-03-28.json"
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "domain_capability_registry_runtime_snapshot.schema.json"
DEFAULT_OUTPUT_DIR = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest"


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate XB-402 domain capability registry runtime artifacts")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--queue-path", default=str(DEFAULT_QUEUE_PATH))
    ap.add_argument("--risk-matrix-path", default=str(DEFAULT_RISK_MATRIX_PATH))
    ap.add_argument("--owner-registry-path", default=str(DEFAULT_OWNER_REGISTRY_PATH))
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--stamp", help="Artifact date stamp YYYY-MM-DD (default: today UTC)")
    ap.add_argument("--json", action="store_true", help="Emit manifest JSON")
    return ap.parse_args(argv)


def _owner_tuple_map(owner_registry: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
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


def _risk_posture_map(risk_matrix: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    posture_by_risk: Dict[str, str] = {}
    for row in risk_matrix.get("risk_classes") if isinstance(risk_matrix.get("risk_classes"), list) else []:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id") or "").strip()
        posture = str(row.get("allowed_activation_posture") or "").strip()
        if rid:
            posture_by_risk[rid] = posture

    risk_by_lane: Dict[str, str] = {}
    for row in risk_matrix.get("lane_risk_assignments") if isinstance(risk_matrix.get("lane_risk_assignments"), list) else []:
        if not isinstance(row, dict):
            continue
        lane_id = str(row.get("lane_id") or "").strip()
        risk_class = str(row.get("risk_class") or "").strip()
        if lane_id:
            risk_by_lane[lane_id] = risk_class or "RG0_LOW"

    return posture_by_risk, risk_by_lane


def _resolve_dependency_states(queue_index: Dict[str, Dict[str, Any]], dependencies: Iterable[str]) -> Tuple[Dict[str, str], List[str]]:
    dep_states: Dict[str, str] = {}
    unresolved: List[str] = []
    for dep in dependencies:
        dep_id = str(dep or "").strip()
        if not dep_id:
            continue
        dep_state = str((queue_index.get(dep_id) or {}).get("state") or "UNKNOWN")
        dep_states[dep_id] = dep_state
        if dep_state != "DONE":
            unresolved.append(dep_id)
    return dep_states, unresolved


def _capability_health(slice_state: str, unresolved_dependencies: List[str]) -> str:
    state = (slice_state or "").strip()
    if state == "DONE":
        return "done"
    if unresolved_dependencies or state.endswith("BLOCKED") or state == "BLOCKED" or state == "DEPENDENCY_BLOCKED":
        return "blocked"
    if state == "READY":
        return "ready"
    return "attention"


def _lane_health(capabilities: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(row.get("health_status") for row in capabilities)
    status = "attention"
    if counts.get("blocked", 0) > 0:
        status = "blocked"
    elif counts.get("ready", 0) > 0:
        status = "ready"
    elif capabilities and counts.get("done", 0) == len(capabilities):
        status = "healthy"

    next_ready = next((row.get("slice_id") for row in capabilities if row.get("health_status") == "ready"), None)

    return {
        "overall_status": status,
        "done_count": counts.get("done", 0),
        "ready_count": counts.get("ready", 0),
        "blocked_count": counts.get("blocked", 0),
        "attention_count": counts.get("attention", 0),
        "next_ready_slice": next_ready,
    }


def build_registry(
    *,
    queue_doc: Dict[str, Any],
    risk_matrix: Dict[str, Any],
    owner_registry: Dict[str, Any],
    generated_at: str,
    queue_path_rel: str,
    risk_path_rel: str,
    owner_path_rel: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    queue_rows = queue_doc.get("slices") if isinstance(queue_doc.get("slices"), list) else []
    queue_index = {str(row.get("id")): row for row in queue_rows if isinstance(row, dict) and str(row.get("id") or "").strip()}

    owner_map = _owner_tuple_map(owner_registry)
    posture_by_risk, risk_by_lane = _risk_posture_map(risk_matrix)

    lane_ids = sorted(owner_map.keys())
    lane_registry: List[Dict[str, Any]] = []

    blocked_items: List[Dict[str, Any]] = []

    for lane_id in lane_ids:
        lane_slices = [row for row in queue_rows if isinstance(row, dict) and str(row.get("lane_id") or "") == lane_id]
        lane_slices.sort(key=lambda row: str(row.get("id") or ""))

        capabilities: List[Dict[str, Any]] = []
        for row in lane_slices:
            dependencies = row.get("dependencies") if isinstance(row.get("dependencies"), list) else []
            dep_states, unresolved = _resolve_dependency_states(queue_index, dependencies)
            capability = {
                "slice_id": str(row.get("id") or ""),
                "title": str(row.get("title") or ""),
                "state": str(row.get("state") or "UNKNOWN"),
                "objective": str(row.get("objective") or ""),
                "dependencies": [str(dep) for dep in dependencies],
                "dependency_states": dep_states,
                "unresolved_dependencies": unresolved,
            }
            capability["health_status"] = _capability_health(capability["state"], unresolved)
            capabilities.append(capability)

            if capability["health_status"] == "blocked":
                owners = owner_map.get(lane_id, {})
                blocked_items.append(
                    {
                        "slice_id": capability["slice_id"],
                        "lane_id": lane_id,
                        "state": capability["state"],
                        "unresolved_dependencies": unresolved,
                        "owners": owners,
                        "primary_owner": owners.get("governance_owner", "unassigned"),
                    }
                )

        risk_class = risk_by_lane.get(lane_id, "RG0_LOW")
        lane_health = _lane_health(capabilities)

        lane_registry.append(
            {
                "lane_id": lane_id,
                "lane_name": str((lane_slices[0] if lane_slices else {}).get("lane_name") or f"{lane_id} lane"),
                "risk_class": risk_class,
                "activation_posture": posture_by_risk.get(risk_class, "governance_validation_only"),
                "owners": owner_map.get(lane_id, {
                    "governance_owner": "unassigned",
                    "lane_owner": "unassigned",
                    "release_owner": "unassigned",
                    "incident_owner": "unassigned",
                }),
                "capability_inventory": capabilities,
                "health_projection": lane_health,
            }
        )

    summary = queue_doc.get("summary") if isinstance(queue_doc.get("summary"), dict) else {}
    queue_summary = {
        "total_slices": int(summary.get("total_slices") or len(queue_rows)),
        "required_slices": int(summary.get("required_slices") or sum(1 for row in queue_rows if row.get("posture") != "optional")),
        "optional_slices": int(summary.get("optional_slices") or sum(1 for row in queue_rows if row.get("posture") == "optional")),
        "done_count": int(summary.get("done_count") or 0),
        "ready_count": int(summary.get("ready_count") or 0),
        "dependency_blocked_count": int(summary.get("dependency_blocked_count") or 0),
        "queued_optional_count": int(summary.get("queued_optional_count") or 0),
        "required_open_count": int(summary.get("required_open_count") or 0),
        "first_launch_recommendation": summary.get("first_launch_recommendation") if isinstance(summary.get("first_launch_recommendation"), list) else [],
    }

    owner_counter: Dict[str, Dict[str, Any]] = {}
    for item in blocked_items:
        owner = str(item.get("primary_owner") or "unassigned")
        row = owner_counter.setdefault(owner, {"owner": owner, "blocked_item_count": 0, "lanes": set()})
        row["blocked_item_count"] += 1
        row["lanes"].add(str(item.get("lane_id") or ""))

    by_owner = [
        {
            "owner": row["owner"],
            "blocked_item_count": row["blocked_item_count"],
            "lanes": sorted(lane for lane in row["lanes"] if lane),
        }
        for row in owner_counter.values()
    ]
    by_owner.sort(key=lambda row: (-row["blocked_item_count"], row["owner"]))

    blocker_rollup = {
        "schema": "clawd.xb_402.blocker_owner_rollup.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "blocked_items": blocked_items,
        "by_owner": by_owner,
        "summary": {
            "total_blocked_items": len(blocked_items),
            "distinct_owners": len(by_owner),
        },
    }

    lane_health_rows = [
        {
            "lane_id": row["lane_id"],
            "overall_status": row["health_projection"]["overall_status"],
            "done_count": row["health_projection"]["done_count"],
            "ready_count": row["health_projection"]["ready_count"],
            "blocked_count": row["health_projection"]["blocked_count"],
            "attention_count": row["health_projection"]["attention_count"],
            "next_ready_slice": row["health_projection"]["next_ready_slice"],
        }
        for row in lane_registry
    ]

    blocked_lanes = sum(1 for row in lane_health_rows if row["overall_status"] == "blocked")
    ready_lanes = sum(1 for row in lane_health_rows if row["overall_status"] == "ready")
    healthy_lanes = sum(1 for row in lane_health_rows if row["overall_status"] == "healthy")

    global_status = "attention"
    if blocked_lanes > 0:
        global_status = "attention"
    elif ready_lanes > 0:
        global_status = "ready"
    elif lane_health_rows and healthy_lanes == len(lane_health_rows):
        global_status = "healthy"

    health_projection = {
        "schema": "clawd.xb_402.health_projection_snapshot.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "lane_health": lane_health_rows,
        "global_health": {
            "overall_status": global_status,
            "total_lanes": len(lane_health_rows),
            "healthy_lanes": healthy_lanes,
            "ready_lanes": ready_lanes,
            "blocked_lanes": blocked_lanes,
            "attention_lanes": len(lane_health_rows) - healthy_lanes - ready_lanes - blocked_lanes,
        },
        "queue_summary": queue_summary,
    }

    registry = {
        "schema": "clawd.xb_402.domain_capability_registry_runtime.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "queue_source": queue_path_rel,
        "risk_matrix_source": risk_path_rel,
        "owner_registry_source": owner_path_rel,
        "queue_summary": queue_summary,
        "lane_registry": lane_registry,
    }

    now = dt.datetime.now(dt.timezone.utc)
    iso_year, iso_week, _ = now.isocalendar()
    weekly_packets = {
        "schema": "clawd.xb_402.weekly_project_posture_packet_samples.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "packets": [
            {
                "packet_id": f"xb402_weekly_posture_{iso_year}w{iso_week:02d}",
                "window": "current_week",
                "summary": {
                    "done_count": queue_summary["done_count"],
                    "ready_count": queue_summary["ready_count"],
                    "dependency_blocked_count": queue_summary["dependency_blocked_count"],
                    "required_open_count": queue_summary["required_open_count"],
                    "global_health": health_projection["global_health"]["overall_status"],
                },
                "attention_lanes": [
                    row["lane_id"]
                    for row in lane_health_rows
                    if row["overall_status"] in {"blocked", "attention", "ready"}
                ],
                "recommended_action": "prioritize blocked lanes by owner rollup before new runtime expansion",
            },
            {
                "packet_id": f"xb402_weekly_posture_{iso_year}w{iso_week + 1:02d}_projection",
                "window": "next_week_projection",
                "summary": {
                    "target_required_open_count": max(queue_summary["required_open_count"] - max(1, queue_summary["ready_count"]), 0),
                    "target_blocked_reduction": blocker_rollup["summary"]["total_blocked_items"],
                },
                "milestones": [
                    "close current READY capability slices",
                    "burn down dependency-blocked items using owner rollup",
                    "refresh parity check after queue transitions",
                ],
            },
        ],
    }

    projected_counts = Counter(str(row.get("state") or "UNKNOWN") for row in queue_rows)
    required_open = sum(1 for row in queue_rows if row.get("posture") != "optional" and row.get("state") != "DONE")
    parity_checks = [
        {
            "check": "done_count_parity",
            "expected": queue_summary["done_count"],
            "observed": projected_counts.get("DONE", 0),
        },
        {
            "check": "ready_count_parity",
            "expected": queue_summary["ready_count"],
            "observed": projected_counts.get("READY", 0),
        },
        {
            "check": "dependency_blocked_count_parity",
            "expected": queue_summary["dependency_blocked_count"],
            "observed": projected_counts.get("DEPENDENCY_BLOCKED", 0),
        },
        {
            "check": "queued_optional_count_parity",
            "expected": queue_summary["queued_optional_count"],
            "observed": projected_counts.get("QUEUED_OPTIONAL", 0),
        },
        {
            "check": "required_open_count_parity",
            "expected": queue_summary["required_open_count"],
            "observed": required_open,
        },
    ]

    overall = "PASS" if all(row["expected"] == row["observed"] for row in parity_checks) else "FAIL"
    for row in parity_checks:
        row["result"] = "PASS" if row["expected"] == row["observed"] else "FAIL"

    history_parity = {
        "schema": "clawd.xb_402.history_projection_parity_check.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "status": overall,
        "checks": parity_checks,
    }

    return registry, health_projection, weekly_packets, blocker_rollup, history_parity


def validate_registry_schema(registry: Dict[str, Any], schema_path: Path, generated_at: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    basic_required = {"schema", "slice_id", "generated_at", "queue_summary", "lane_registry"}
    missing = sorted(field for field in basic_required if field not in registry)
    checks.append(
        {
            "name": "basic_required_fields_present",
            "result": "PASS" if not missing else "FAIL",
            "missing": missing,
        }
    )

    if Draft202012Validator is None or FormatChecker is None:
        checks.append(
            {
                "name": "jsonschema_runtime_validation",
                "result": "SKIP",
                "reason": "jsonschema_validator_unavailable",
            }
        )
    else:
        if not schema_path.exists() or not schema_path.is_file():
            checks.append(
                {
                    "name": "jsonschema_runtime_validation",
                    "result": "FAIL",
                    "reason": "schema_missing",
                    "schema_path": str(schema_path),
                }
            )
        else:
            schema_doc = load_json(schema_path)
            validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
            errors = sorted(validator.iter_errors(registry), key=lambda err: (list(err.absolute_path), str(err.message)))
            if not errors:
                checks.append(
                    {
                        "name": "jsonschema_runtime_validation",
                        "result": "PASS",
                        "schema_path": str(schema_path),
                    }
                )
            else:
                err = errors[0]
                checks.append(
                    {
                        "name": "jsonschema_runtime_validation",
                        "result": "FAIL",
                        "schema_path": str(schema_path),
                        "path": "/".join(str(p) for p in err.absolute_path),
                        "message": str(err.message),
                    }
                )

    status = "PASS" if all(row["result"] in {"PASS", "SKIP"} for row in checks) else "FAIL"
    return {
        "schema": "clawd.validation_packet.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "status": status,
        "checks": checks,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    queue_path = Path(args.queue_path).expanduser().resolve()
    risk_path = Path(args.risk_matrix_path).expanduser().resolve()
    owner_path = Path(args.owner_registry_path).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    stamp = args.stamp or dt.datetime.now(dt.timezone.utc).date().isoformat()

    queue_doc = load_json(queue_path)
    risk_matrix = load_json(risk_path)
    owner_registry = load_json(owner_path)

    generated_at = utc_now_iso()

    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(repo_root))
        except Exception:
            return str(path)

    registry, health_projection, weekly_packets, blocker_rollup, history_parity = build_registry(
        queue_doc=queue_doc,
        risk_matrix=risk_matrix,
        owner_registry=owner_registry,
        generated_at=generated_at,
        queue_path_rel=rel(queue_path),
        risk_path_rel=rel(risk_path),
        owner_path_rel=rel(owner_path),
    )

    registry_path = output_dir / f"xb_402_domain_capability_registry_runtime_{stamp}.json"
    validation_path = output_dir / f"xb_402_registry_schema_validation_{stamp}.json"
    health_path = output_dir / f"xb_402_health_projection_snapshot_{stamp}.json"
    weekly_path = output_dir / f"xb_402_weekly_project_posture_packet_samples_{stamp}.json"
    blocker_path = output_dir / f"xb_402_blocker_owner_rollup_artifacts_{stamp}.json"
    parity_path = output_dir / f"xb_402_history_projection_parity_check_{stamp}.json"

    registry["generated_packets"] = {
        "registry_schema_validation": rel(validation_path),
        "health_projection_snapshot": rel(health_path),
        "weekly_project_posture_packet_samples": rel(weekly_path),
        "blocker_owner_rollup_artifacts": rel(blocker_path),
        "history_projection_parity_check": rel(parity_path),
    }

    validation = validate_registry_schema(registry, schema_path, generated_at)

    write_json(registry_path, registry)
    write_json(validation_path, validation)
    write_json(health_path, health_projection)
    write_json(weekly_path, weekly_packets)
    write_json(blocker_path, blocker_rollup)
    write_json(parity_path, history_parity)

    manifest = {
        "schema": "clawd.xb_402.runtime_artifact_manifest.v1",
        "slice_id": "XB-402",
        "generated_at": generated_at,
        "status": "PASS" if validation.get("status") == "PASS" and history_parity.get("status") == "PASS" else "FAIL",
        "artifacts": {
            "registry": rel(registry_path),
            "registry_schema_validation": rel(validation_path),
            "health_projection_snapshot": rel(health_path),
            "weekly_project_posture_packet_samples": rel(weekly_path),
            "blocker_owner_rollup_artifacts": rel(blocker_path),
            "history_projection_parity_check": rel(parity_path),
        },
    }

    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(manifest, ensure_ascii=False))

    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
