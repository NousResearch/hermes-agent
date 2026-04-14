#!/usr/bin/env python3
"""Document intake batch integration gate (Wave 6, v1).

Validates subagent-first doc/PDF batch closeout packets and emits deterministic
PASS/BLOCK decisions with append-only decision logging.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "document_intake_batch_integration.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_ingestion" / "document_intake_batch_integration_decisions.jsonl"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_sha256(raw: str) -> str:
    token = str(raw or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    return token


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def gate_schema(packet: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(packet),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
            "message": str(err.message),
        },
    )


def gate_synthesis_note(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    note = packet.get("synthesis_note") if isinstance(packet.get("synthesis_note"), dict) else None
    if note is None:
        return False, "synthesis_note_failed", {"error": "synthesis_note_missing"}

    raw_path = str(note.get("path") or "").strip()
    if not raw_path:
        return False, "synthesis_note_failed", {"error": "synthesis_note_path_missing"}

    path = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, path):
        return False, "synthesis_note_failed", {"error": "synthesis_note_path_outside_repo", "path": raw_path}
    if not path.exists() or not path.is_file():
        return False, "synthesis_note_failed", {"error": "synthesis_note_path_unresolved", "path": raw_path}

    if not is_within((repo_root / "reports").resolve(), path):
        return False, "synthesis_note_failed", {"error": "synthesis_note_must_live_in_reports", "path": raw_path}

    declared_hash = str(note.get("sha256") or "").strip()
    actual = file_sha256(path)
    if declared_hash:
        if normalize_sha256(declared_hash) != actual:
            return False, "synthesis_note_failed", {
                "error": "synthesis_note_sha256_mismatch",
                "path": raw_path,
                "declared": normalize_sha256(declared_hash),
                "actual": actual,
            }

    return True, None, {"path": raw_path, "sha256": actual}


def gate_inbound_artifacts(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    rows = packet.get("inbound_materials") if isinstance(packet.get("inbound_materials"), list) else []
    if not rows:
        return False, "inbound_artifacts_failed", {"error": "inbound_materials_missing"}

    checks: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            return False, "inbound_artifacts_failed", {"error": "inbound_material_not_object", "index": idx}
        raw_path = str(row.get("path") or "").strip()
        raw_hash = str(row.get("sha256") or "").strip()
        material_id = str(row.get("material_id") or f"material_{idx + 1}")

        if not raw_path:
            return False, "inbound_artifacts_failed", {"error": "inbound_material_path_missing", "index": idx}
        if not raw_hash:
            return False, "inbound_artifacts_failed", {"error": "inbound_material_sha256_missing", "index": idx}

        path = resolve_repo_path(repo_root, raw_path)
        if not is_within(repo_root, path):
            return False, "inbound_artifacts_failed", {"error": "inbound_material_path_outside_repo", "path": raw_path}
        if not path.exists() or not path.is_file():
            return False, "inbound_artifacts_failed", {"error": "inbound_material_path_unresolved", "path": raw_path}

        declared = normalize_sha256(raw_hash)
        actual = file_sha256(path)
        if declared != actual:
            return False, "inbound_artifacts_failed", {
                "error": "inbound_material_sha256_mismatch",
                "path": raw_path,
                "declared": declared,
                "actual": actual,
            }

        checks.append({"material_id": material_id, "path": raw_path, "sha256": actual})

    return True, None, {"checks": checks}


def gate_lane_mapping(packet: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    recommendations = packet.get("recommendations") if isinstance(packet.get("recommendations"), list) else []
    if not recommendations:
        return False, "lane_mapping_failed", {"error": "recommendations_missing"}

    for idx, row in enumerate(recommendations):
        if not isinstance(row, dict):
            return False, "lane_mapping_failed", {"error": "recommendation_not_object", "index": idx}
        lanes = row.get("target_lanes") if isinstance(row.get("target_lanes"), list) else []
        if not lanes:
            return False, "lane_mapping_failed", {"error": "recommendation_target_lanes_missing", "index": idx}

    return True, None, {"recommendation_count": len(recommendations)}


def gate_promotion_tier_accounting(packet: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    recommendations = packet.get("recommendations") if isinstance(packet.get("recommendations"), list) else []
    summary = packet.get("integration_summary") if isinstance(packet.get("integration_summary"), dict) else {}
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}

    observed = {
        "promote_now": 0,
        "promote_later": 0,
        "reference_only": 0,
    }
    for row in recommendations:
        if not isinstance(row, dict):
            continue
        tier = str(row.get("promotion_tier") or "")
        if tier in observed:
            observed[tier] += 1

    declared = {
        "promote_now": int(counts.get("promote_now") or 0),
        "promote_later": int(counts.get("promote_later") or 0),
        "reference_only": int(counts.get("reference_only") or 0),
    }

    if observed != declared:
        return False, "promotion_tier_accounting_failed", {
            "error": "tier_counts_mismatch",
            "observed": observed,
            "declared": declared,
        }

    return True, None, {"counts": observed}


def gate_minimal_edit_plan(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    summary = packet.get("integration_summary") if isinstance(packet.get("integration_summary"), dict) else {}
    promote_now_paths = summary.get("promote_now_edit_paths") if isinstance(summary.get("promote_now_edit_paths"), list) else []

    if len(promote_now_paths) > 20:
        return False, "minimal_edit_plan_failed", {
            "error": "promote_now_edit_paths_exceeds_limit",
            "count": len(promote_now_paths),
            "max": 20,
        }

    for raw in promote_now_paths:
        token = str(raw or "").strip()
        if not token:
            return False, "minimal_edit_plan_failed", {"error": "promote_now_edit_path_missing"}
        resolved = resolve_repo_path(repo_root, token)
        if not is_within(repo_root, resolved):
            return False, "minimal_edit_plan_failed", {"error": "promote_now_edit_path_outside_repo", "path": token}

    return True, None, {"promote_now_edit_count": len(promote_now_paths)}


def append_decision_record(repo_root: Path, decision_log: Optional[Path], payload: Dict[str, Any]) -> Dict[str, Any]:
    if decision_log is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log if decision_log.is_absolute() else (repo_root / decision_log).resolve()
    if not is_within(repo_root, path):
        return {"enabled": True, "appended": False, "reason": "unsafe_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"enabled": True, "appended": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(payload) + "\n")
        return {"enabled": True, "appended": True, "path": str(path)}
    except Exception as exc:
        return {"enabled": True, "appended": False, "reason": "append_failed", "path": str(path), "detail": str(exc)}


def evaluate_packet(packet: Any, packet_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    packet_dict = packet if isinstance(packet, dict) else {}

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(packet, schema_path)),
        ("synthesis_note", lambda: gate_synthesis_note(packet_dict, repo_root)),
        ("inbound_artifacts", lambda: gate_inbound_artifacts(packet_dict, repo_root)),
        ("lane_mapping", lambda: gate_lane_mapping(packet_dict)),
        ("promotion_tier_accounting", lambda: gate_promotion_tier_accounting(packet_dict)),
        ("minimal_edit_plan", lambda: gate_minimal_edit_plan(packet_dict, repo_root)),
    ]

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gates.append({"gate": gate_name, "status": "skipped", "reason": "blocked_by_previous_gate"})
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gates.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_gate = gate_name
        block_reason = reason or "gate_unavailable"
        gates.append({"gate": gate_name, "status": "fail", "reason": block_reason, "details": details})

    try:
        packet_sha = file_sha256(packet_path)
    except Exception:
        packet_sha = None

    return {
        "schema": "clawd.document_intake_batch_integration.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "batch_id": packet_dict.get("batch_id"),
        "packet": {"path": str(packet_path), "sha256": packet_sha},
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Document intake batch integration gate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Integration packet schema path")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Decision log path")
    ap.add_argument("--no-decision-log", action="store_true", help="Disable decision log append")
    ap.add_argument("--packet", required=True, help="Integration packet path")
    ap.add_argument("--json", action="store_true", help="Pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    packet_path = Path(args.packet).expanduser().resolve()
    decision_log = None if args.no_decision_log else Path(args.decision_log).expanduser()

    try:
        packet = load_json_file(packet_path)
    except Exception as exc:
        result = {
            "schema": "clawd.document_intake_batch_integration.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "batch_id": None,
            "packet": {"path": str(packet_path), "sha256": None},
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {"error": "packet_json_unreadable", "detail": str(exc)},
                },
                {"gate": "synthesis_note", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "inbound_artifacts", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "lane_mapping", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "promotion_tier_accounting", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "minimal_edit_plan", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_packet(packet, packet_path, repo_root, schema_path)

    result["decision_record"] = append_decision_record(repo_root, decision_log, result)

    rc = 0 if result.get("decision") == "PASS" else 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
