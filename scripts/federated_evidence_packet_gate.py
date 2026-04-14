#!/usr/bin/env python3
"""LT-08 support gate for clawd.federated_evidence.v1 packets.

Bounded goal:
- validate standalone federated evidence packets (without full triage payload wrapper),
- enforce minimal consistency invariants for retrieval/evidence federation UX,
- fail closed with machine-readable BLOCK reasons.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_OPERATOR_TRIAGE_SCHEMA = (
    DEFAULT_REPO_ROOT / "ops" / "openclaw" / "architecture" / "schemas" / "operator_triage_console.schema.json"
)

RESULT_SCHEMA = "clawd.federated_evidence_validation.result.v1"
PACKET_SCHEMA_ID = "clawd.federated_evidence.v1"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path or "").strip()).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def json_ptr(parts: Sequence[Any]) -> str:
    if not parts:
        return "$"
    return "$/" + "/".join(str(p) for p in parts)


def load_federated_subschema(schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    if not schema_path.exists() or not schema_path.is_file():
        return False, "schema_missing", {"error": "schema_missing", "schema_path": str(schema_path)}, None

    try:
        schema_doc = load_json(schema_path)
    except Exception as exc:
        return False, "schema_unreadable", {"error": "schema_unreadable", "detail": str(exc)}, None

    if not isinstance(schema_doc, dict):
        return False, "schema_not_object", {"error": "schema_not_object"}, None

    props = schema_doc.get("properties") if isinstance(schema_doc.get("properties"), dict) else {}
    federated_schema = props.get("federated_evidence") if isinstance(props.get("federated_evidence"), dict) else None
    if federated_schema is None:
        return False, "schema_missing_federated_subschema", {"error": "schema_missing_federated_subschema"}, None

    normalized = dict(federated_schema)
    normalized.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
    normalized.setdefault("title", "Federated evidence packet schema")
    return True, None, {"schema_path": str(schema_path)}, normalized


def gate_schema(packet: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "schema_validator_unavailable", {"error": "jsonschema_validator_unavailable"}

    ok, reason, details, subschema = load_federated_subschema(schema_path)
    if not ok or subschema is None:
        return False, reason, details

    validator = Draft202012Validator(subschema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(packet),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, details

    err = errors[0]
    return (
        False,
        "schema_invalid",
        {
            "error": "schema_validation_failed",
            "message": str(err.message),
            "data_path": json_ptr(err.absolute_path),
            "schema_path": json_ptr(err.absolute_schema_path),
        },
    )


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def gate_summary_consistency(packet: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    summary = _as_dict(packet.get("summary"))
    sources = _as_list(packet.get("sources_consulted"))
    evidence = _as_list(packet.get("evidence"))

    source_count = summary.get("source_count")
    if source_count is not None:
        try:
            source_count_int = int(source_count)
        except Exception:
            return False, "summary_consistency", {"error": "summary_source_count_not_int"}
        if source_count_int != len(sources):
            return False, "summary_consistency", {
                "error": "summary_source_count_mismatch",
                "expected": len(sources),
                "actual": source_count_int,
            }

    evidence_count = summary.get("evidence_count")
    if evidence_count is not None:
        try:
            evidence_count_int = int(evidence_count)
        except Exception:
            return False, "summary_consistency", {"error": "summary_evidence_count_not_int"}
        if evidence_count_int != len(evidence):
            return False, "summary_consistency", {
                "error": "summary_evidence_count_mismatch",
                "expected": len(evidence),
                "actual": evidence_count_int,
            }

    max_items = summary.get("max_items")
    if max_items is not None:
        try:
            max_items_int = int(max_items)
        except Exception:
            return False, "summary_consistency", {"error": "summary_max_items_not_int"}
        if max_items_int > 0 and len(evidence) > max_items_int:
            return False, "summary_consistency", {
                "error": "evidence_count_exceeds_max_items",
                "evidence_count": len(evidence),
                "max_items": max_items_int,
            }

    return True, None, {
        "source_count": len(sources),
        "evidence_count": len(evidence),
        "max_items": summary.get("max_items"),
    }


def gate_status_consistency(packet: Mapping[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    status = str(packet.get("status") or "").strip().lower()
    summary = _as_dict(packet.get("summary"))
    evidence = _as_list(packet.get("evidence"))
    sources = _as_list(packet.get("sources_consulted"))

    degraded_reasons = [str(x).strip() for x in _as_list(summary.get("degraded_reasons")) if str(x).strip()]
    degraded_reason_count = summary.get("degraded_reason_count")
    if degraded_reason_count is None:
        degraded_reason_count_int = 0
    else:
        try:
            degraded_reason_count_int = int(degraded_reason_count)
        except Exception:
            return False, "status_consistency", {"error": "degraded_reason_count_not_int"}

    degraded_source_statuses = {
        "degraded",
        "blocked",
        "fail_closed",
        "invalid",
        "missing",
        "stale",
        "error",
    }
    degraded_sources = 0
    for row in sources:
        if not isinstance(row, dict):
            continue
        source_status = str(row.get("status") or "").strip().lower()
        if source_status in degraded_source_statuses:
            degraded_sources += 1

    if degraded_reason_count_int and degraded_reason_count_int != len(degraded_reasons):
        return False, "status_consistency", {
            "error": "degraded_reason_count_mismatch",
            "count": degraded_reason_count_int,
            "reasons": len(degraded_reasons),
        }

    if status == "empty" and evidence:
        return False, "status_consistency", {"error": "empty_status_with_evidence", "evidence_count": len(evidence)}

    if status == "ready" and not evidence:
        return False, "status_consistency", {"error": "ready_status_without_evidence"}

    if status == "degraded":
        if degraded_sources == 0 and not degraded_reasons and degraded_reason_count_int == 0:
            return False, "status_consistency", {
                "error": "degraded_status_without_reasons_or_degraded_sources"
            }

    return True, None, {
        "status": status,
        "evidence_count": len(evidence),
        "degraded_sources": degraded_sources,
        "degraded_reason_count": degraded_reason_count_int,
    }


def gate_source_paths(packet: Mapping[str, Any], repo_root: Path, *, require_exists: bool) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    sources = _as_list(packet.get("sources_consulted"))
    checked = 0
    for row in sources:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("path") or "").strip()
        if not raw:
            continue
        resolved = resolve_repo_path(repo_root, raw)
        if not is_within(repo_root, resolved):
            return False, "source_paths", {"error": "source_path_outside_repo", "path": raw}
        if require_exists and not resolved.exists():
            return False, "source_paths", {"error": "source_path_missing", "path": raw}
        checked += 1

    if checked == 0:
        return False, "source_paths", {"error": "no_source_paths_checked"}

    return True, None, {"checked_source_paths": checked, "require_exists": require_exists}


def evaluate_packet(
    packet: Mapping[str, Any],
    *,
    repo_root: Path,
    schema_path: Path,
    require_source_paths_exist: bool,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    for gate_name, fn in [
        ("schema", lambda: gate_schema(packet, schema_path)),
        ("summary_consistency", lambda: gate_summary_consistency(packet)),
        ("status_consistency", lambda: gate_status_consistency(packet)),
        (
            "source_paths",
            lambda: gate_source_paths(packet, repo_root, require_exists=require_source_paths_exist),
        ),
    ]:
        ok, reason, details = fn()
        checks.append({"gate": gate_name, "status": "pass" if ok else "block", "details": details})
        if not ok:
            return {
                "schema": RESULT_SCHEMA,
                "validated_at": now_iso(),
                "decision": "BLOCK",
                "block_gate": gate_name,
                "block_reason": reason or gate_name,
                "checks": checks,
            }

    summary = _as_dict(packet.get("summary"))
    return {
        "schema": RESULT_SCHEMA,
        "validated_at": now_iso(),
        "decision": "PASS",
        "block_gate": None,
        "block_reason": None,
        "checks": checks,
        "projection": {
            "requested_task_id": packet.get("requested_task_id"),
            "status": packet.get("status"),
            "source_count": summary.get("source_count"),
            "evidence_count": summary.get("evidence_count"),
            "degraded_reason_count": summary.get("degraded_reason_count"),
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate clawd.federated_evidence.v1 packets")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root used for source-path checks")
    ap.add_argument("--schema-path", default=str(DEFAULT_OPERATOR_TRIAGE_SCHEMA), help="Path to operator triage schema")
    ap.add_argument("--packet", required=True, help="Path to packet JSON (standalone federated_evidence object)")
    ap.add_argument(
        "--allow-missing-source-paths",
        action="store_true",
        help="Allow source paths that do not exist on disk (still enforces in-repo constraint)",
    )
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON result")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser()
    if not schema_path.is_absolute():
        schema_path = (repo_root / schema_path).resolve()
    else:
        schema_path = schema_path.resolve()

    packet_path = Path(args.packet).expanduser()
    if not packet_path.is_absolute():
        packet_path = (repo_root / packet_path).resolve()
    else:
        packet_path = packet_path.resolve()

    if not packet_path.exists() or not packet_path.is_file():
        result = {
            "schema": RESULT_SCHEMA,
            "validated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "packet_load",
            "block_reason": "packet_missing",
            "checks": [{"gate": "packet_load", "status": "block", "details": {"path": str(packet_path)}}],
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("BLOCK packet_missing", str(packet_path))
        return 2

    try:
        packet_obj = load_json(packet_path)
    except Exception as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "validated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "packet_load",
            "block_reason": "packet_unreadable",
            "checks": [
                {
                    "gate": "packet_load",
                    "status": "block",
                    "details": {"path": str(packet_path), "error": "packet_unreadable", "detail": str(exc)},
                }
            ],
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("BLOCK packet_unreadable", str(exc))
        return 2

    if not isinstance(packet_obj, dict):
        result = {
            "schema": RESULT_SCHEMA,
            "validated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "packet_load",
            "block_reason": "packet_not_object",
            "checks": [
                {
                    "gate": "packet_load",
                    "status": "block",
                    "details": {"path": str(packet_path), "error": "packet_not_object"},
                }
            ],
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("BLOCK packet_not_object")
        return 2

    if str(packet_obj.get("schema") or "").strip() != PACKET_SCHEMA_ID:
        result = {
            "schema": RESULT_SCHEMA,
            "validated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "packet_load",
            "block_reason": "packet_schema_mismatch",
            "checks": [
                {
                    "gate": "packet_load",
                    "status": "block",
                    "details": {
                        "error": "packet_schema_mismatch",
                        "expected": PACKET_SCHEMA_ID,
                        "actual": packet_obj.get("schema"),
                    },
                }
            ],
        }
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("BLOCK packet_schema_mismatch")
        return 2

    result = evaluate_packet(
        packet_obj,
        repo_root=repo_root,
        schema_path=schema_path,
        require_source_paths_exist=not bool(args.allow_missing_source_paths),
    )
    result["packet_path"] = str(packet_path)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        decision = result.get("decision")
        print(f"{decision} federated_evidence_packet_gate")
        if decision != "PASS":
            print(f"block_gate={result.get('block_gate')} block_reason={result.get('block_reason')}")

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
