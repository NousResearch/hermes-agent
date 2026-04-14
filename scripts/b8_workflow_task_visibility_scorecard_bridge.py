#!/usr/bin/env python3
"""Build an LT-02 visibility scorecard support packet from operator triage output.

This is a support-only bridge utility for DSG-05/LT-02 followthrough:
- input: `state/continuity/latest/operator_triage_console.json`-style payload
- output: `clawd.b8_workflow_task_visibility_scorecard.v1` packet
- optional: run schema + semantic validation before closeout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

EXPECTED_DIMENSIONS = (
    "task_freshness",
    "worker_health",
    "blockage_severity",
    "evidence_quality",
    "execution_path_stability",
)

DEFAULT_LANE_MAPPING = {
    "track": "SX",
    "queue_id": "LT-02",
    "board_id": "DSG-05",
    "lanes": ["B8", "C1", "XU", "triage"],
}

DEFAULT_INTEGRATION_TARGETS = [
    {
        "lane": "triage",
        "surface_id": "operator_triage_console",
        "integration_method": (
            "ingest execution_snapshot.visibility_scorecard and "
            "task_detail.active_task_cards[*].visibility_scorecard"
        ),
        "priority": "immediate",
    },
    {
        "lane": "B8",
        "surface_id": "operator_task_state_critique_packet",
        "integration_method": "attach scorecard as evidence-linked rubric block in critique packets",
        "priority": "queued",
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_to_stamp(value: str) -> str:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return dt.strftime("%Y%m%dT%H%M%SZ").lower()


def _as_dict(parent: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}/{key} must be an object")
    return value


def _as_list(parent: Dict[str, Any], key: str, path: str) -> List[Any]:
    value = parent.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{path}/{key} must be an array")
    return value


def _dimension_vector(raw: Dict[str, Any], path: str) -> Dict[str, float]:
    keys = set(raw.keys())
    expected = set(EXPECTED_DIMENSIONS)
    if keys != expected:
        raise ValueError(f"{path} must contain dimensions {sorted(expected)}")
    out: Dict[str, float] = {}
    for key in EXPECTED_DIMENSIONS:
        try:
            out[key] = float(raw[key])
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"{path}/{key} must be numeric") from exc
    return out


def build_support_packet(triage: Dict[str, Any], source_rel: str) -> Dict[str, Any]:
    generated_at = str(triage.get("generated_at") or "").strip() or _iso_now()
    scorecard_id = f"lt02_operator_triage_runtime_{_iso_to_stamp(generated_at)}"

    execution_snapshot = _as_dict(triage, "execution_snapshot", "$")
    system_scorecard = _as_dict(execution_snapshot, "visibility_scorecard", "$/execution_snapshot")

    task_detail = _as_dict(execution_snapshot, "task_detail", "$/execution_snapshot")
    active_cards = _as_list(task_detail, "active_task_cards", "$/execution_snapshot/task_detail")

    task_scorecards: List[Dict[str, Any]] = []
    for idx, card in enumerate(active_cards):
        if not isinstance(card, dict):
            raise ValueError(f"$/execution_snapshot/task_detail/active_task_cards/{idx} must be an object")

        task_id = str(card.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"$/execution_snapshot/task_detail/active_task_cards/{idx}/task_id is required")

        card_scorecard = _as_dict(card, "visibility_scorecard", f"$/execution_snapshot/task_detail/active_task_cards/{idx}")
        card_dimensions = _as_dict(card_scorecard, "dimensions", f"$/execution_snapshot/task_detail/active_task_cards/{idx}/visibility_scorecard")

        task_scorecards.append(
            {
                "task_id": task_id,
                "score": float(card.get("visibility_score") if card.get("visibility_score") is not None else card_scorecard.get("weighted_score", 0.0)),
                "rating": str(card.get("visibility_rating") or card_scorecard.get("rating") or "critical").strip().lower(),
                "dimensions": _dimension_vector(
                    card_dimensions,
                    f"$/execution_snapshot/task_detail/active_task_cards/{idx}/visibility_scorecard/dimensions",
                ),
                "suppressed": bool(card.get("suppressed")),
                "suppression_reasons": [str(x) for x in (card.get("suppression_reasons") or []) if str(x).strip()],
            }
        )

    system_dimensions = _as_dict(system_scorecard, "dimensions", "$/execution_snapshot/visibility_scorecard")
    system_weights = _as_dict(system_scorecard, "weights", "$/execution_snapshot/visibility_scorecard")
    thresholds = _as_dict(system_scorecard, "thresholds", "$/execution_snapshot/visibility_scorecard")
    task_rating_counts = _as_dict(system_scorecard, "task_rating_counts", "$/execution_snapshot/visibility_scorecard")

    raw_evidence_refs = triage.get("evidence_refs") if isinstance(triage.get("evidence_refs"), list) else []
    evidence_refs = [str(x) for x in raw_evidence_refs if str(x).strip()]
    if source_rel not in evidence_refs:
        evidence_refs.insert(0, source_rel)

    source_records = [source_rel]
    source_obj = triage.get("source") if isinstance(triage.get("source"), dict) else {}
    mission_control_path = str(source_obj.get("mission_control_path") or "").strip()
    if mission_control_path:
        source_records.append(mission_control_path)

    packet = {
        "schema_version": "clawd.b8_workflow_task_visibility_scorecard.v1",
        "scope_posture": "support_only",
        "scorecard_id": scorecard_id,
        "generated_at": generated_at,
        "lane_mapping": DEFAULT_LANE_MAPPING,
        "surface": {
            "surface_id": "operator_triage_console",
            "surface_name": "Operator Triage Console",
            "surface_kind": "operator_triage",
            "snapshot_at": generated_at,
        },
        "system_scorecard": {
            "score": float(system_scorecard.get("score", 0.0)),
            "rating": str(system_scorecard.get("rating") or "critical").strip().lower(),
            "dimensions": _dimension_vector(system_dimensions, "$/execution_snapshot/visibility_scorecard/dimensions"),
            "weights": _dimension_vector(system_weights, "$/execution_snapshot/visibility_scorecard/weights"),
            "thresholds": {
                "healthy_min": float(thresholds.get("healthy_min", 80)),
                "degraded_min": float(thresholds.get("degraded_min", 55)),
            },
            "task_rating_counts": {
                "healthy": int(task_rating_counts.get("healthy", 0)),
                "degraded": int(task_rating_counts.get("degraded", 0)),
                "critical": int(task_rating_counts.get("critical", 0)),
            },
            "task_count": int(system_scorecard.get("task_count", len(task_scorecards))),
            "visible_task_count": int(system_scorecard.get("visible_task_count", len(task_scorecards))),
            "suppressed_task_count": int(system_scorecard.get("suppressed_task_count", 0)),
        },
        "task_scorecards": task_scorecards,
        "integration_targets": DEFAULT_INTEGRATION_TARGETS,
        "evidence_refs": evidence_refs or [source_rel],
        "provenance": {
            "generated_by": "b8_workflow_task_visibility_scorecard_bridge.py",
            "generated_at": generated_at,
            "source_records": source_records,
        },
    }
    return packet


def run_validation(packet: Dict[str, Any], schema_path: Path) -> Dict[str, Any]:
    if Draft202012Validator is None or FormatChecker is None:
        return {"ok": False, "error": "jsonschema_validator_unavailable"}

    schema = _load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(packet),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if errors:
        err = errors[0]
        return {
            "ok": False,
            "error": "schema_validation_failed",
            "data_path": "$/" + "/".join(str(p) for p in err.absolute_path),
            "message": str(err.message),
        }

    from b8_workflow_task_visibility_scorecard_validate import semantic_checks

    issues = semantic_checks(packet)
    if issues:
        return {
            "ok": False,
            "error": "semantic_validation_failed",
            "issues": [issue.as_dict() for issue in issues],
        }

    return {"ok": True}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Bridge operator triage output into LT-02 visibility scorecard support packet")
    parser.add_argument("--triage", required=True, help="Path to operator_triage_console JSON")
    parser.add_argument("--output", required=True, help="Path to write bridged scorecard JSON")
    parser.add_argument(
        "--schema",
        default=str(repo_root / "docs" / "ops" / "schemas" / "workflow_task_visibility_scorecard.v1.schema.json"),
        help="Path to LT-02 scorecard schema",
    )
    parser.add_argument("--validate", action="store_true", help="Run schema + semantic validation on bridged output")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print command result JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    triage_path = Path(args.triage).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    schema_path = Path(args.schema).expanduser().resolve()

    try:
        triage = _load_json(triage_path)
        if not isinstance(triage, dict):
            raise ValueError("triage payload must be a JSON object")

        try:
            repo_root = Path(__file__).resolve().parents[1]
            source_rel = str(triage_path.relative_to(repo_root))
        except Exception:
            source_rel = str(triage_path)

        packet = build_support_packet(triage, source_rel=source_rel)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        result: Dict[str, Any] = {
            "ok": True,
            "source_triage_path": str(triage_path),
            "output_path": str(output_path),
            "scorecard_id": packet.get("scorecard_id"),
        }

        if args.validate:
            validation = run_validation(packet, schema_path=schema_path)
            result["validation"] = validation
            result["ok"] = bool(validation.get("ok"))
            print(json.dumps(result, indent=2 if args.pretty else None))
            return 0 if result["ok"] else 1

        print(json.dumps(result, indent=2 if args.pretty else None))
        return 0

    except Exception as exc:
        payload = {
            "ok": False,
            "error": "bridge_build_failed",
            "source_triage_path": str(triage_path),
            "output_path": str(output_path),
            "detail": str(exc),
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
