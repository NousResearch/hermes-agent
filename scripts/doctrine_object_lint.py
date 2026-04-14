#!/usr/bin/env python3
"""Doctrine Object Lint / Pre-Promotion Check v1.

Validates a doctrine object against Doctrine Object Contract v1 with fail-closed
blocking semantics before promotion/review workflows.

Gate order (fail-closed):
1) schema
2) provenance
3) confidence
4) conflicts
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:  # pragma: no cover (environment wiring)
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "doctrine_object.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "doctrine_object_lint" / "decisions.jsonl"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def normalize_sha256(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def gate_schema(candidate: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    if not isinstance(schema_doc, dict):
        return False, "gate_unavailable", {"error": "schema_not_object"}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(candidate),
        key=lambda err: (list(err.absolute_path), list(err.absolute_schema_path), str(err.message)),
    )
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    details = {
        "error": "schema_validation_failed",
        "data_path": json_ptr(err.absolute_path),
        "schema_path": json_ptr(err.absolute_schema_path),
        "message": str(err.message),
    }
    return False, "schema_invalid", details


def _resolve_source_ref_path(repo_root: Path, raw: str) -> Tuple[Optional[Path], Optional[str]]:
    src = Path(raw).expanduser()
    if src.is_absolute():
        return src.resolve(), None

    resolved = (repo_root / src).resolve()
    if not is_within(repo_root, resolved):
        return None, "relative_path_outside_repo"
    return resolved, None


def gate_provenance(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    refs = candidate.get("source_refs")
    if not isinstance(refs, list) or not refs:
        return False, "provenance_unresolved", {"issues": [{"reason": "source_refs_missing"}]}

    issues: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    checked = 0
    verified_local_sources = 0

    for idx, ref in enumerate(refs):
        checked += 1
        if not isinstance(ref, dict):
            issues.append({"ref_index": idx, "reason": "source_ref_not_object"})
            continue

        source_id = ref.get("source_id")
        locator = ref.get("locator")
        uri_or_path = ref.get("uri_or_path")
        evidence_hash = ref.get("evidence_hash")

        if not isinstance(source_id, str) or not source_id.strip():
            issues.append({"ref_index": idx, "reason": "source_id_missing"})
        else:
            sid = source_id.strip()
            if sid in seen_ids:
                issues.append({"ref_index": idx, "source_id": sid, "reason": "source_id_duplicate"})
            seen_ids.add(sid)

        if not isinstance(locator, str) or not locator.strip():
            issues.append({"ref_index": idx, "source_id": source_id, "reason": "locator_missing"})

        if not isinstance(uri_or_path, str) or not uri_or_path.strip():
            issues.append({"ref_index": idx, "source_id": source_id, "reason": "uri_or_path_missing"})
            continue

        parsed = urlparse(uri_or_path)
        is_http_uri = parsed.scheme.lower() in {"http", "https"}
        if is_http_uri:
            # URL-based refs are accepted as explicit provenance handles.
            continue

        resolved, path_err = _resolve_source_ref_path(repo_root, uri_or_path)
        if path_err is not None:
            issues.append(
                {
                    "ref_index": idx,
                    "source_id": source_id,
                    "reason": path_err,
                    "uri_or_path": uri_or_path,
                }
            )
            continue

        if resolved is None or not resolved.exists() or not resolved.is_file():
            issues.append(
                {
                    "ref_index": idx,
                    "source_id": source_id,
                    "reason": "source_path_unresolved",
                    "uri_or_path": uri_or_path,
                    "resolved": str(resolved) if resolved is not None else None,
                }
            )
            continue

        verified_local_sources += 1

        if isinstance(evidence_hash, str) and evidence_hash.strip():
            declared = normalize_sha256(evidence_hash)
            try:
                actual = file_sha256(resolved)
            except Exception as exc:
                issues.append(
                    {
                        "ref_index": idx,
                        "source_id": source_id,
                        "reason": "evidence_hash_compute_failed",
                        "detail": str(exc),
                    }
                )
                continue

            if declared != actual:
                issues.append(
                    {
                        "ref_index": idx,
                        "source_id": source_id,
                        "reason": "evidence_hash_mismatch",
                        "declared": declared,
                        "actual": actual,
                        "uri_or_path": uri_or_path,
                    }
                )

    if issues:
        return (
            False,
            "provenance_unresolved",
            {
                "checked": checked,
                "verified_local_sources": verified_local_sources,
                "issues": issues,
            },
        )

    return True, None, {"checked": checked, "verified_local_sources": verified_local_sources}


def gate_confidence(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    confidence = candidate.get("confidence")
    governance = candidate.get("governance") if isinstance(candidate.get("governance"), dict) else {}

    if not isinstance(confidence, dict):
        return False, "confidence_semantics_invalid", {"error": "confidence_missing"}

    score = confidence.get("score")
    evidence_quality = confidence.get("evidence_quality")
    uncertainty_notes = confidence.get("uncertainty_notes")
    promotion_state = governance.get("promotion_state")
    status = candidate.get("status")

    if not isinstance(score, (int, float)):
        return False, "confidence_semantics_invalid", {"error": "confidence_score_invalid"}

    numeric_score = float(score)
    if numeric_score < 0.0 or numeric_score > 1.0:
        return (
            False,
            "confidence_semantics_invalid",
            {"error": "confidence_score_out_of_range", "score": numeric_score},
        )

    if not isinstance(evidence_quality, str) or evidence_quality not in {"low", "medium", "high"}:
        return False, "confidence_semantics_invalid", {"error": "evidence_quality_invalid"}

    if numeric_score < 0.5 and (not isinstance(uncertainty_notes, str) or not uncertainty_notes.strip()):
        return (
            False,
            "confidence_semantics_invalid",
            {
                "error": "low_confidence_requires_uncertainty_notes",
                "score": numeric_score,
            },
        )

    if numeric_score >= 0.9 and evidence_quality == "low":
        return (
            False,
            "confidence_semantics_invalid",
            {
                "error": "high_score_with_low_evidence_quality",
                "score": numeric_score,
                "evidence_quality": evidence_quality,
            },
        )

    if status == "active" and numeric_score < 0.5:
        return (
            False,
            "confidence_semantics_invalid",
            {
                "error": "active_doctrine_low_confidence",
                "score": numeric_score,
            },
        )

    if promotion_state == "approved" and numeric_score < 0.5:
        return (
            False,
            "confidence_below_promotion_floor",
            {
                "error": "approved_doctrine_requires_confidence_floor",
                "score": numeric_score,
                "threshold": 0.5,
            },
        )

    return (
        True,
        None,
        {
            "score": numeric_score,
            "evidence_quality": evidence_quality,
            "promotion_state": promotion_state,
        },
    )


def gate_conflicts(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    contradictions = candidate.get("contradictions")
    doctrine_id = candidate.get("doctrine_id")
    governance = candidate.get("governance") if isinstance(candidate.get("governance"), dict) else {}
    promotion_state = governance.get("promotion_state")

    if not isinstance(contradictions, list):
        return False, "conflict_semantics_invalid", {"error": "contradictions_missing"}

    issues: List[Dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    unresolved_high = 0

    for idx, row in enumerate(contradictions):
        if not isinstance(row, dict):
            issues.append({"index": idx, "reason": "contradiction_not_object"})
            continue

        other_id = row.get("doctrine_id")
        relation = row.get("relation")
        severity = row.get("severity")
        resolution_state = row.get("resolution_state")
        resolution_note = row.get("resolution_note")

        if isinstance(other_id, str) and isinstance(doctrine_id, str) and other_id == doctrine_id:
            issues.append({"index": idx, "reason": "self_conflict_reference", "doctrine_id": doctrine_id})

        if isinstance(other_id, str) and isinstance(relation, str):
            edge = (other_id, relation)
            if edge in seen_edges:
                issues.append(
                    {
                        "index": idx,
                        "reason": "duplicate_conflict_edge",
                        "doctrine_id": other_id,
                        "relation": relation,
                    }
                )
            seen_edges.add(edge)

        if severity == "high" and resolution_state == "unresolved":
            unresolved_high += 1

        if severity == "high" and resolution_state in {"unresolved", "in_review"}:
            if not isinstance(resolution_note, str) or not resolution_note.strip():
                issues.append(
                    {
                        "index": idx,
                        "reason": "high_conflict_missing_resolution_note",
                        "doctrine_id": other_id,
                        "resolution_state": resolution_state,
                    }
                )

    if promotion_state == "approved" and unresolved_high > 0:
        issues.append(
            {
                "reason": "unresolved_high_severity_conflict",
                "count": unresolved_high,
                "promotion_state": promotion_state,
            }
        )

    if issues:
        return (
            False,
            "conflict_semantics_invalid",
            {
                "issues": issues,
                "unresolved_high_count": unresolved_high,
            },
        )

    return True, None, {"contradictions_count": len(contradictions), "unresolved_high_count": unresolved_high}


def append_decision_record(
    *,
    decision_log_path: Optional[Path],
    repo_root: Path,
    decision_row: Dict[str, Any],
) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log_path
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()

    if not is_within(repo_root, path):
        return {
            "enabled": True,
            "appended": False,
            "reason": "unsafe_path",
            "path": str(path),
        }

    try:
        if path.exists() and not path.is_file():
            return {
                "enabled": True,
                "appended": False,
                "reason": "path_not_file",
                "path": str(path),
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(decision_row) + "\n")
        return {
            "enabled": True,
            "appended": True,
            "path": str(path),
        }
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(path),
            "error": str(exc),
        }


def evaluate_doctrine_object(*, candidate: Any, candidate_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    evaluated_at = now_iso()

    doctrine_id = None
    if isinstance(candidate, dict) and isinstance(candidate.get("doctrine_id"), str):
        doctrine_id = candidate.get("doctrine_id")

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_reason: Optional[str] = None
    block_gate: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(candidate, schema_path)),
        ("provenance", lambda: gate_provenance(candidate if isinstance(candidate, dict) else {}, repo_root)),
        ("confidence", lambda: gate_confidence(candidate if isinstance(candidate, dict) else {})),
        ("conflicts", lambda: gate_conflicts(candidate if isinstance(candidate, dict) else {})),
    ]

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gates.append({"gate": gate_name, "status": "skipped", "reason": "blocked_by_previous_gate"})
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gates.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_reason = reason or "gate_unavailable"
        block_gate = gate_name
        gates.append({"gate": gate_name, "status": "fail", "reason": block_reason, "details": details})

    try:
        candidate_sha = file_sha256(candidate_path)
    except Exception:
        candidate_sha = None

    decision = "BLOCK" if blocked else "PASS"

    return {
        "schema": "clawd.doctrine_object_lint.decision.v1",
        "evaluated_at": evaluated_at,
        "decision": decision,
        "final_state": "BLOCKED" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "doctrine_id": doctrine_id,
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
        },
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Doctrine object lint / pre-promotion check v1")
    ap.add_argument("--object", required=True, help="Path to doctrine object JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root for relative path resolution")
    ap.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to doctrine object JSON schema",
    )
    ap.add_argument(
        "--decision-log",
        default=str(DEFAULT_DECISION_LOG),
        help="Append-only decision log path (relative to repo root unless absolute)",
    )
    ap.add_argument("--no-decision-log", action="store_true", help="Disable decision recording")
    ap.add_argument("--json", action="store_true", help="Emit pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    object_path = Path(args.object).expanduser().resolve()

    try:
        candidate_doc = load_json_file(object_path)
    except Exception as exc:
        result = {
            "schema": "clawd.doctrine_object_lint.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "doctrine_id": None,
            "candidate": {
                "path": str(object_path),
                "sha256": None,
            },
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {
                        "error": "candidate_json_unreadable",
                        "detail": str(exc),
                    },
                },
                {"gate": "provenance", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "confidence", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "conflicts", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_doctrine_object(
            candidate=candidate_doc,
            candidate_path=object_path,
            repo_root=repo_root,
            schema_path=schema_path,
        )

    decision_log_path: Optional[Path] = None
    if not args.no_decision_log:
        decision_log_path = Path(args.decision_log).expanduser()

    record = append_decision_record(decision_log_path=decision_log_path, repo_root=repo_root, decision_row=result)
    result["decision_record"] = record

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
