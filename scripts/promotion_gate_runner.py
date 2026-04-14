#!/usr/bin/env python3
"""Deterministic Promotion Gate Runner v1.

Evaluates a promotion candidate JSON against the six gates from
`docs/ops/promotion_protocol_contract_v1.md` and emits machine-readable
pass/block output.

Fail-closed behavior:
- Unknown/unavailable validators block.
- Any gate error blocks.
- Remaining gates are marked `skipped` after first fail.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover (environment wiring)
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "promotion_candidate.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "promotion_gate_runner" / "decisions.jsonl"

CONFIDENCE_THRESHOLDS = {
    "doctrine": 0.80,
    "playbook": 0.70,
    "memory": 0.60,
}

REVIEW_ROLE_RULES = {
    "doctrine": {"VALIDATOR", "LIBRARIAN"},
    "memory": {"VALIDATOR", "LIBRARIAN"},
    # Contract says VALIDATOR or domain-owner role; schema constrains role enum,
    # so treat any non-null role as domain-owner candidate for v1.
    "playbook": {"PLANNER", "EXECUTOR", "VALIDATOR", "RESEARCHER", "SRE", "LIBRARIAN"},
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def stable_json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (repo_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


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


def gate_provenance(candidate: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    refs = candidate.get("source_refs")
    issues: List[Dict[str, Any]] = []
    checked = 0

    if not isinstance(refs, list) or not refs:
        return False, "provenance_unresolved", {"issues": [{"reason": "source_refs_missing"}]}

    for idx, ref in enumerate(refs):
        checked += 1
        ref_id = None
        if isinstance(ref, dict):
            ref_id = ref.get("ref_id")
        if not isinstance(ref, dict):
            issues.append({"ref_index": idx, "reason": "source_ref_not_object"})
            continue

        raw_path = ref.get("path")
        raw_hash = ref.get("content_hash")

        if not isinstance(raw_path, str) or not raw_path.strip():
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "path_missing"})
            continue

        if not isinstance(raw_hash, str) or not raw_hash.strip():
            issues.append({"ref_index": idx, "ref_id": ref_id, "reason": "content_hash_missing"})
            continue

        resolved = resolve_repo_path(repo_root, raw_path)
        if not is_within(repo_root, resolved):
            issues.append(
                {
                    "ref_index": idx,
                    "ref_id": ref_id,
                    "reason": "path_outside_repo",
                    "path": raw_path,
                }
            )
            continue

        if not resolved.exists() or not resolved.is_file():
            issues.append(
                {
                    "ref_index": idx,
                    "ref_id": ref_id,
                    "reason": "path_unresolved",
                    "path": raw_path,
                }
            )
            continue

        declared = normalize_sha256(raw_hash)
        try:
            actual = file_sha256(resolved)
        except Exception as exc:
            issues.append(
                {
                    "ref_index": idx,
                    "ref_id": ref_id,
                    "reason": "hash_compute_failed",
                    "detail": str(exc),
                }
            )
            continue

        if declared != actual:
            issues.append(
                {
                    "ref_index": idx,
                    "ref_id": ref_id,
                    "reason": "content_hash_mismatch",
                    "path": raw_path,
                    "declared": declared,
                    "actual": actual,
                }
            )

    if issues:
        return False, "provenance_unresolved", {"checked": checked, "issues": issues}

    return True, None, {"checked": checked}


def gate_confidence(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    target = candidate.get("target")
    confidence = candidate.get("confidence")

    surface = target.get("surface") if isinstance(target, dict) else None
    score = confidence.get("score") if isinstance(confidence, dict) else None
    method = confidence.get("method") if isinstance(confidence, dict) else None

    threshold = CONFIDENCE_THRESHOLDS.get(str(surface))
    if threshold is None:
        return False, "gate_unavailable", {"error": "unknown_target_surface", "surface": surface}

    if not isinstance(score, (int, float)):
        return False, "confidence_below_threshold", {"error": "confidence_score_invalid", "threshold": threshold}

    numeric_score = float(score)
    if not (0.0 <= numeric_score <= 1.0):
        return False, "confidence_below_threshold", {"error": "confidence_score_out_of_range", "score": numeric_score}

    if not isinstance(method, str) or not method.strip():
        return False, "confidence_below_threshold", {"error": "confidence_method_missing", "threshold": threshold}

    if numeric_score < threshold:
        return (
            False,
            "confidence_below_threshold",
            {
                "surface": surface,
                "score": numeric_score,
                "threshold": threshold,
            },
        )

    return True, None, {"surface": surface, "score": numeric_score, "threshold": threshold}


def gate_review(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    target = candidate.get("target")
    review = candidate.get("review")

    surface = target.get("surface") if isinstance(target, dict) else None
    if not isinstance(review, dict):
        return False, "review_not_approved", {"error": "review_missing"}

    state = review.get("state")
    if state != "approved":
        return False, "review_not_approved", {"error": "review_state_not_approved", "state": state}

    allowed = REVIEW_ROLE_RULES.get(str(surface))
    role = review.get("reviewer_role")
    if not allowed or role not in allowed:
        return (
            False,
            "review_not_approved",
            {
                "error": "reviewer_role_not_allowed",
                "surface": surface,
                "role": role,
                "allowed_roles": sorted(list(allowed or [])),
            },
        )

    reviewer_id = review.get("reviewer_id")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        return False, "review_not_approved", {"error": "reviewer_id_missing"}
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        return False, "review_not_approved", {"error": "reviewed_at_missing"}

    return True, None, {"surface": surface, "reviewer_role": role}


def gate_leakage(candidate: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    safety = candidate.get("safety")
    if not isinstance(safety, dict):
        return False, "leakage_risk", {"error": "safety_missing"}

    classification = safety.get("classification")
    leakage_check = safety.get("leakage_check")
    redaction_applied = safety.get("redaction_applied")

    if classification == "secret":
        return False, "leakage_risk", {"error": "secret_not_promotable"}

    if leakage_check != "pass":
        return False, "leakage_risk", {"error": "leakage_check_failed", "leakage_check": leakage_check}

    if classification == "restricted" and redaction_applied is not True:
        return False, "leakage_risk", {"error": "restricted_requires_redaction"}

    return True, None, {"classification": classification}


def _surface_path_allowed(surface: str, rel_target: str) -> bool:
    if surface == "doctrine":
        return rel_target.startswith("docs/ops/")
    if surface == "memory":
        return rel_target.startswith("memory/")
    if surface == "playbook":
        return (
            rel_target.startswith("memory/skills/")
            or rel_target.startswith("docs/ops/playbooks/")
            or rel_target.startswith("ops/playbooks/")
        )
    return False


def _publish_trace_present(candidate: Dict[str, Any], promotion_id: Optional[str]) -> bool:
    if not promotion_id:
        return False
    refs = candidate.get("decision_refs")
    if not isinstance(refs, list):
        return False
    for ref in refs:
        if isinstance(ref, str) and promotion_id in ref:
            return True
    return False


def gate_publish(
    candidate: Dict[str, Any],
    repo_root: Path,
    *,
    promotion_id: Optional[str],
    publish_note_path: Optional[Path],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    target = candidate.get("target")
    if not isinstance(target, dict):
        return False, "publish_unready", {"error": "target_missing"}

    surface = target.get("surface")
    raw_target_path = target.get("target_path")
    if not isinstance(surface, str) or not isinstance(raw_target_path, str) or not raw_target_path.strip():
        return False, "publish_unready", {"error": "target_invalid"}

    resolved_target = resolve_repo_path(repo_root, raw_target_path)
    if not is_within(repo_root, resolved_target):
        return False, "publish_unready", {"error": "target_outside_repo", "target_path": raw_target_path}

    if not resolved_target.exists() or not resolved_target.is_file():
        return False, "publish_unready", {"error": "target_path_unresolved", "target_path": raw_target_path}

    rel_target = resolved_target.relative_to(repo_root).as_posix()
    if not _surface_path_allowed(surface, rel_target):
        return (
            False,
            "publish_unready",
            {
                "error": "target_surface_mismatch",
                "surface": surface,
                "target_path": rel_target,
            },
        )

    trace_ok = False
    trace_source = "decision_refs"

    if publish_note_path is not None:
        note_path = publish_note_path
        if not is_within(repo_root, note_path):
            return False, "publish_unready", {"error": "publish_note_outside_repo", "publish_note_path": str(note_path)}
        if not note_path.exists() or not note_path.is_file():
            return False, "publish_unready", {"error": "publish_note_missing", "publish_note_path": str(note_path)}
        try:
            note_text = note_path.read_text(encoding="utf-8")
        except Exception as exc:
            return False, "publish_unready", {"error": "publish_note_unreadable", "detail": str(exc)}
        trace_ok = bool(promotion_id and promotion_id in note_text)
        trace_source = "publish_note_path"
    else:
        trace_ok = _publish_trace_present(candidate, promotion_id)

    if not trace_ok:
        return (
            False,
            "publish_unready",
            {
                "error": "promotion_id_trace_missing",
                "trace_source": trace_source,
                "promotion_id": promotion_id,
            },
        )

    return True, None, {"surface": surface, "target_path": rel_target, "trace_source": trace_source}


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


def evaluate_candidate(
    *,
    candidate: Any,
    candidate_path: Path,
    repo_root: Path,
    schema_path: Path,
    publish_note_path: Optional[Path],
) -> Dict[str, Any]:
    decision_at = now_iso()

    promotion_id: Optional[str] = None
    if isinstance(candidate, dict):
        raw_pid = candidate.get("promotion_id")
        if isinstance(raw_pid, str):
            promotion_id = raw_pid

    gate_rows: List[Dict[str, Any]] = []
    blocked = False
    block_reason: Optional[str] = None
    block_gate: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(candidate, schema_path)),
        ("provenance", lambda: gate_provenance(candidate if isinstance(candidate, dict) else {}, repo_root)),
        ("confidence", lambda: gate_confidence(candidate if isinstance(candidate, dict) else {})),
        ("review", lambda: gate_review(candidate if isinstance(candidate, dict) else {})),
        ("leakage", lambda: gate_leakage(candidate if isinstance(candidate, dict) else {})),
        (
            "publish",
            lambda: gate_publish(
                candidate if isinstance(candidate, dict) else {},
                repo_root,
                promotion_id=promotion_id,
                publish_note_path=publish_note_path,
            ),
        ),
    ]

    for gate_name, gate_fn in gate_specs:
        if blocked:
            gate_rows.append(
                {
                    "gate": gate_name,
                    "status": "skipped",
                    "reason": "blocked_by_previous_gate",
                }
            )
            continue

        try:
            ok, reason, details = gate_fn()
        except Exception as exc:  # pragma: no cover - fail-closed fallback
            ok = False
            reason = "gate_unavailable"
            details = {"error": "gate_exception", "detail": str(exc)}

        if ok:
            gate_rows.append({"gate": gate_name, "status": "pass", "details": details})
            continue

        blocked = True
        block_reason = reason or "gate_unavailable"
        block_gate = gate_name
        gate_rows.append(
            {
                "gate": gate_name,
                "status": "fail",
                "reason": block_reason,
                "details": details,
            }
        )

    decision = "BLOCK" if blocked else "PASS"
    final_state = "BLOCKED" if blocked else "PROMOTED"

    try:
        candidate_sha = file_sha256(candidate_path)
    except Exception:
        candidate_sha = None

    return {
        "schema": "clawd.promotion_gate.decision.v1",
        "evaluated_at": decision_at,
        "decision": decision,
        "final_state": final_state,
        "block_gate": block_gate,
        "block_reason": block_reason,
        "promotion_id": promotion_id,
        "candidate": {
            "path": str(candidate_path),
            "sha256": candidate_sha,
        },
        "policy": {
            "confidence_thresholds": CONFIDENCE_THRESHOLDS,
            "review_role_rules": {k: sorted(list(v)) for k, v in REVIEW_ROLE_RULES.items()},
        },
        "gates": gate_rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic promotion gate runner (contract v1)")
    ap.add_argument("--candidate", required=True, help="Path to promotion candidate JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root for relative path resolution")
    ap.add_argument(
        "--schema-path",
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to promotion candidate JSON schema",
    )
    ap.add_argument(
        "--decision-log",
        default=str(DEFAULT_DECISION_LOG),
        help="Append-only decision log path (relative to repo root unless absolute)",
    )
    ap.add_argument("--no-decision-log", action="store_true", help="Disable append-only decision recording")
    ap.add_argument(
        "--publish-note-path",
        default=None,
        help="Optional publish-note artifact path; if provided it must contain promotion_id",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    publish_note_path: Optional[Path] = None
    if args.publish_note_path:
        candidate_note_path = Path(args.publish_note_path).expanduser()
        if not candidate_note_path.is_absolute():
            publish_note_path = (repo_root / candidate_note_path).resolve()
        else:
            publish_note_path = candidate_note_path.resolve()

    try:
        candidate_doc = load_json_file(candidate_path)
    except Exception as exc:
        candidate_doc = {
            "promotion_id": None,
        }
        result = {
            "schema": "clawd.promotion_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "final_state": "BLOCKED",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "promotion_id": None,
            "candidate": {
                "path": str(candidate_path),
                "sha256": None,
            },
            "policy": {
                "confidence_thresholds": CONFIDENCE_THRESHOLDS,
                "review_role_rules": {k: sorted(list(v)) for k, v in REVIEW_ROLE_RULES.items()},
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
                {"gate": "review", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "leakage", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "publish", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_candidate(
            candidate=candidate_doc,
            candidate_path=candidate_path,
            repo_root=repo_root,
            schema_path=schema_path,
            publish_note_path=publish_note_path,
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
