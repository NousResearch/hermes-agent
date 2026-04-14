#!/usr/bin/env python3
"""Source Material Classification Layer v1.

Deterministic, fail-closed classifier for material packets used by production ingestion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "source_material_classification_packet.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_ingestion" / "source_material_classification_decisions.jsonl"

DEFAULT_POLICY: Dict[str, Any] = {
    "allow_classes": [
        "architecture_spec",
        "runbook",
        "policy_doctrine",
        "research_report",
        "runtime_evidence",
        "source_document",
    ],
    "block_classes": ["unknown"],
    "min_confidence": 0.60,
    "allow_generated": False,
}

CLASS_NAMES = {
    "architecture_spec",
    "runbook",
    "policy_doctrine",
    "research_report",
    "runtime_evidence",
    "source_document",
    "unknown",
}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def normalize_sha256(raw: str) -> str:
    text = (raw or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


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


def json_ptr(parts: Any) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(p) for p in seq)


def apply_policy_overrides(packet: Dict[str, Any]) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    raw = packet.get("classification_policy")
    if not isinstance(raw, dict):
        return policy

    allow_classes = raw.get("allow_classes")
    if isinstance(allow_classes, list):
        cleaned = sorted({str(x) for x in allow_classes if str(x) in CLASS_NAMES and str(x) != "unknown"})
        if cleaned:
            policy["allow_classes"] = cleaned

    block_classes = raw.get("block_classes")
    if isinstance(block_classes, list):
        cleaned = sorted({str(x) for x in block_classes if str(x) in CLASS_NAMES})
        policy["block_classes"] = cleaned

    min_conf = raw.get("min_confidence")
    if isinstance(min_conf, (int, float)):
        policy["min_confidence"] = float(min_conf)

    allow_generated = raw.get("allow_generated")
    if isinstance(allow_generated, bool):
        policy["allow_generated"] = allow_generated

    return policy


def gate_schema(packet: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
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


def validate_material_artifact(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    artifact = packet.get("material_artifact") if isinstance(packet.get("material_artifact"), dict) else None
    if artifact is None:
        return False, "material_unresolved", {"error": "material_artifact_missing"}

    raw_path = artifact.get("path")
    raw_hash = artifact.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False, "material_unresolved", {"error": "path_missing"}
    if not isinstance(raw_hash, str) or not raw_hash.strip():
        return False, "material_unresolved", {"error": "sha256_missing", "path": raw_path}

    resolved = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, resolved):
        return False, "material_unresolved", {"error": "path_outside_repo", "path": raw_path}
    if not resolved.exists() or not resolved.is_file():
        return False, "material_unresolved", {"error": "path_unresolved", "path": raw_path}

    declared = normalize_sha256(raw_hash)
    try:
        actual = file_sha256(resolved)
    except Exception as exc:
        return False, "material_unresolved", {"error": "hash_compute_failed", "detail": str(exc), "path": raw_path}

    if declared != actual:
        return False, "material_unresolved", {
            "error": "sha256_mismatch",
            "path": raw_path,
            "declared": declared,
            "actual": actual,
        }

    return True, None, {"path": raw_path, "sha256": actual}


def _add_score(scores: Dict[str, float], cls: str, weight: float, reason: str, evidence: List[str]) -> None:
    if cls not in scores:
        scores[cls] = 0.0
    scores[cls] += weight
    evidence.append(reason)


def classify_material(packet: Dict[str, Any], repo_root: Path) -> Tuple[str, float, List[str], Dict[str, Any]]:
    artifact = packet.get("material_artifact") if isinstance(packet.get("material_artifact"), dict) else {}
    hints = packet.get("hints") if isinstance(packet.get("hints"), dict) else {}

    raw_path = str(artifact.get("path") or "")
    resolved = resolve_repo_path(repo_root, raw_path)
    suffix = resolved.suffix.lower()
    path_lower = raw_path.lower()

    evidence: List[str] = []
    scores: Dict[str, float] = {
        "architecture_spec": 0.0,
        "runbook": 0.0,
        "policy_doctrine": 0.0,
        "research_report": 0.0,
        "runtime_evidence": 0.0,
        "source_document": 0.0,
    }

    # Path/extension heuristics
    if "/state/" in path_lower or suffix in {".json", ".jsonl", ".log"}:
        _add_score(scores, "runtime_evidence", 0.86, "state_path_or_runtime_extension", evidence)
    if "runbook" in path_lower or "playbook" in path_lower:
        _add_score(scores, "runbook", 0.82, "runbook_keyword_in_path", evidence)
    if any(tok in path_lower for tok in ["doctrine", "policy", "protocol"]):
        _add_score(scores, "policy_doctrine", 0.82, "policy_keyword_in_path", evidence)
    if "/reports/" in path_lower or re.search(r"wave\d+", path_lower):
        _add_score(scores, "research_report", 0.72, "reports_or_wave_path", evidence)
    if any(tok in path_lower for tok in ["architecture", "design", "spec"]):
        _add_score(scores, "architecture_spec", 0.70, "architecture_keyword_in_path", evidence)
    if suffix in {".pdf", ".epub", ".doc", ".docx"}:
        _add_score(scores, "source_document", 0.70, "longform_source_extension", evidence)

    # Declared hint nudges
    declared = str(hints.get("declared_type") or "").strip().lower()
    hint_map = {
        "architecture_spec": "architecture_spec",
        "runbook": "runbook",
        "policy_doctrine": "policy_doctrine",
        "research_report": "research_report",
        "runtime_evidence": "runtime_evidence",
        "source_document": "source_document",
    }
    if declared in hint_map:
        _add_score(scores, hint_map[declared], 0.35, f"declared_type_hint:{declared}", evidence)

    # Content heuristics (best-effort; still deterministic)
    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception:
        text = ""

    probe = "\n".join(text.splitlines()[:160]).lower()
    if probe:
        if "runbook" in probe or "operational steps" in probe:
            _add_score(scores, "runbook", 0.40, "runbook_keywords_in_content", evidence)
        if any(tok in probe for tok in ["doctrine", "governance", "policy", "protocol"]):
            _add_score(scores, "policy_doctrine", 0.40, "policy_keywords_in_content", evidence)
        if "what landed" in probe or "verification" in probe or "checkpoint" in probe:
            _add_score(scores, "research_report", 0.40, "reporting_keywords_in_content", evidence)
        if "architecture" in probe or "system design" in probe or "contract" in probe:
            _add_score(scores, "architecture_spec", 0.32, "architecture_keywords_in_content", evidence)
        if "incident" in probe or "error" in probe or "trace" in probe:
            _add_score(scores, "runtime_evidence", 0.25, "runtime_keywords_in_content", evidence)

    best_class = max(scores, key=lambda k: scores[k])
    best_score = float(scores.get(best_class, 0.0))
    confidence = min(0.99, max(0.0, best_score))

    if confidence < 0.55:
        return "unknown", confidence, evidence, {"scores": scores, "path": raw_path, "suffix": suffix}

    return best_class, confidence, evidence, {"scores": scores, "path": raw_path, "suffix": suffix}


def gate_policy_enforcement(
    packet: Dict[str, Any],
    classification: Dict[str, Any],
    policy: Dict[str, Any],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    label = str(classification.get("label") or "unknown")
    confidence = float(classification.get("confidence") or 0.0)

    if label in set(policy.get("block_classes") or []):
        return False, "classification_blocked", {"error": "class_blocked", "label": label}

    allow_classes = set(policy.get("allow_classes") or [])
    if allow_classes and label not in allow_classes:
        return False, "classification_blocked", {
            "error": "class_not_allowed",
            "label": label,
            "allow_classes": sorted(allow_classes),
        }

    min_conf = float(policy.get("min_confidence") or 0.0)
    if confidence < min_conf:
        return False, "classification_blocked", {
            "error": "confidence_below_min",
            "label": label,
            "confidence": confidence,
            "min_confidence": min_conf,
        }

    hints = packet.get("hints") if isinstance(packet.get("hints"), dict) else {}
    origin = str(hints.get("origin") or "").strip().lower()
    if origin == "generated" and not bool(policy.get("allow_generated")):
        return False, "classification_blocked", {
            "error": "generated_material_disallowed",
            "origin": origin,
        }

    return True, None, {"label": label, "confidence": confidence}


def append_decision_record(decision_log_path: Optional[Path], repo_root: Path, row: Dict[str, Any]) -> Dict[str, Any]:
    if decision_log_path is None:
        return {"enabled": False, "appended": False, "reason": "disabled"}

    path = decision_log_path if decision_log_path.is_absolute() else (repo_root / decision_log_path).resolve()
    if not is_within(repo_root, path):
        return {"enabled": True, "appended": False, "reason": "unsafe_path", "path": str(path)}

    try:
        if path.exists() and not path.is_file():
            return {"enabled": True, "appended": False, "reason": "path_not_file", "path": str(path)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(row) + "\n")
        return {"enabled": True, "appended": True, "path": str(path)}
    except Exception as exc:
        return {
            "enabled": True,
            "appended": False,
            "reason": "append_failed",
            "path": str(path),
            "error": str(exc),
        }


def evaluate_packet(packet: Any, packet_path: Path, repo_root: Path, schema_path: Path) -> Dict[str, Any]:
    packet_dict = packet if isinstance(packet, dict) else {}
    policy = apply_policy_overrides(packet_dict)
    classification_id = packet_dict.get("classification_id") if isinstance(packet_dict.get("classification_id"), str) else None

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    classification_payload: Dict[str, Any] = {
        "label": "unknown",
        "confidence": 0.0,
        "evidence": [],
        "diagnostics": {},
    }

    gate_specs = [
        ("schema", lambda: gate_schema(packet, schema_path)),
        ("artifact_integrity", lambda: validate_material_artifact(packet_dict, repo_root)),
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

    if not blocked:
        label, confidence, evidence, diagnostics = classify_material(packet_dict, repo_root)
        classification_payload = {
            "label": label,
            "confidence": confidence,
            "evidence": evidence,
            "diagnostics": diagnostics,
        }
        gates.append({"gate": "rule_classification", "status": "pass", "details": classification_payload})

        ok, reason, details = gate_policy_enforcement(packet_dict, classification_payload, policy)
        if ok:
            gates.append({"gate": "policy_enforcement", "status": "pass", "details": details})
        else:
            blocked = True
            block_gate = "policy_enforcement"
            block_reason = reason or "classification_blocked"
            gates.append({"gate": "policy_enforcement", "status": "fail", "reason": block_reason, "details": details})
    else:
        gates.append({"gate": "rule_classification", "status": "skipped", "reason": "blocked_by_previous_gate"})
        gates.append({"gate": "policy_enforcement", "status": "skipped", "reason": "blocked_by_previous_gate"})

    try:
        packet_sha = file_sha256(packet_path)
    except Exception:
        packet_sha = None

    return {
        "schema": "clawd.source_material_classification.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "classification_id": classification_id,
        "packet": {"path": str(packet_path), "sha256": packet_sha},
        "classification": classification_payload,
        "policy": policy,
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Source material classification runner (v1)")
    ap.add_argument("--packet", required=True, help="Classification packet JSON")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Packet schema path")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Append-only decision log path")
    ap.add_argument("--no-decision-log", action="store_true", help="Disable decision log append")
    ap.add_argument("--json", action="store_true", help="Pretty JSON output")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    packet_path = Path(args.packet).expanduser().resolve()

    try:
        packet = load_json_file(packet_path)
    except Exception as exc:
        result = {
            "schema": "clawd.source_material_classification.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "classification_id": None,
            "packet": {"path": str(packet_path), "sha256": None},
            "classification": {"label": "unknown", "confidence": 0.0, "evidence": [], "diagnostics": {}},
            "policy": dict(DEFAULT_POLICY),
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {"error": "packet_json_unreadable", "detail": str(exc)},
                },
                {"gate": "artifact_integrity", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "rule_classification", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "policy_enforcement", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate_packet(packet, packet_path, repo_root, schema_path)

    decision_log_path: Optional[Path] = None if args.no_decision_log else Path(args.decision_log).expanduser()
    result["decision_record"] = append_decision_record(decision_log_path, repo_root, result)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return 0 if result.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
