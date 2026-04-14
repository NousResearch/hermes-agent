#!/usr/bin/env python3
"""Markdown Conversion Quality Gate v1.

Deterministic, fail-closed quality gate for source->markdown conversion packets.
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
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "markdown_conversion_gate_packet.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_ingestion" / "markdown_conversion_gate_decisions.jsonl"

DEFAULT_POLICY: Dict[str, Any] = {
    "min_markdown_bytes": 256,
    "min_nonempty_lines": 8,
    "min_heading_count": 1,
    "require_fence_balance": True,
    "max_control_char_ratio": 0.01,
    "max_repeated_line_ratio": 0.35,
    "min_alpha_char_ratio": 0.45,
    "min_word_coverage_ratio": 0.55,
    "min_reference_words": 80,
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


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}", text or "")


def apply_policy_overrides(packet: Dict[str, Any]) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    raw = packet.get("quality_policy")
    if not isinstance(raw, dict):
        return policy
    for key, default in DEFAULT_POLICY.items():
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(default, bool):
            if isinstance(value, bool):
                policy[key] = value
            continue
        if isinstance(default, int):
            if isinstance(value, int):
                policy[key] = value
            continue
        if isinstance(default, float):
            if isinstance(value, (int, float)):
                policy[key] = float(value)
            continue
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


def validate_artifact(repo_root: Path, artifact: Dict[str, Any], label: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    raw_path = artifact.get("path")
    raw_hash = artifact.get("sha256")

    if not isinstance(raw_path, str) or not raw_path.strip():
        return False, "artifact_unresolved", {"error": "path_missing", "artifact": label}
    if not isinstance(raw_hash, str) or not raw_hash.strip():
        return False, "artifact_unresolved", {"error": "sha256_missing", "artifact": label, "path": raw_path}

    resolved = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, resolved):
        return False, "artifact_unresolved", {"error": "path_outside_repo", "artifact": label, "path": raw_path}
    if not resolved.exists() or not resolved.is_file():
        return False, "artifact_unresolved", {"error": "path_unresolved", "artifact": label, "path": raw_path}

    declared = normalize_sha256(raw_hash)
    try:
        actual = file_sha256(resolved)
    except Exception as exc:
        return False, "artifact_unresolved", {
            "error": "hash_compute_failed",
            "artifact": label,
            "path": raw_path,
            "detail": str(exc),
        }

    if declared != actual:
        return False, "artifact_unresolved", {
            "error": "sha256_mismatch",
            "artifact": label,
            "path": raw_path,
            "declared": declared,
            "actual": actual,
        }

    return True, None, {
        "artifact": label,
        "path": raw_path,
        "sha256": actual,
    }


def gate_artifact_integrity(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    source_artifact = packet.get("source_artifact") if isinstance(packet.get("source_artifact"), dict) else None
    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else None
    source_text_artifact = packet.get("source_text_artifact") if isinstance(packet.get("source_text_artifact"), dict) else None

    if source_artifact is None:
        return False, "artifact_unresolved", {"error": "source_artifact_missing"}
    if markdown_artifact is None:
        return False, "artifact_unresolved", {"error": "markdown_artifact_missing"}

    checks: List[Dict[str, Any]] = []
    for label, artifact in [("source_artifact", source_artifact), ("markdown_artifact", markdown_artifact)]:
        ok, reason, details = validate_artifact(repo_root, artifact, label)
        if not ok:
            return False, reason, details
        checks.append(details)

    if source_text_artifact is not None:
        ok, reason, details = validate_artifact(repo_root, source_text_artifact, "source_text_artifact")
        if not ok:
            return False, reason, details
        checks.append(details)

    return True, None, {"checks": checks}


def read_artifact_text(repo_root: Path, artifact: Dict[str, Any]) -> str:
    resolved = resolve_repo_path(repo_root, str(artifact.get("path") or ""))
    return resolved.read_text(encoding="utf-8", errors="replace")


def gate_markdown_structure(packet: Dict[str, Any], repo_root: Path, policy: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else None
    if markdown_artifact is None:
        return False, "markdown_structure_failed", {"error": "markdown_artifact_missing"}

    text = read_artifact_text(repo_root, markdown_artifact)
    encoded = text.encode("utf-8")
    byte_count = len(encoded)

    lines = text.splitlines()
    nonempty_lines = [line for line in lines if line.strip()]
    heading_count = len(re.findall(r"(?m)^\s{0,3}#{1,6}\s+\S", text))
    fence_markers = len(re.findall(r"(?m)^\s*(```|~~~)", text))
    fence_balanced = (fence_markers % 2) == 0

    metrics = {
        "markdown_bytes": byte_count,
        "nonempty_lines": len(nonempty_lines),
        "heading_count": heading_count,
        "fence_markers": fence_markers,
        "fence_balanced": fence_balanced,
    }

    violations: List[Dict[str, Any]] = []
    if byte_count < int(policy["min_markdown_bytes"]):
        violations.append({
            "metric": "markdown_bytes",
            "value": byte_count,
            "threshold": int(policy["min_markdown_bytes"]),
        })
    if len(nonempty_lines) < int(policy["min_nonempty_lines"]):
        violations.append({
            "metric": "nonempty_lines",
            "value": len(nonempty_lines),
            "threshold": int(policy["min_nonempty_lines"]),
        })
    if heading_count < int(policy["min_heading_count"]):
        violations.append({
            "metric": "heading_count",
            "value": heading_count,
            "threshold": int(policy["min_heading_count"]),
        })
    if bool(policy["require_fence_balance"]) and not fence_balanced:
        violations.append({"metric": "fence_balance", "value": False, "threshold": True})

    if violations:
        return False, "markdown_structure_failed", {"metrics": metrics, "violations": violations}
    return True, None, {"metrics": metrics}


def gate_markdown_noise(packet: Dict[str, Any], repo_root: Path, policy: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else None
    if markdown_artifact is None:
        return False, "markdown_noise_failed", {"error": "markdown_artifact_missing"}

    text = read_artifact_text(repo_root, markdown_artifact)
    if not text:
        return False, "markdown_noise_failed", {"error": "markdown_empty"}

    control_chars = sum(1 for c in text if (ord(c) < 32 and c not in "\n\t\r"))
    control_char_ratio = control_chars / max(1, len(text))

    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_ratio = 0.0
    if nonempty:
        duplicate_count = len(nonempty) - len(set(nonempty))
        repeated_line_ratio = duplicate_count / len(nonempty)

    non_ws = [c for c in text if not c.isspace()]
    alpha_chars = sum(1 for c in non_ws if c.isalpha())
    alpha_char_ratio = alpha_chars / max(1, len(non_ws))

    metrics = {
        "control_char_ratio": control_char_ratio,
        "repeated_line_ratio": repeated_line_ratio,
        "alpha_char_ratio": alpha_char_ratio,
    }

    violations: List[Dict[str, Any]] = []
    if control_char_ratio > float(policy["max_control_char_ratio"]):
        violations.append({
            "metric": "control_char_ratio",
            "value": control_char_ratio,
            "threshold": float(policy["max_control_char_ratio"]),
        })
    if repeated_line_ratio > float(policy["max_repeated_line_ratio"]):
        violations.append({
            "metric": "repeated_line_ratio",
            "value": repeated_line_ratio,
            "threshold": float(policy["max_repeated_line_ratio"]),
        })
    if alpha_char_ratio < float(policy["min_alpha_char_ratio"]):
        violations.append({
            "metric": "alpha_char_ratio",
            "value": alpha_char_ratio,
            "threshold": float(policy["min_alpha_char_ratio"]),
        })

    if violations:
        return False, "markdown_noise_failed", {"metrics": metrics, "violations": violations}
    return True, None, {"metrics": metrics}


def gate_reference_coverage(packet: Dict[str, Any], repo_root: Path, policy: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    source_artifact = packet.get("source_artifact") if isinstance(packet.get("source_artifact"), dict) else {}
    source_kind = str(source_artifact.get("kind") or "")

    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else None
    if markdown_artifact is None:
        return False, "reference_coverage_failed", {"error": "markdown_artifact_missing"}

    source_text_artifact = packet.get("source_text_artifact") if isinstance(packet.get("source_text_artifact"), dict) else None
    if source_text_artifact is None:
        if source_kind == "markdown":
            return True, None, {"mode": "markdown_source", "coverage_ratio": 1.0}
        return False, "reference_coverage_missing", {
            "error": "source_text_artifact_missing",
            "hint": "Provide deterministic source_text_artifact for non-markdown source kinds",
        }

    md_text = read_artifact_text(repo_root, markdown_artifact)
    src_text = read_artifact_text(repo_root, source_text_artifact)

    md_words = tokenize_words(md_text)
    src_words = tokenize_words(src_text)

    src_word_count = len(src_words)
    md_word_count = len(md_words)

    if src_word_count < int(policy["min_reference_words"]):
        return False, "reference_coverage_failed", {
            "error": "reference_text_too_small",
            "source_word_count": src_word_count,
            "min_reference_words": int(policy["min_reference_words"]),
        }

    coverage_ratio = md_word_count / max(1, src_word_count)
    min_ratio = float(policy["min_word_coverage_ratio"])
    if coverage_ratio < min_ratio:
        return False, "reference_coverage_failed", {
            "metrics": {
                "markdown_word_count": md_word_count,
                "source_word_count": src_word_count,
                "coverage_ratio": coverage_ratio,
            },
            "violations": [{"metric": "coverage_ratio", "value": coverage_ratio, "threshold": min_ratio}],
        }

    return True, None, {
        "metrics": {
            "markdown_word_count": md_word_count,
            "source_word_count": src_word_count,
            "coverage_ratio": coverage_ratio,
        }
    }


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

    conversion_id = packet_dict.get("conversion_id") if isinstance(packet_dict.get("conversion_id"), str) else None

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(packet, schema_path)),
        ("artifact_integrity", lambda: gate_artifact_integrity(packet_dict, repo_root)),
        ("markdown_structure", lambda: gate_markdown_structure(packet_dict, repo_root, policy)),
        ("markdown_noise", lambda: gate_markdown_noise(packet_dict, repo_root, policy)),
        ("reference_coverage", lambda: gate_reference_coverage(packet_dict, repo_root, policy)),
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
        "schema": "clawd.markdown_conversion_gate.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "conversion_id": conversion_id,
        "packet": {"path": str(packet_path), "sha256": packet_sha},
        "policy": policy,
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Markdown conversion quality gate runner (v1)")
    ap.add_argument("--packet", required=True, help="Markdown conversion gate packet JSON")
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
            "schema": "clawd.markdown_conversion_gate.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "conversion_id": None,
            "packet": {"path": str(packet_path), "sha256": None},
            "policy": dict(DEFAULT_POLICY),
            "gates": [
                {
                    "gate": "schema",
                    "status": "fail",
                    "reason": "schema_invalid",
                    "details": {"error": "packet_json_unreadable", "detail": str(exc)},
                },
                {"gate": "artifact_integrity", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "markdown_structure", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "markdown_noise", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "reference_coverage", "status": "skipped", "reason": "blocked_by_previous_gate"},
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
