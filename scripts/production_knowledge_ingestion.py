#!/usr/bin/env python3
"""Production Knowledge Ingestion Layer v1.

Evaluates and (optionally) applies deterministic production-ingestion packets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
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
DEFAULT_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "production_knowledge_ingestion_packet.schema.json"
DEFAULT_DECISION_LOG = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_ingestion" / "production_knowledge_ingestion_decisions.jsonl"
DEFAULT_LEDGER_PATH = DEFAULT_REPO_ROOT / "state" / "continuity" / "knowledge_ingestion" / "production_ingestion_ledger.jsonl"
DEFAULT_LATEST_SNAPSHOT = DEFAULT_REPO_ROOT / "state" / "continuity" / "latest" / "production_knowledge_ingestion_latest.json"
DEFAULT_ALLOWED_ROOT = Path("memory/knowledge_ingestion/production")
DEFAULT_DESTINATION_PROFILES = Path("docs/ops/document_ingestion_destination_profiles_v1.json")
DEFAULT_MULTI_HOST_FIXTURE_SCHEMA_PATH = Path("docs/ops/schemas/b3_multi_host_fault_injection_fixture.schema.json")
DEFAULT_MULTI_HOST_RUNTIME_LATEST = Path("state/continuity/latest/b3_multi_host_fault_injection_latest.json")

DEFAULT_POLICY: Dict[str, Any] = {
    "require_markdown_gate_pass": True,
    "require_classification_pass": True,
    "allowed_classes": [
        "architecture_spec",
        "runbook",
        "policy_doctrine",
        "research_report",
        "source_document",
    ],
    "allow_overwrite": False,
}

WRAPPER_REQUIRED_SCHEMA = "clawd.production_knowledge_ingestion.wrapper_contract.v1"
DEFAULT_ALLOWED_MUTATION_CALLSITES = {
    "continuity.sh:knowledge-ingest",
}
DEFAULT_MUTATING_COMMANDS = {
    "ingest",
    "apply",
    "ingest-multi-host",
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


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def apply_policy_overrides(packet: Dict[str, Any], cli_allow_overwrite: bool) -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    raw = packet.get("ingestion_policy")
    if isinstance(raw, dict):
        for key in ["require_markdown_gate_pass", "require_classification_pass", "allow_overwrite"]:
            value = raw.get(key)
            if isinstance(value, bool):
                policy[key] = value
        classes = raw.get("allowed_classes")
        if isinstance(classes, list):
            cleaned = sorted({str(x) for x in classes if str(x)})
            if cleaned:
                policy["allowed_classes"] = cleaned

    if cli_allow_overwrite:
        policy["allow_overwrite"] = True

    return policy


def resolve_allowed_mutation_callsites() -> List[str]:
    allowed = set(DEFAULT_ALLOWED_MUTATION_CALLSITES)
    raw = str(os.environ.get("OPENCLAW_PRODUCTION_KNOWLEDGE_INGESTION_ALLOWED_CALLSITES") or "").strip()
    if raw:
        for token in raw.split(","):
            value = token.strip()
            if value:
                allowed.add(value)
    return sorted(allowed)


def enforce_wrapper_only_contract(command: str) -> Optional[Dict[str, Any]]:
    if command not in DEFAULT_MUTATING_COMMANDS:
        return None

    internal_mutation = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION") or "").strip()
    callsite = str(os.environ.get("OPENCLAW_INTERNAL_MUTATION_CALLSITE") or "").strip()
    allowed_callsites = resolve_allowed_mutation_callsites()

    if internal_mutation != "1":
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_env_missing",
            "required_env": ["OPENCLAW_INTERNAL_MUTATION=1", "OPENCLAW_INTERNAL_MUTATION_CALLSITE=<allowlisted>"],
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest <ingest|apply|ingest-multi-host> ... --json",
        }

    if not callsite:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_missing",
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest <ingest|apply|ingest-multi-host> ... --json",
        }

    if callsite not in allowed_callsites:
        return {
            "ok": False,
            "schema": WRAPPER_REQUIRED_SCHEMA,
            "error": "wrapper_only_entrypoint",
            "detail": "internal_mutation_callsite_not_allowlisted",
            "callsite": callsite,
            "allowed_callsites": allowed_callsites,
            "hint": "bash ops/openclaw/continuity.sh --action-token <...> knowledge-ingest <ingest|apply|ingest-multi-host> ... --json",
        }

    return None


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

    return True, None, {"artifact": label, "path": raw_path, "sha256": actual}


def gate_artifact_integrity(packet: Dict[str, Any], repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    material_artifact = packet.get("material_artifact") if isinstance(packet.get("material_artifact"), dict) else None
    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else None

    if material_artifact is None:
        return False, "artifact_unresolved", {"error": "material_artifact_missing"}
    if markdown_artifact is None:
        return False, "artifact_unresolved", {"error": "markdown_artifact_missing"}

    checks: List[Dict[str, Any]] = []
    for label, artifact in [("material_artifact", material_artifact), ("markdown_artifact", markdown_artifact)]:
        ok, reason, details = validate_artifact(repo_root, artifact, label)
        if not ok:
            return False, reason, details
        checks.append(details)

    return True, None, {"checks": checks}


def load_decision_file(repo_root: Path, ref_obj: Dict[str, Any], label: str) -> Tuple[bool, Optional[str], Dict[str, Any], Dict[str, Any]]:
    raw_path = ref_obj.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False, "decision_unresolved", {"error": "decision_path_missing", "decision": label}, {}

    resolved = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, resolved):
        return False, "decision_unresolved", {"error": "decision_path_outside_repo", "decision": label, "path": raw_path}, {}
    if not resolved.exists() or not resolved.is_file():
        return False, "decision_unresolved", {"error": "decision_path_unresolved", "decision": label, "path": raw_path}, {}

    try:
        payload = load_json_file(resolved)
    except Exception as exc:
        return False, "decision_unresolved", {
            "error": "decision_json_unreadable",
            "decision": label,
            "path": raw_path,
            "detail": str(exc),
        }, {}

    if not isinstance(payload, dict):
        return False, "decision_unresolved", {
            "error": "decision_not_object",
            "decision": label,
            "path": raw_path,
        }, {}

    return True, None, {"decision": label, "path": raw_path}, payload


def extract_markdown_sha_from_gate(decision: Dict[str, Any]) -> Optional[str]:
    gates = decision.get("gates") if isinstance(decision.get("gates"), list) else []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("gate") != "artifact_integrity":
            continue
        details = gate.get("details") if isinstance(gate.get("details"), dict) else {}
        checks = details.get("checks") if isinstance(details.get("checks"), list) else []
        for row in checks:
            if not isinstance(row, dict):
                continue
            if row.get("artifact") == "markdown_artifact":
                sha = row.get("sha256")
                if isinstance(sha, str) and sha:
                    return normalize_sha256(sha)
    return None


def extract_material_sha_from_classification(decision: Dict[str, Any]) -> Optional[str]:
    gates = decision.get("gates") if isinstance(decision.get("gates"), list) else []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("gate") != "artifact_integrity":
            continue
        details = gate.get("details") if isinstance(gate.get("details"), dict) else {}
        sha = details.get("sha256")
        if isinstance(sha, str) and sha:
            return normalize_sha256(sha)
    return None


def gate_markdown_decision(packet: Dict[str, Any], repo_root: Path, policy: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    ref = packet.get("markdown_gate_decision") if isinstance(packet.get("markdown_gate_decision"), dict) else None
    if ref is None:
        return False, "upstream_gate_blocked", {"error": "markdown_gate_decision_missing"}

    ok, reason, details, payload = load_decision_file(repo_root, ref, "markdown_gate")
    if not ok:
        return False, reason, details

    schema = str(payload.get("schema") or "")
    decision = str(payload.get("decision") or "")
    if schema != "clawd.markdown_conversion_gate.decision.v1":
        return False, "upstream_gate_blocked", {
            "error": "markdown_decision_schema_invalid",
            "observed_schema": schema,
        }

    require_pass = bool(policy.get("require_markdown_gate_pass"))
    if require_pass and decision != "PASS":
        return False, "upstream_gate_blocked", {
            "error": "markdown_gate_not_pass",
            "decision": decision,
            "block_reason": payload.get("block_reason"),
        }

    expected_markdown_sha = normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or ""))
    observed_markdown_sha = extract_markdown_sha_from_gate(payload)
    if not observed_markdown_sha:
        return False, "upstream_gate_blocked", {"error": "markdown_gate_sha_unavailable"}
    if observed_markdown_sha != expected_markdown_sha:
        return False, "upstream_gate_blocked", {
            "error": "markdown_gate_sha_mismatch",
            "expected": expected_markdown_sha,
            "observed": observed_markdown_sha,
        }

    return True, None, {
        "decision_path": details.get("path"),
        "decision": decision,
        "markdown_sha256": observed_markdown_sha,
    }


def gate_classification_decision(packet: Dict[str, Any], repo_root: Path, policy: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    ref = packet.get("classification_decision") if isinstance(packet.get("classification_decision"), dict) else None
    if ref is None:
        return False, "classification_blocked", {"error": "classification_decision_missing"}

    ok, reason, details, payload = load_decision_file(repo_root, ref, "classification")
    if not ok:
        return False, reason, details

    schema = str(payload.get("schema") or "")
    decision = str(payload.get("decision") or "")
    if schema != "clawd.source_material_classification.decision.v1":
        return False, "classification_blocked", {
            "error": "classification_decision_schema_invalid",
            "observed_schema": schema,
        }

    require_pass = bool(policy.get("require_classification_pass"))
    if require_pass and decision != "PASS":
        return False, "classification_blocked", {
            "error": "classification_not_pass",
            "decision": decision,
            "block_reason": payload.get("block_reason"),
        }

    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}
    label = str(classification.get("label") or "unknown")
    allowed = set(policy.get("allowed_classes") or [])
    if allowed and label not in allowed:
        return False, "classification_blocked", {
            "error": "classification_label_not_allowed",
            "label": label,
            "allowed_classes": sorted(allowed),
        }

    expected_material_sha = normalize_sha256(str((packet.get("material_artifact") or {}).get("sha256") or ""))
    observed_material_sha = extract_material_sha_from_classification(payload)
    if not observed_material_sha:
        return False, "classification_blocked", {"error": "classification_material_sha_unavailable"}
    if observed_material_sha != expected_material_sha:
        return False, "classification_blocked", {
            "error": "classification_material_sha_mismatch",
            "expected": expected_material_sha,
            "observed": observed_material_sha,
        }

    return True, None, {
        "decision_path": details.get("path"),
        "decision": decision,
        "classification_label": label,
        "material_sha256": observed_material_sha,
    }


def load_destination_profiles(repo_root: Path, destination_profiles_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    profile_path = destination_profiles_path if destination_profiles_path.is_absolute() else (repo_root / destination_profiles_path).resolve()
    if not is_within(repo_root, profile_path):
        return False, "destination_policy_failed", {"error": "destination_profiles_path_outside_repo", "path": str(profile_path)}
    if not profile_path.exists() or not profile_path.is_file():
        return False, "destination_policy_failed", {"error": "destination_profiles_path_unresolved", "path": str(profile_path)}

    try:
        payload = load_json_file(profile_path)
    except Exception as exc:
        return False, "destination_policy_failed", {
            "error": "destination_profiles_unreadable",
            "path": str(profile_path),
            "detail": str(exc),
        }

    if not isinstance(payload, dict):
        return False, "destination_policy_failed", {"error": "destination_profiles_not_object", "path": str(profile_path)}

    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else None
    if profiles is None:
        return False, "destination_policy_failed", {"error": "destination_profiles_missing", "path": str(profile_path)}

    return True, None, {
        "path": str(profile_path.relative_to(repo_root)),
        "schema_version": payload.get("schema_version"),
        "default_profile": payload.get("default_profile"),
        "profiles": profiles,
    }


def gate_destination_policy(
    packet: Dict[str, Any],
    repo_root: Path,
    policy: Dict[str, Any],
    allowed_root: Path,
    destination_profiles_path: Path,
    ledger_path: Path,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    destination = packet.get("destination") if isinstance(packet.get("destination"), dict) else None
    if destination is None:
        return False, "destination_policy_failed", {"error": "destination_missing"}

    raw_path = destination.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False, "destination_policy_failed", {"error": "destination_path_missing"}

    resolved = resolve_repo_path(repo_root, raw_path)
    if not is_within(repo_root, resolved):
        return False, "destination_policy_failed", {"error": "destination_outside_repo", "path": raw_path}

    allowed_abs = allowed_root if allowed_root.is_absolute() else (repo_root / allowed_root).resolve()
    destination_profile = str(destination.get("profile") or "").strip()
    profile_details: Dict[str, Any] = {}

    if destination_profile:
        ok, reason, destination_profiles = load_destination_profiles(repo_root, destination_profiles_path)
        if not ok:
            return False, reason, destination_profiles

        profiles = destination_profiles.get("profiles") if isinstance(destination_profiles.get("profiles"), dict) else {}
        selected = profiles.get(destination_profile) if isinstance(profiles.get(destination_profile), dict) else None
        if selected is None:
            return False, "destination_policy_failed", {
                "error": "destination_profile_unknown",
                "profile": destination_profile,
                "profiles_path": destination_profiles.get("path"),
            }

        root_raw = selected.get("root")
        if not isinstance(root_raw, str) or not root_raw.strip():
            return False, "destination_policy_failed", {
                "error": "destination_profile_root_missing",
                "profile": destination_profile,
                "profiles_path": destination_profiles.get("path"),
            }

        allowed_abs = resolve_repo_path(repo_root, root_raw)
        if not is_within(repo_root, allowed_abs):
            return False, "destination_policy_failed", {
                "error": "destination_profile_root_outside_repo",
                "profile": destination_profile,
                "root": root_raw,
            }
        profile_allowed_classes = selected.get("allowed_classes") if isinstance(selected.get("allowed_classes"), list) else None

        if profile_allowed_classes:
            classification_ref = packet.get("classification_decision") if isinstance(packet.get("classification_decision"), dict) else None
            if classification_ref is None:
                return False, "destination_policy_failed", {
                    "error": "destination_profile_requires_classification",
                    "profile": destination_profile,
                }
            ok_cls, _, _, classification_payload = load_decision_file(repo_root, classification_ref, "classification")
            if not ok_cls:
                return False, "destination_policy_failed", {
                    "error": "destination_profile_classification_unresolved",
                    "profile": destination_profile,
                }

            classification = classification_payload.get("classification") if isinstance(classification_payload.get("classification"), dict) else {}
            label = str(classification.get("label") or "unknown")
            allowed_labels = sorted({str(x) for x in profile_allowed_classes if str(x)})
            if allowed_labels and label not in set(allowed_labels):
                return False, "destination_policy_failed", {
                    "error": "destination_profile_classification_not_allowed",
                    "profile": destination_profile,
                    "label": label,
                    "allowed_classes": allowed_labels,
                }
            profile_details["classification_label"] = label
            profile_details["profile_allowed_classes"] = allowed_labels

        profile_details["profile"] = destination_profile
        profile_details["profiles_path"] = destination_profiles.get("path")

    allowed_root_display = str(allowed_abs.relative_to(repo_root)) if is_within(repo_root, allowed_abs) else str(allowed_abs)

    if not is_within(allowed_abs, resolved):
        return False, "destination_policy_failed", {
            "error": "destination_outside_allowed_root",
            "path": raw_path,
            "allowed_root": allowed_root_display,
            **profile_details,
        }

    if resolved.exists() and not bool(policy.get("allow_overwrite")):
        destination_sha: Optional[str]
        try:
            destination_sha = file_sha256(resolved)
        except Exception:
            destination_sha = None

        expected_markdown_sha = normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or ""))
        expected_material_sha = normalize_sha256(str((packet.get("material_artifact") or {}).get("sha256") or ""))
        expected_ingestion_id = str(packet.get("ingestion_id") or "")

        ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
        replay_safe = False
        if expected_ingestion_id and destination_sha and expected_markdown_sha and destination_sha == expected_markdown_sha:
            if is_within(repo_root, ledger_abs):
                ok_ledger, _, rows = load_ledger(ledger_abs)
                if ok_ledger:
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("ingestion_id") or "") != expected_ingestion_id:
                            continue
                        if str(row.get("destination_path") or "") != raw_path:
                            continue
                        if normalize_sha256(str(row.get("material_sha256") or "")) != expected_material_sha:
                            continue
                        if normalize_sha256(str(row.get("markdown_sha256") or "")) != expected_markdown_sha:
                            continue
                        replay_safe = True
                        break

        if replay_safe:
            return True, None, {
                "path": raw_path,
                "resolved_path": str(resolved),
                "allowed_root": allowed_root_display,
                "allow_overwrite": False,
                "replay_idempotent_noop": True,
                "destination_sha256": destination_sha,
                **profile_details,
            }

        return False, "destination_policy_failed", {
            "error": "destination_exists",
            "path": raw_path,
            "allow_overwrite": False,
            "destination_sha256": destination_sha,
            "replay_idempotent_noop": False,
            **profile_details,
        }

    return True, None, {
        "path": raw_path,
        "resolved_path": str(resolved),
        "allowed_root": allowed_root_display,
        "allow_overwrite": bool(policy.get("allow_overwrite")),
        **profile_details,
    }


def load_ledger(ledger_path: Path) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
    if not ledger_path.exists():
        return True, None, []
    if not ledger_path.is_file():
        return False, "ledger_unavailable", []

    rows: List[Dict[str, Any]] = []
    try:
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return False, "ledger_unreadable", []
    return True, None, rows


def gate_duplicate_guard(packet: Dict[str, Any], ledger_path: Path, repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    ingestion_id = str(packet.get("ingestion_id") or "")
    destination = packet.get("destination") if isinstance(packet.get("destination"), dict) else {}
    destination_path = str(destination.get("path") or "")

    material_sha = normalize_sha256(str((packet.get("material_artifact") or {}).get("sha256") or ""))
    markdown_sha = normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or ""))

    ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
    if not is_within(repo_root, ledger_abs):
        return False, "duplicate_guard_failed", {"error": "unsafe_ledger_path", "ledger_path": str(ledger_abs)}

    ok, reason, rows = load_ledger(ledger_abs)
    if not ok:
        return False, "duplicate_guard_failed", {"error": reason, "ledger_path": str(ledger_abs)}

    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("ingestion_id") or "") != ingestion_id:
            continue

        same_tuple = (
            normalize_sha256(str(row.get("material_sha256") or "")) == material_sha
            and normalize_sha256(str(row.get("markdown_sha256") or "")) == markdown_sha
            and str(row.get("destination_path") or "") == destination_path
        )

        if same_tuple:
            return True, None, {
                "ledger_path": str(ledger_abs),
                "existing_rows": len(rows),
                "replay_idempotent_noop": True,
                "replay_match": "ingestion_id_exact",
            }

        return False, "duplicate_guard_failed", {
            "error": "ingestion_id_already_recorded",
            "ingestion_id": ingestion_id,
            "ledger_path": str(ledger_abs),
        }

    for row in rows:
        if not isinstance(row, dict):
            continue
        if normalize_sha256(str(row.get("material_sha256") or "")) != material_sha:
            continue
        if normalize_sha256(str(row.get("markdown_sha256") or "")) != markdown_sha:
            continue
        if str(row.get("destination_path") or "") != destination_path:
            continue
        return False, "duplicate_guard_failed", {
            "error": "duplicate_material_markdown_destination_tuple",
            "ingestion_id": ingestion_id,
            "ledger_path": str(ledger_abs),
            "duplicate_entry_id": row.get("ingestion_id"),
        }

    return True, None, {"ledger_path": str(ledger_abs), "existing_rows": len(rows)}


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


def evaluation_replay_idempotent_noop(evaluation: Dict[str, Any]) -> bool:
    for gate in evaluation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        if gate.get("status") != "pass":
            continue
        details = gate.get("details") if isinstance(gate.get("details"), dict) else {}
        if bool(details.get("replay_idempotent_noop")):
            return True
    return False


def evaluate_packet(
    *,
    packet: Any,
    packet_path: Path,
    repo_root: Path,
    schema_path: Path,
    policy: Dict[str, Any],
    ledger_path: Path,
    allowed_root: Path,
    destination_profiles_path: Path,
) -> Dict[str, Any]:
    packet_dict = packet if isinstance(packet, dict) else {}
    ingestion_id = packet_dict.get("ingestion_id") if isinstance(packet_dict.get("ingestion_id"), str) else None

    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    gate_specs = [
        ("schema", lambda: gate_schema(packet, schema_path)),
        ("artifact_integrity", lambda: gate_artifact_integrity(packet_dict, repo_root)),
        ("markdown_gate", lambda: gate_markdown_decision(packet_dict, repo_root, policy)),
        ("classification_gate", lambda: gate_classification_decision(packet_dict, repo_root, policy)),
        ("destination_policy", lambda: gate_destination_policy(packet_dict, repo_root, policy, allowed_root, destination_profiles_path, ledger_path)),
        ("duplicate_guard", lambda: gate_duplicate_guard(packet_dict, ledger_path, repo_root)),
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
        "schema": "clawd.production_knowledge_ingestion.decision.v1",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "ingestion_id": ingestion_id,
        "packet": {"path": str(packet_path), "sha256": packet_sha},
        "policy": policy,
        "destination_profiles_path": str(destination_profiles_path),
        "gates": gates,
    }


def apply_ingestion(
    *,
    packet: Dict[str, Any],
    evaluation: Dict[str, Any],
    repo_root: Path,
    ledger_path: Path,
    latest_snapshot_path: Path,
) -> Tuple[bool, Dict[str, Any]]:
    destination = packet.get("destination") if isinstance(packet.get("destination"), dict) else {}
    raw_dest_path = str(destination.get("path") or "")
    if not raw_dest_path:
        return False, {"error": "destination_path_missing"}

    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else {}
    source_markdown_path = resolve_repo_path(repo_root, str(markdown_artifact.get("path") or ""))
    if not source_markdown_path.exists() or not source_markdown_path.is_file():
        return False, {"error": "markdown_source_missing", "path": str(source_markdown_path)}

    dest_abs = resolve_repo_path(repo_root, raw_dest_path)
    if not is_within(repo_root, dest_abs):
        return False, {"error": "destination_outside_repo", "path": raw_dest_path}

    meta_abs = dest_abs.with_suffix(dest_abs.suffix + ".meta.json")

    ingestion_id = str(packet.get("ingestion_id") or "")
    material_sha = normalize_sha256(str((packet.get("material_artifact") or {}).get("sha256") or ""))
    markdown_sha = normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or ""))

    if evaluation_replay_idempotent_noop(evaluation):
        if not dest_abs.exists() or not dest_abs.is_file():
            return False, {"error": "replay_idempotent_destination_missing", "path": raw_dest_path}
        existing_sha = file_sha256(dest_abs)
        if markdown_sha and existing_sha != markdown_sha:
            return False, {
                "error": "replay_idempotent_destination_sha_mismatch",
                "path": raw_dest_path,
                "expected": markdown_sha,
                "observed": existing_sha,
            }

        ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
        return True, {
            "destination_path": str(dest_abs.relative_to(repo_root)),
            "metadata_path": str(meta_abs.relative_to(repo_root)),
            "ledger_path": str(ledger_abs.relative_to(repo_root)) if is_within(repo_root, ledger_abs) else str(ledger_abs),
            "replay_idempotent_noop": True,
            "record": None,
        }

    record = {
        "schema": "clawd.production_knowledge_ingestion.record.v1",
        "ingested_at": now_iso(),
        "ingestion_id": ingestion_id,
        "material_sha256": material_sha,
        "markdown_sha256": markdown_sha,
        "destination_path": raw_dest_path,
        "destination_profile": str(destination.get("profile") or "") or None,
        "packet_path": str((evaluation.get("packet") or {}).get("path") or ""),
        "packet_sha256": str((evaluation.get("packet") or {}).get("sha256") or ""),
        "classification_label": None,
    }

    for gate in evaluation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        if gate.get("gate") == "classification_gate" and gate.get("status") == "pass":
            details = gate.get("details") if isinstance(gate.get("details"), dict) else {}
            record["classification_label"] = details.get("classification_label")

    wrote_destination = False
    wrote_meta = False

    try:
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest_abs.with_suffix(dest_abs.suffix + ".tmp")
        shutil.copy2(source_markdown_path, tmp_dest)
        tmp_dest.replace(dest_abs)
        wrote_destination = True

        metadata = {
            "schema": "clawd.production_knowledge_ingestion.metadata.v1",
            "ingestion_record": record,
            "source_material": packet.get("material_artifact"),
            "source_markdown": packet.get("markdown_artifact"),
            "markdown_gate_decision": packet.get("markdown_gate_decision"),
            "classification_decision": packet.get("classification_decision"),
            "evaluation": {
                "decision": evaluation.get("decision"),
                "block_reason": evaluation.get("block_reason"),
                "gates": evaluation.get("gates"),
            },
        }
        atomic_write_json(meta_abs, metadata)
        wrote_meta = True

        ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
        if not is_within(repo_root, ledger_abs):
            raise RuntimeError("unsafe_ledger_path")
        ledger_abs.parent.mkdir(parents=True, exist_ok=True)
        with ledger_abs.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(record) + "\n")

        latest_abs = latest_snapshot_path if latest_snapshot_path.is_absolute() else (repo_root / latest_snapshot_path).resolve()
        if is_within(repo_root, latest_abs):
            atomic_write_json(
                latest_abs,
                {
                    "schema": "clawd.production_knowledge_ingestion.latest.v1",
                    "updated_at": now_iso(),
                    "last_record": record,
                    "ledger_path": str(ledger_abs.relative_to(repo_root)),
                },
            )

        return True, {
            "destination_path": str(dest_abs.relative_to(repo_root)),
            "metadata_path": str(meta_abs.relative_to(repo_root)),
            "ledger_path": str(ledger_abs.relative_to(repo_root)),
            "record": record,
        }
    except Exception as exc:
        # Fail-closed best effort rollback
        if wrote_meta:
            try:
                meta_abs.unlink(missing_ok=True)
            except Exception:
                pass
        if wrote_destination:
            try:
                dest_abs.unlink(missing_ok=True)
            except Exception:
                pass
        return False, {"error": "ingestion_apply_failed", "detail": str(exc)}


def split_bytes_into_chunks(payload: bytes, chunk_count: int) -> List[bytes]:
    if chunk_count <= 0:
        raise ValueError("chunk_count_must_be_positive")
    if not payload:
        return [b""]

    total = len(payload)
    chunk_count = max(1, min(chunk_count, total))
    base = total // chunk_count
    rem = total % chunk_count

    chunks: List[bytes] = []
    start = 0
    for idx in range(chunk_count):
        step = base + (1 if idx < rem else 0)
        end = start + step
        chunks.append(payload[start:end])
        start = end
    return chunks


def gate_multi_host_fixture_contract(fixture: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "fixture_contract_invalid", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists():
        return False, "fixture_contract_invalid", {"error": "fixture_schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "fixture_contract_invalid", {"error": "fixture_schema_unreadable", "detail": str(exc)}

    try:
        Draft202012Validator(schema_doc, format_checker=FormatChecker()).validate(fixture)
    except Exception as exc:
        return False, "fixture_contract_invalid", {"error": "fixture_schema_validation_failed", "detail": str(exc)}

    return True, None, {"schema_path": str(schema_path)}


def apply_ingestion_multi_host(
    *,
    packet: Dict[str, Any],
    evaluation: Dict[str, Any],
    repo_root: Path,
    ledger_path: Path,
    latest_snapshot_path: Path,
    fixture_path: Path,
    fixture_schema_path: Path,
    runtime_latest_path: Path,
) -> Tuple[bool, Dict[str, Any]]:
    destination = packet.get("destination") if isinstance(packet.get("destination"), dict) else {}
    raw_dest_path = str(destination.get("path") or "")
    if not raw_dest_path:
        return False, {"error": "destination_path_missing"}

    markdown_artifact = packet.get("markdown_artifact") if isinstance(packet.get("markdown_artifact"), dict) else {}
    source_markdown_path = resolve_repo_path(repo_root, str(markdown_artifact.get("path") or ""))
    if not source_markdown_path.exists() or not source_markdown_path.is_file():
        return False, {"error": "markdown_source_missing", "path": str(source_markdown_path)}

    fixture_abs = fixture_path if fixture_path.is_absolute() else (repo_root / fixture_path).resolve()
    if not is_within(repo_root, fixture_abs):
        return False, {"error": "fixture_outside_repo", "path": str(fixture_abs)}
    if not fixture_abs.exists() or not fixture_abs.is_file():
        return False, {"error": "fixture_unresolved", "path": str(fixture_abs)}

    try:
        fixture = load_json_file(fixture_abs)
    except Exception as exc:
        return False, {"error": "fixture_unreadable", "path": str(fixture_abs), "detail": str(exc)}

    fixture_schema_abs = fixture_schema_path if fixture_schema_path.is_absolute() else (repo_root / fixture_schema_path).resolve()
    ok_fixture, reason_fixture, fixture_gate = gate_multi_host_fixture_contract(fixture, fixture_schema_abs)
    if not ok_fixture:
        return False, {"error": reason_fixture or "fixture_contract_invalid", **fixture_gate}

    hosts = [str(x) for x in (fixture.get("hosts") or []) if str(x)]
    host_set = set(hosts)
    if len(hosts) < 2:
        return False, {"error": "fixture_hosts_invalid", "detail": "at_least_two_hosts_required"}

    chunk_count = int(fixture.get("chunk_count") or 0)
    if chunk_count <= 0:
        return False, {"error": "fixture_chunk_count_invalid"}

    source_bytes = source_markdown_path.read_bytes()
    chunks = split_bytes_into_chunks(source_bytes, chunk_count)

    injected = fixture.get("fault_injections") if isinstance(fixture.get("fault_injections"), list) else []
    host_failures: Dict[str, int] = {}
    for row in injected:
        if not isinstance(row, dict):
            continue
        host_id = str(row.get("host") or "")
        trigger_chunk_index = row.get("trigger_chunk_index")
        if host_id in host_set and isinstance(trigger_chunk_index, int):
            prev = host_failures.get(host_id)
            host_failures[host_id] = min(prev, trigger_chunk_index) if isinstance(prev, int) else trigger_chunk_index

    assignments: List[Dict[str, Any]] = []
    handoffs: List[Dict[str, Any]] = []
    failed_chunks: List[int] = []
    recovered = True
    rebuilt_parts: List[bytes] = []

    for idx, chunk in enumerate(chunks):
        preferred = hosts[idx % len(hosts)]
        selected = preferred

        fail_from = host_failures.get(preferred)
        if isinstance(fail_from, int) and idx >= fail_from:
            failed_chunks.append(idx)
            fallback = None
            for candidate in hosts:
                fail_candidate = host_failures.get(candidate)
                if isinstance(fail_candidate, int) and idx >= fail_candidate:
                    continue
                if candidate == preferred:
                    continue
                fallback = candidate
                break

            if fallback is None:
                recovered = False
                assignments.append(
                    {
                        "chunk_index": idx,
                        "primary_host": preferred,
                        "selected_host": None,
                        "status": "failed_no_handoff",
                    }
                )
                break

            selected = fallback
            handoffs.append(
                {
                    "chunk_index": idx,
                    "from_host": preferred,
                    "to_host": selected,
                    "reason": "host_fault_injected",
                }
            )

        assignments.append(
            {
                "chunk_index": idx,
                "primary_host": preferred,
                "selected_host": selected,
                "status": "processed",
            }
        )
        rebuilt_parts.append(chunk)

    rebuilt = b"".join(rebuilt_parts)
    recovered = recovered and rebuilt == source_bytes

    if not recovered:
        return False, {
            "error": "multi_host_recovery_failed",
            "fixture_path": str(fixture_abs),
            "failed_chunks": failed_chunks,
            "rebuilt_matches_source": rebuilt == source_bytes,
            "assignments": assignments,
            "handoffs": handoffs,
        }

    dest_abs = resolve_repo_path(repo_root, raw_dest_path)
    if not is_within(repo_root, dest_abs):
        return False, {"error": "destination_outside_repo", "path": raw_dest_path}
    meta_abs = dest_abs.with_suffix(dest_abs.suffix + ".meta.json")

    ingestion_id = str(packet.get("ingestion_id") or "")
    material_sha = normalize_sha256(str((packet.get("material_artifact") or {}).get("sha256") or ""))
    markdown_sha = normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or ""))

    record = {
        "schema": "clawd.production_knowledge_ingestion.record.v1",
        "ingested_at": now_iso(),
        "ingestion_id": ingestion_id,
        "material_sha256": material_sha,
        "markdown_sha256": markdown_sha,
        "destination_path": raw_dest_path,
        "destination_profile": str(destination.get("profile") or "") or None,
        "packet_path": str((evaluation.get("packet") or {}).get("path") or ""),
        "packet_sha256": str((evaluation.get("packet") or {}).get("sha256") or ""),
        "classification_label": None,
        "ingestion_mode": "multi_host_fault_handoff_harness",
    }

    for gate in evaluation.get("gates") or []:
        if not isinstance(gate, dict):
            continue
        if gate.get("gate") == "classification_gate" and gate.get("status") == "pass":
            details = gate.get("details") if isinstance(gate.get("details"), dict) else {}
            record["classification_label"] = details.get("classification_label")

    runtime_artifact = {
        "schema": "clawd.b3.multi_host_fault_injection.runtime.v1",
        "generated_at": now_iso(),
        "fixture": {
            "path": str(fixture_abs.relative_to(repo_root)),
            "fixture_id": fixture.get("fixture_id"),
            "schema_version": fixture.get("schema_version"),
        },
        "ingestion_id": ingestion_id,
        "destination_path": raw_dest_path,
        "chunk_count": len(chunks),
        "hosts": hosts,
        "fault_injections": injected,
        "failed_chunks": failed_chunks,
        "handoff_count": len(handoffs),
        "handoffs": handoffs,
        "assignments": assignments,
        "recovery": {
            "status": "PASS",
            "rebuilt_matches_source": True,
            "destination_sha256": hashlib.sha256(rebuilt).hexdigest(),
            "source_sha256": normalize_sha256(str((packet.get("markdown_artifact") or {}).get("sha256") or "")),
        },
    }

    runtime_latest_abs = runtime_latest_path if runtime_latest_path.is_absolute() else (repo_root / runtime_latest_path).resolve()

    if evaluation_replay_idempotent_noop(evaluation):
        if not dest_abs.exists() or not dest_abs.is_file():
            return False, {"error": "replay_idempotent_destination_missing", "path": raw_dest_path}
        existing_sha = file_sha256(dest_abs)
        if markdown_sha and existing_sha != markdown_sha:
            return False, {
                "error": "replay_idempotent_destination_sha_mismatch",
                "path": raw_dest_path,
                "expected": markdown_sha,
                "observed": existing_sha,
            }

        runtime_latest_rel: Optional[str] = None
        if is_within(repo_root, runtime_latest_abs):
            atomic_write_json(runtime_latest_abs, runtime_artifact)
            runtime_latest_rel = str(runtime_latest_abs.relative_to(repo_root))

        ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
        return True, {
            "destination_path": str(dest_abs.relative_to(repo_root)),
            "metadata_path": str(meta_abs.relative_to(repo_root)),
            "ledger_path": str(ledger_abs.relative_to(repo_root)) if is_within(repo_root, ledger_abs) else str(ledger_abs),
            "runtime_latest_path": runtime_latest_rel,
            "replay_idempotent_noop": True,
            "record": None,
            "runtime": runtime_artifact,
        }

    wrote_destination = False
    wrote_meta = False
    wrote_runtime = False

    try:
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest_abs.with_suffix(dest_abs.suffix + ".tmp")
        tmp_dest.write_bytes(rebuilt)
        tmp_dest.replace(dest_abs)
        wrote_destination = True

        metadata = {
            "schema": "clawd.production_knowledge_ingestion.metadata.v1",
            "ingestion_record": record,
            "source_material": packet.get("material_artifact"),
            "source_markdown": packet.get("markdown_artifact"),
            "markdown_gate_decision": packet.get("markdown_gate_decision"),
            "classification_decision": packet.get("classification_decision"),
            "evaluation": {
                "decision": evaluation.get("decision"),
                "block_reason": evaluation.get("block_reason"),
                "gates": evaluation.get("gates"),
            },
            "multi_host_runtime": runtime_artifact,
        }
        atomic_write_json(meta_abs, metadata)
        wrote_meta = True

        ledger_abs = ledger_path if ledger_path.is_absolute() else (repo_root / ledger_path).resolve()
        if not is_within(repo_root, ledger_abs):
            raise RuntimeError("unsafe_ledger_path")
        ledger_abs.parent.mkdir(parents=True, exist_ok=True)
        with ledger_abs.open("a", encoding="utf-8") as fh:
            fh.write(stable_json_dumps(record) + "\n")

        latest_abs = latest_snapshot_path if latest_snapshot_path.is_absolute() else (repo_root / latest_snapshot_path).resolve()
        if is_within(repo_root, latest_abs):
            atomic_write_json(
                latest_abs,
                {
                    "schema": "clawd.production_knowledge_ingestion.latest.v1",
                    "updated_at": now_iso(),
                    "last_record": record,
                    "ledger_path": str(ledger_abs.relative_to(repo_root)),
                },
            )

        if is_within(repo_root, runtime_latest_abs):
            atomic_write_json(runtime_latest_abs, runtime_artifact)
            wrote_runtime = True

        return True, {
            "destination_path": str(dest_abs.relative_to(repo_root)),
            "metadata_path": str(meta_abs.relative_to(repo_root)),
            "ledger_path": str(ledger_abs.relative_to(repo_root)),
            "runtime_latest_path": str(runtime_latest_abs.relative_to(repo_root)) if wrote_runtime else None,
            "record": record,
            "runtime": runtime_artifact,
        }
    except Exception as exc:
        if wrote_meta:
            try:
                meta_abs.unlink(missing_ok=True)
            except Exception:
                pass
        if wrote_destination:
            try:
                dest_abs.unlink(missing_ok=True)
            except Exception:
                pass
        return False, {"error": "ingestion_apply_failed", "detail": str(exc)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Production knowledge ingestion runner (v1)")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="Ingestion packet schema path")
    ap.add_argument("--decision-log", default=str(DEFAULT_DECISION_LOG), help="Append-only decision log path")
    ap.add_argument("--ledger-path", default=str(DEFAULT_LEDGER_PATH), help="Append-only ingestion ledger path")
    ap.add_argument("--latest-snapshot-path", default=str(DEFAULT_LATEST_SNAPSHOT), help="Latest ingestion snapshot JSON")
    ap.add_argument("--allowed-destination-root", default=str(DEFAULT_ALLOWED_ROOT), help="Allowed destination root")
    ap.add_argument("--destination-profiles-path", default=str(DEFAULT_DESTINATION_PROFILES), help="Destination profile policy JSON")
    ap.add_argument("--json", action="store_true", help="Pretty JSON output")

    sub = ap.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate ingestion packet only")
    p_eval.add_argument("--packet", required=True)
    p_eval.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_ingest = sub.add_parser("ingest", help="Evaluate + apply ingestion packet")
    p_ingest.add_argument("--packet", required=True)
    p_ingest.add_argument("--allow-overwrite", action="store_true")
    p_ingest.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_apply = sub.add_parser("apply", help="Alias of ingest")
    p_apply.add_argument("--packet", required=True)
    p_apply.add_argument("--allow-overwrite", action="store_true")
    p_apply.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    p_multi = sub.add_parser("ingest-multi-host", help="Evaluate + apply ingestion packet with multi-host fault-injection handoff harness")
    p_multi.add_argument("--packet", required=True)
    p_multi.add_argument("--fault-fixture", required=True, help="B3 multi-host fault-injection fixture JSON path")
    p_multi.add_argument("--fault-fixture-schema", default=str(DEFAULT_MULTI_HOST_FIXTURE_SCHEMA_PATH), help="Multi-host fault fixture schema path")
    p_multi.add_argument("--runtime-latest-path", default=str(DEFAULT_MULTI_HOST_RUNTIME_LATEST), help="Latest multi-host runtime artifact path")
    p_multi.add_argument("--allow-overwrite", action="store_true")
    p_multi.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    wrapper_guard_error = enforce_wrapper_only_contract(str(args.command or ""))
    if wrapper_guard_error is not None:
        print(json.dumps(wrapper_guard_error, ensure_ascii=False, indent=2 if args.json else None))
        return 2

    repo_root = Path(args.repo_root).expanduser().resolve()
    schema_path = Path(args.schema_path).expanduser().resolve()
    ledger_path = Path(args.ledger_path).expanduser()
    decision_log_path: Optional[Path] = Path(args.decision_log).expanduser() if args.decision_log else None
    latest_snapshot_path = Path(args.latest_snapshot_path).expanduser()
    allowed_root = Path(args.allowed_destination_root).expanduser()
    destination_profiles_path = Path(args.destination_profiles_path).expanduser()

    packet_path = Path(args.packet).expanduser().resolve()

    try:
        packet = load_json_file(packet_path)
    except Exception as exc:
        result = {
            "schema": "clawd.production_knowledge_ingestion.decision.v1",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "schema",
            "block_reason": "schema_invalid",
            "ingestion_id": None,
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
                {"gate": "markdown_gate", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "classification_gate", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "destination_policy", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "duplicate_guard", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        packet_dict = packet if isinstance(packet, dict) else {}
        cli_allow_overwrite = bool(getattr(args, "allow_overwrite", False))
        policy = apply_policy_overrides(packet_dict, cli_allow_overwrite)
        result = evaluate_packet(
            packet=packet,
            packet_path=packet_path,
            repo_root=repo_root,
            schema_path=schema_path,
            policy=policy,
            ledger_path=ledger_path,
            allowed_root=allowed_root,
            destination_profiles_path=destination_profiles_path,
        )

    result["decision_record"] = append_decision_record(decision_log_path, repo_root, result)

    if args.command in {"ingest", "apply", "ingest-multi-host"}:
        if result.get("decision") != "PASS":
            result["apply_result"] = {"applied": False, "reason": "evaluation_blocked"}
            rc = 2
        else:
            packet_dict = packet if isinstance(packet, dict) else {}
            if args.command == "ingest-multi-host":
                ok, apply_details = apply_ingestion_multi_host(
                    packet=packet_dict,
                    evaluation=result,
                    repo_root=repo_root,
                    ledger_path=ledger_path,
                    latest_snapshot_path=latest_snapshot_path,
                    fixture_path=Path(args.fault_fixture).expanduser(),
                    fixture_schema_path=Path(args.fault_fixture_schema).expanduser(),
                    runtime_latest_path=Path(args.runtime_latest_path).expanduser(),
                )
            else:
                ok, apply_details = apply_ingestion(
                    packet=packet_dict,
                    evaluation=result,
                    repo_root=repo_root,
                    ledger_path=ledger_path,
                    latest_snapshot_path=latest_snapshot_path,
                )
            result["apply_result"] = {"applied": ok, **apply_details}
            rc = 0 if ok else 2
    else:
        rc = 0 if result.get("decision") == "PASS" else 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(stable_json_dumps(result))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
