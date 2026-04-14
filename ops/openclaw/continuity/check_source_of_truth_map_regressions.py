#!/usr/bin/env python3
"""Deterministic anti-drift guard for canonical source-of-truth map governance."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_MAP_PATH = "reports/openclaw_system_source_of_truth_map_2026-03-20.md"
DEFAULT_WAIVER_PATH = "state/continuity/latest/anti_drift_waiver_register.json"
DEFAULT_DRIFT_OUTPUT_PATH = "state/continuity/latest/source_of_truth_map_drift_latest.json"
DEFAULT_WAIVER_SCHEMA_PATH = "docs/ops/schemas/anti_drift_waiver_register.schema.json"
DEFAULT_POLICY_PACK_PATH = "state/continuity/latest/source_of_truth_map_guard_policy_v1.json"
DEFAULT_POLICY_SCHEMA_PATH = "docs/ops/schemas/source_of_truth_map_guard_policy_pack.schema.json"

CHECKER_ID = "source_of_truth_map_guard"
CHECKER_VERSION = "1.1.0"
EXPECTED_WAIVER_SCHEMA_VERSION = "clawd.anti_drift_waiver_register.v1"
EXPECTED_POLICY_SCHEMA_VERSION = "clawd.source_of_truth_map_guard_policy_pack.v1"

REQUIRED_ACTIVE_DOCS = [
    "reports/openclaw_full_roadmap_2026-03-20.md",
    "reports/openclaw_full_roadmap_execution_table_2026-03-20.md",
    "reports/openclaw_system_source_of_truth_map_2026-03-20.md",
    "docs/ops/source_of_truth_and_subagent_bootstrap_doctrine_v1.md",
    "docs/ops/unified_operating_doctrine_v1.md",
]

REQUIRED_HISTORICAL_DOCS = [
    "reports/system_master_roadmap_2026-03-13.md",
    "reports/system_prioritized_roadmap_table_2026-03-13.md",
]

REQUIRED_LANE_MARKERS = [
    "- **Name:**",
    "- **Purpose:**",
    "- **Canonical roadmap doc(s):**",
    "- **Canonical spec(s):**",
    "- **Implementation files:**",
    "- **Tests / validation entrypoints:**",
    "- **Operator surfaces:**",
    "- **Status of docs:**",
]

FORBIDDEN_B6_REFERENCE_DOCS = [
    "reports/kimi_k25_integration_synthesis_openclaw_2026-03-21.md",
]

SYSTEM_NON_WAIVABLE_REASON_IDS = {
    "policy_pack_missing",
    "policy_pack_invalid",
    "policy_contract_violation",
    "waiver_register_missing",
    "waiver_register_invalid",
    "waiver_contract_violation",
    "drift_marker_write_failed",
}

SYSTEM_REASON_IDS = SYSTEM_NON_WAIVABLE_REASON_IDS | {
    "map_missing",
}

SYSTEM_CHECK_IDS = {
    "map_path_exists",
    "policy_pack_valid",
    "waiver_register_valid",
    "drift_artifact_written",
}

LANE_HEADING_RE = re.compile(r"^##\s+(.+?\s(?:-|—)\s.+?)\s*$")


class PolicyValidationError(RuntimeError):
    pass


class WaiverValidationError(RuntimeError):
    pass


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_iso(raw: Any) -> Optional[dt.datetime]:
    txt = str(raw or "").strip()
    if not txt:
        return None
    probe = txt[:-1] + "+00:00" if txt.endswith("Z") else txt
    try:
        parsed = dt.datetime.fromisoformat(probe)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_repo_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    else:
        p = p.resolve()
    return p


def to_rel(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            tmp_path = Path(fh.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _collect_section_paths(text: str, section_title: str) -> List[str]:
    lines = text.splitlines()
    start: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip() == section_title:
            start = idx + 1
            break
    if start is None:
        return []

    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("## ") or line.startswith("### "):
            break
        if not line.strip().startswith("- "):
            continue
        m = re.search(r"`([^`]+)`", line)
        if not m:
            continue
        out.append(m.group(1).strip())
    return out


def _collect_lane_blocks(text: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    indices: List[int] = []
    headings: List[str] = []

    for idx, line in enumerate(lines):
        m = LANE_HEADING_RE.match(line)
        if not m:
            continue
        indices.append(idx)
        headings.append(m.group(1).strip())

    blocks: List[Tuple[str, str]] = []
    for i, start in enumerate(indices):
        end = indices[i + 1] if i + 1 < len(indices) else len(lines)
        blocks.append((headings[i], "\n".join(lines[start + 1 : end])))

    return blocks


def _collect_markdown_paths(text: str) -> List[str]:
    paths: List[str] = []
    for m in re.finditer(r"`([^`]+)`", text):
        token = m.group(1).strip()
        if not token:
            continue
        if token.startswith("pytest ") or token.startswith("bash "):
            continue
        if " " in token:
            continue
        if "*" in token:
            continue
        if "/" not in token:
            continue
        if token.startswith("state/"):
            continue
        if not token.startswith(("docs/", "ops/", "scripts/", "tests/", "reports/")):
            continue
        paths.append(token)
    return sorted(set(paths))


def _validate_policy_payload_minimal(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise PolicyValidationError("policy_pack_not_object")

    required_keys = {
        "schema",
        "generated_at",
        "policy_pack_id",
        "checker",
        "scope",
        "status",
        "default_decision",
        "checks",
        "source_refs",
    }
    missing = [k for k in sorted(required_keys) if k not in payload]
    if missing:
        raise PolicyValidationError(f"policy_pack_missing_keys:{','.join(missing)}")

    if payload.get("schema") != EXPECTED_POLICY_SCHEMA_VERSION:
        raise PolicyValidationError("policy_pack_schema_version_mismatch")
    if payload.get("checker") != CHECKER_ID:
        raise PolicyValidationError("policy_pack_checker_mismatch")
    if payload.get("default_decision") != "BLOCK":
        raise PolicyValidationError("policy_pack_default_decision_invalid")
    if _parse_iso(payload.get("generated_at")) is None:
        raise PolicyValidationError("policy_pack_generated_at_invalid")
    if str(payload.get("status") or "").strip() not in {"draft", "approved", "deprecated"}:
        raise PolicyValidationError("policy_pack_status_invalid")
    if not isinstance(payload.get("checks"), list):
        raise PolicyValidationError("policy_pack_checks_not_array")
    if not isinstance(payload.get("source_refs"), list) or not payload.get("source_refs"):
        raise PolicyValidationError("policy_pack_source_refs_invalid")


def _load_and_validate_policy_pack(
    *,
    repo_root: Path,
    policy_pack_path: Path,
    policy_schema_path: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not policy_pack_path.exists() or not policy_pack_path.is_file():
        raise PolicyValidationError("policy_pack_missing")

    try:
        payload = json.loads(policy_pack_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise PolicyValidationError(f"policy_pack_json_invalid:{exc}") from exc

    _validate_policy_payload_minimal(payload)

    if not policy_schema_path.exists() or not policy_schema_path.is_file():
        raise PolicyValidationError("policy_schema_missing")

    try:
        schema_doc = json.loads(policy_schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise PolicyValidationError(f"policy_schema_json_invalid:{exc}") from exc

    try:
        import jsonschema  # type: ignore

        jsonschema.validate(payload, schema_doc)
    except Exception as exc:
        raise PolicyValidationError(f"policy_schema_validation_failed:{exc}") from exc

    checks_by_id: Dict[str, Dict[str, Any]] = {}
    reasons_by_id: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(payload.get("checks") or []):
        if not isinstance(row, dict):
            raise PolicyValidationError(f"policy_row_not_object:{idx}")

        check_id = str(row.get("check_id") or "").strip()
        reason_id = str(row.get("reason_id") or "").strip()
        if not check_id or not reason_id:
            raise PolicyValidationError(f"policy_row_missing_ids:{idx}")
        if check_id in checks_by_id:
            raise PolicyValidationError(f"policy_duplicate_check_id:{check_id}")
        if reason_id in reasons_by_id:
            raise PolicyValidationError(f"policy_duplicate_reason_id:{reason_id}")

        checks_by_id[check_id] = dict(row)
        reasons_by_id[reason_id] = dict(row)

    if not checks_by_id:
        raise PolicyValidationError("policy_pack_checks_empty")

    summary = {
        "policy_pack_path": to_rel(repo_root, policy_pack_path),
        "policy_schema_path": to_rel(repo_root, policy_schema_path),
        "policy_pack_id": payload.get("policy_pack_id"),
        "status": payload.get("status"),
        "policy_check_count": len(checks_by_id),
        "waivable_reason_ids": sorted(
            reason_id
            for reason_id, row in reasons_by_id.items()
            if bool(row.get("waivable"))
        ),
        "non_waivable_reason_ids": sorted(
            reason_id
            for reason_id, row in reasons_by_id.items()
            if not bool(row.get("waivable"))
        ),
    }
    return checks_by_id, reasons_by_id, summary


def _validate_waiver_payload_minimal(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise WaiverValidationError("waiver_register_not_object")

    required_keys = {"schema_version", "checker", "generated_at", "max_active_waivers", "waivers"}
    missing = [k for k in sorted(required_keys) if k not in payload]
    if missing:
        raise WaiverValidationError(f"waiver_register_missing_keys:{','.join(missing)}")

    if payload.get("schema_version") != EXPECTED_WAIVER_SCHEMA_VERSION:
        raise WaiverValidationError("waiver_register_schema_version_mismatch")
    if payload.get("checker") != CHECKER_ID:
        raise WaiverValidationError("waiver_register_checker_mismatch")

    if _parse_iso(payload.get("generated_at")) is None:
        raise WaiverValidationError("waiver_register_generated_at_invalid")

    max_active = payload.get("max_active_waivers")
    if not isinstance(max_active, int) or max_active < 0 or max_active > 10:
        raise WaiverValidationError("waiver_register_max_active_waivers_invalid")

    waivers = payload.get("waivers")
    if not isinstance(waivers, list):
        raise WaiverValidationError("waiver_register_waivers_not_array")


def _load_and_validate_waivers(
    *,
    repo_root: Path,
    waiver_path: Path,
    waiver_schema_path: Path,
    policy_checks_by_id: Dict[str, Dict[str, Any]],
    policy_reasons_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not waiver_path.exists() or not waiver_path.is_file():
        raise WaiverValidationError("waiver_register_missing")

    try:
        payload = json.loads(waiver_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise WaiverValidationError(f"waiver_register_json_invalid:{exc}") from exc

    _validate_waiver_payload_minimal(payload)

    if not waiver_schema_path.exists() or not waiver_schema_path.is_file():
        raise WaiverValidationError("waiver_schema_missing")

    try:
        schema_doc = json.loads(waiver_schema_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        raise WaiverValidationError(f"waiver_schema_json_invalid:{exc}") from exc

    try:
        import jsonschema  # type: ignore

        jsonschema.validate(payload, schema_doc)
    except Exception as exc:
        raise WaiverValidationError(f"waiver_schema_validation_failed:{exc}") from exc

    waivers_raw = payload.get("waivers") if isinstance(payload.get("waivers"), list) else []
    waivers: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    active_status_count = 0

    for idx, row in enumerate(waivers_raw):
        if not isinstance(row, dict):
            raise WaiverValidationError(f"waiver_row_not_object:{idx}")

        waiver_id = str(row.get("waiver_id") or "").strip()
        if not waiver_id:
            raise WaiverValidationError(f"waiver_id_missing:{idx}")
        if waiver_id in seen_ids:
            raise WaiverValidationError(f"waiver_id_duplicate:{waiver_id}")
        seen_ids.add(waiver_id)

        issued_at = _parse_iso(row.get("issued_at"))
        expires_at = _parse_iso(row.get("expires_at"))
        if issued_at is None or expires_at is None:
            raise WaiverValidationError(f"waiver_timestamp_invalid:{waiver_id}")
        if expires_at <= issued_at:
            raise WaiverValidationError(f"waiver_expiry_not_after_issue:{waiver_id}")

        status = str(row.get("status") or "").strip()
        if status == "active":
            active_status_count += 1

        scope = row.get("scope")
        if not isinstance(scope, dict):
            raise WaiverValidationError(f"waiver_scope_invalid:{waiver_id}")

        reason_ids = [str(v).strip() for v in (scope.get("reason_ids") or []) if str(v).strip()]
        check_ids = [str(v).strip() for v in (scope.get("check_ids") or []) if str(v).strip()]
        if not reason_ids and not check_ids:
            raise WaiverValidationError(f"waiver_scope_targets_missing:{waiver_id}")

        unknown_reasons = [rid for rid in reason_ids if rid not in policy_reasons_by_id]
        unknown_checks = [cid for cid in check_ids if cid not in policy_checks_by_id]
        if unknown_reasons:
            raise WaiverValidationError(f"waiver_unknown_reason_ids:{waiver_id}:{','.join(sorted(set(unknown_reasons)))}")
        if unknown_checks:
            raise WaiverValidationError(f"waiver_unknown_check_ids:{waiver_id}:{','.join(sorted(set(unknown_checks)))}")

        non_waivable_reasons = [
            rid for rid in reason_ids if not bool((policy_reasons_by_id.get(rid) or {}).get("waivable"))
        ]
        non_waivable_checks = [
            cid for cid in check_ids if not bool((policy_checks_by_id.get(cid) or {}).get("waivable"))
        ]
        if non_waivable_reasons:
            raise WaiverValidationError(
                f"waiver_targets_non_waivable_reason_ids:{waiver_id}:{','.join(sorted(set(non_waivable_reasons)))}"
            )
        if non_waivable_checks:
            raise WaiverValidationError(
                f"waiver_targets_non_waivable_check_ids:{waiver_id}:{','.join(sorted(set(non_waivable_checks)))}"
            )

        waivers.append(dict(row))

    max_active = int(payload.get("max_active_waivers"))
    if active_status_count > max_active:
        raise WaiverValidationError(f"waiver_active_count_exceeds_cap:{active_status_count}>{max_active}")

    summary = {
        "waiver_path": to_rel(repo_root, waiver_path),
        "waiver_schema_path": to_rel(repo_root, waiver_schema_path),
        "waiver_count": len(waivers),
        "max_active_waivers": payload.get("max_active_waivers"),
    }
    return waivers, summary


def _apply_policy_to_failures(
    *,
    failures: List[Dict[str, Any]],
    policy_checks_by_id: Dict[str, Dict[str, Any]],
    policy_reasons_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    enriched: List[Dict[str, Any]] = []
    violations: List[str] = []

    for row in failures:
        reason_id = str(row.get("reason_id") or "").strip()
        check_id = str(row.get("check_id") or "").strip()

        if reason_id in SYSTEM_REASON_IDS or check_id in SYSTEM_CHECK_IDS:
            enriched.append(dict(row))
            continue

        policy_by_reason = policy_reasons_by_id.get(reason_id)
        policy_by_check = policy_checks_by_id.get(check_id)
        if not policy_by_reason or not policy_by_check:
            violations.append(f"missing_policy_mapping:{check_id}:{reason_id}")
            enriched.append(dict(row))
            continue
        if policy_by_reason.get("check_id") != check_id or policy_by_check.get("reason_id") != reason_id:
            violations.append(f"policy_pair_mismatch:{check_id}:{reason_id}")
            enriched.append(dict(row))
            continue

        merged = dict(row)
        merged["policy"] = {
            "category": policy_by_check.get("category"),
            "severity": policy_by_check.get("severity"),
            "waivable": bool(policy_by_check.get("waivable")),
            "summary": policy_by_check.get("summary"),
        }
        enriched.append(merged)

    return enriched, violations


def _apply_waivers(
    *,
    failures: List[Dict[str, Any]],
    waivers: Sequence[Dict[str, Any]],
    map_path_rel: str,
    policy_reasons_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    now = dt.datetime.now(dt.timezone.utc)

    applied_failures: List[Dict[str, Any]] = []
    applied_waiver_ids: List[str] = []
    expired_waiver_ids: List[str] = []
    inactive_waiver_ids: List[str] = []

    active_waivers_seen = 0

    for failure in failures:
        reason_id = str(failure.get("reason_id") or "").strip()
        check_id = str(failure.get("check_id") or "").strip()

        row = dict(failure)
        row["waived"] = False
        row["waived_by"] = None

        if reason_id in SYSTEM_NON_WAIVABLE_REASON_IDS:
            applied_failures.append(row)
            continue

        policy_row = policy_reasons_by_id.get(reason_id)
        if not policy_row or not bool(policy_row.get("waivable")):
            applied_failures.append(row)
            continue

        for waiver in waivers:
            waiver_id = str(waiver.get("waiver_id") or "").strip()
            status = str(waiver.get("status") or "").strip()
            scope = waiver.get("scope") if isinstance(waiver.get("scope"), dict) else {}

            if status != "active":
                inactive_waiver_ids.append(waiver_id)
                continue

            active_waivers_seen += 1

            expires_at = _parse_iso(waiver.get("expires_at"))
            if expires_at is None or expires_at <= now:
                expired_waiver_ids.append(waiver_id)
                continue

            reason_ids = {str(v).strip() for v in (scope.get("reason_ids") or []) if str(v).strip()}
            check_ids = {str(v).strip() for v in (scope.get("check_ids") or []) if str(v).strip()}
            path_allowlist = {str(v).strip() for v in (scope.get("path_allowlist") or []) if str(v).strip()}

            if path_allowlist and map_path_rel not in path_allowlist:
                continue

            if reason_id not in reason_ids and check_id not in check_ids:
                continue

            row["waived"] = True
            row["waived_by"] = waiver_id
            applied_waiver_ids.append(waiver_id)
            break

        applied_failures.append(row)

    unique_applied = sorted(set(wid for wid in applied_waiver_ids if wid))
    unique_expired = sorted(set(wid for wid in expired_waiver_ids if wid))
    unique_inactive = sorted(set(wid for wid in inactive_waiver_ids if wid))

    waiver_summary = {
        "evaluated_failure_count": len(failures),
        "active_waiver_evaluations": active_waivers_seen,
        "applied_waiver_ids": unique_applied,
        "expired_waiver_ids": unique_expired,
        "inactive_waiver_ids": unique_inactive,
        "applied_failure_count": sum(1 for row in applied_failures if bool(row.get("waived"))),
    }

    return applied_failures, waiver_summary


def evaluate(
    *,
    repo_root: Path,
    map_path: Path,
    waiver_path: Path,
    waiver_schema_path: Path,
    policy_pack_path: Path,
    policy_schema_path: Path,
    drift_output_path: Path,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    map_path_rel = to_rel(repo_root, map_path)
    waiver_path_rel = to_rel(repo_root, waiver_path)
    waiver_schema_path_rel = to_rel(repo_root, waiver_schema_path)
    policy_pack_rel = to_rel(repo_root, policy_pack_path)
    policy_schema_rel = to_rel(repo_root, policy_schema_path)
    drift_output_rel = to_rel(repo_root, drift_output_path)

    if not map_path.exists() or not map_path.is_file():
        return {
            "schema": "clawd.source_of_truth_map_guard.v1",
            "checker_id": CHECKER_ID,
            "checker_version": CHECKER_VERSION,
            "generated_at": now_iso(),
            "decision": "BLOCK",
            "block_reason": "map_missing",
            "block_reasons": ["map_missing"],
            "map_path": map_path_rel,
            "policy_pack_path": policy_pack_rel,
            "policy_schema_path": policy_schema_rel,
            "waiver_register_path": waiver_path_rel,
            "waiver_schema_path": waiver_schema_path_rel,
            "drift_output_path": drift_output_rel,
            "checks": [
                {
                    "check": "map_path_exists",
                    "check_id": "map_path_exists",
                    "ok": False,
                    "missing": [map_path_rel],
                }
            ],
            "failures": [
                {
                    "reason_id": "map_missing",
                    "check_id": "map_path_exists",
                    "waived": False,
                    "waived_by": None,
                }
            ],
            "policy_summary": {
                "policy_pack_path": policy_pack_rel,
                "policy_schema_path": policy_schema_rel,
                "status": "not_evaluated_map_missing",
            },
            "waiver_summary": {
                "waiver_path": waiver_path_rel,
                "waiver_schema_path": waiver_schema_path_rel,
                "status": "not_evaluated_map_missing",
            },
        }

    text = load_text(map_path)

    policy_summary: Dict[str, Any] = {
        "policy_pack_path": policy_pack_rel,
        "policy_schema_path": policy_schema_rel,
        "status": "unchecked",
    }
    waiver_summary: Dict[str, Any] = {
        "waiver_path": waiver_path_rel,
        "waiver_schema_path": waiver_schema_path_rel,
        "status": "unchecked",
    }
    policy_checks_by_id: Dict[str, Dict[str, Any]] = {}
    policy_reasons_by_id: Dict[str, Dict[str, Any]] = {}

    try:
        policy_checks_by_id, policy_reasons_by_id, loaded_policy_summary = _load_and_validate_policy_pack(
            repo_root=repo_root,
            policy_pack_path=policy_pack_path,
            policy_schema_path=policy_schema_path,
        )
        policy_summary.update(loaded_policy_summary)
        policy_summary["status"] = "valid"
        checks.append(
            {
                "check": "policy_pack_valid",
                "check_id": "policy_pack_valid",
                "ok": True,
                "policy_check_count": len(policy_checks_by_id),
            }
        )
    except PolicyValidationError as exc:
        checks.append(
            {
                "check": "policy_pack_valid",
                "check_id": "policy_pack_valid",
                "ok": False,
                "error": str(exc),
            }
        )
        policy_summary["status"] = "invalid"
        policy_summary["error"] = str(exc)
        failures.append(
            {
                "reason_id": "policy_pack_invalid" if str(exc) != "policy_pack_missing" else "policy_pack_missing",
                "check_id": "policy_pack_valid",
                "detail": str(exc),
                "waived": False,
                "waived_by": None,
            }
        )

    active = _collect_section_paths(text, "### Active / canonical planning + doctrine")
    historical = _collect_section_paths(text, "### Historical steering snapshots (reference-only)")

    active_missing = [row for row in REQUIRED_ACTIVE_DOCS if row not in active]
    checks.append(
        {
            "check": "active_registry_required_docs",
            "check_id": "active_registry_required_docs",
            "ok": len(active_missing) == 0,
            "missing": active_missing,
        }
    )
    if active_missing:
        failures.append(
            {
                "reason_id": "active_registry_missing_required_docs",
                "check_id": "active_registry_required_docs",
                "detail": active_missing,
            }
        )

    historical_missing = [row for row in REQUIRED_HISTORICAL_DOCS if row not in historical]
    checks.append(
        {
            "check": "historical_registry_required_docs",
            "check_id": "historical_registry_required_docs",
            "ok": len(historical_missing) == 0,
            "missing": historical_missing,
        }
    )
    if historical_missing:
        failures.append(
            {
                "reason_id": "historical_registry_missing_required_docs",
                "check_id": "historical_registry_required_docs",
                "detail": historical_missing,
            }
        )

    overlap = sorted(set(active).intersection(set(historical)))
    checks.append(
        {
            "check": "canonical_historical_no_overlap",
            "check_id": "canonical_historical_no_overlap",
            "ok": len(overlap) == 0,
            "overlap": overlap,
        }
    )
    if overlap:
        failures.append(
            {
                "reason_id": "canonical_historical_overlap",
                "check_id": "canonical_historical_no_overlap",
                "detail": overlap,
            }
        )

    historical_in_active = [row for row in REQUIRED_HISTORICAL_DOCS if row in active]
    checks.append(
        {
            "check": "historical_docs_not_in_active_registry",
            "check_id": "historical_docs_not_in_active_registry",
            "ok": len(historical_in_active) == 0,
            "violations": historical_in_active,
        }
    )
    if historical_in_active:
        failures.append(
            {
                "reason_id": "historical_docs_promoted_to_active",
                "check_id": "historical_docs_not_in_active_registry",
                "detail": historical_in_active,
            }
        )

    lane_blocks = _collect_lane_blocks(text)

    lane_extraction_ok = len(lane_blocks) > 0
    checks.append(
        {
            "check": "lane_block_extraction_non_empty",
            "check_id": "lane_block_extraction_non_empty",
            "ok": lane_extraction_ok,
            "lane_count": len(lane_blocks),
        }
    )
    if not lane_extraction_ok:
        failures.append(
            {
                "reason_id": "lane_block_extraction_failed",
                "check_id": "lane_block_extraction_non_empty",
                "detail": "No lane blocks were extracted from ## lane headings.",
            }
        )

    lane_missing_markers: List[Dict[str, Any]] = []
    for heading, block in lane_blocks:
        missing = [marker for marker in REQUIRED_LANE_MARKERS if marker not in block]
        if missing:
            lane_missing_markers.append({"lane": heading, "missing_markers": missing})

    checks.append(
        {
            "check": "lane_blocks_have_required_markers",
            "check_id": "lane_blocks_have_required_markers",
            "ok": len(lane_missing_markers) == 0,
            "lane_count": len(lane_blocks),
            "violations": lane_missing_markers,
        }
    )
    if lane_missing_markers:
        failures.append(
            {
                "reason_id": "lane_block_marker_drift",
                "check_id": "lane_blocks_have_required_markers",
                "detail": lane_missing_markers,
            }
        )

    b6_block = ""
    for heading, block in lane_blocks:
        if heading.startswith("B6 "):
            b6_block = block
            break

    b6_forbidden_refs = [row for row in FORBIDDEN_B6_REFERENCE_DOCS if row in b6_block]
    checks.append(
        {
            "check": "b6_forbidden_historical_refs_absent",
            "check_id": "b6_forbidden_historical_refs_absent",
            "ok": len(b6_forbidden_refs) == 0,
            "violations": b6_forbidden_refs,
        }
    )
    if b6_forbidden_refs:
        failures.append(
            {
                "reason_id": "b6_historical_ref_drift",
                "check_id": "b6_forbidden_historical_refs_absent",
                "detail": b6_forbidden_refs,
            }
        )

    referenced_paths = _collect_markdown_paths(text)
    unresolved_paths: List[str] = []
    for raw in referenced_paths:
        p = resolve_repo_path(repo_root, raw)
        if not p.exists():
            unresolved_paths.append(raw)

    checks.append(
        {
            "check": "referenced_paths_resolve",
            "check_id": "referenced_paths_resolve",
            "ok": len(unresolved_paths) == 0,
            "referenced_count": len(referenced_paths),
            "unresolved_count": len(unresolved_paths),
            "unresolved": unresolved_paths[:50],
        }
    )
    if unresolved_paths:
        failures.append(
            {
                "reason_id": "referenced_path_drift",
                "check_id": "referenced_paths_resolve",
                "detail": unresolved_paths,
            }
        )

    if policy_checks_by_id and policy_reasons_by_id:
        failures, policy_violations = _apply_policy_to_failures(
            failures=failures,
            policy_checks_by_id=policy_checks_by_id,
            policy_reasons_by_id=policy_reasons_by_id,
        )
        policy_summary["runtime_policy_coverage_ok"] = len(policy_violations) == 0
        policy_summary["runtime_policy_violations"] = policy_violations
        if policy_violations:
            failures.append(
                {
                    "reason_id": "policy_contract_violation",
                    "check_id": "policy_pack_valid",
                    "detail": policy_violations,
                    "waived": False,
                    "waived_by": None,
                }
            )
    else:
        policy_summary["runtime_policy_coverage_ok"] = False
        policy_summary["runtime_policy_violations"] = ["policy_pack_unavailable"]

    if policy_checks_by_id and policy_reasons_by_id:
        try:
            waivers, loaded_summary = _load_and_validate_waivers(
                repo_root=repo_root,
                waiver_path=waiver_path,
                waiver_schema_path=waiver_schema_path,
                policy_checks_by_id=policy_checks_by_id,
                policy_reasons_by_id=policy_reasons_by_id,
            )
            waiver_summary.update(loaded_summary)
            waiver_summary["status"] = "valid"
            checks.append(
                {
                    "check": "waiver_register_valid",
                    "check_id": "waiver_register_valid",
                    "ok": True,
                    "waiver_count": len(waivers),
                }
            )
            failures, applied_summary = _apply_waivers(
                failures=failures,
                waivers=waivers,
                map_path_rel=map_path_rel,
                policy_reasons_by_id=policy_reasons_by_id,
            )
            waiver_summary.update(applied_summary)
        except WaiverValidationError as exc:
            checks.append(
                {
                    "check": "waiver_register_valid",
                    "check_id": "waiver_register_valid",
                    "ok": False,
                    "error": str(exc),
                }
            )
            waiver_summary["status"] = "invalid"
            waiver_summary["error"] = str(exc)
            failures.append(
                {
                    "reason_id": "waiver_register_invalid" if str(exc) != "waiver_register_missing" else "waiver_register_missing",
                    "check_id": "waiver_register_valid",
                    "detail": str(exc),
                    "waived": False,
                    "waived_by": None,
                }
            )
    else:
        waiver_summary["status"] = "not_evaluated_policy_invalid"

    for row in failures:
        row.setdefault("waived", False)
        row.setdefault("waived_by", None)

    block_reasons = [
        str(row.get("reason_id") or "")
        for row in failures
        if not bool(row.get("waived")) and str(row.get("reason_id") or "")
    ]

    seen_reason: set[str] = set()
    ordered_block_reasons: List[str] = []
    for reason in block_reasons:
        if reason in seen_reason:
            continue
        seen_reason.add(reason)
        ordered_block_reasons.append(reason)

    decision = "PASS" if not ordered_block_reasons else "BLOCK"

    return {
        "schema": "clawd.source_of_truth_map_guard.v1",
        "checker_id": CHECKER_ID,
        "checker_version": CHECKER_VERSION,
        "generated_at": now_iso(),
        "decision": decision,
        "block_reason": ordered_block_reasons[0] if ordered_block_reasons else None,
        "block_reasons": ordered_block_reasons,
        "map_path": map_path_rel,
        "policy_pack_path": policy_pack_rel,
        "policy_schema_path": policy_schema_rel,
        "waiver_register_path": waiver_path_rel,
        "waiver_schema_path": waiver_schema_path_rel,
        "drift_output_path": drift_output_rel,
        "active_registry": active,
        "historical_registry": historical,
        "checks": checks,
        "failures": failures,
        "policy_summary": policy_summary,
        "waiver_summary": waiver_summary,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Deterministic source-of-truth map anti-drift guard")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT), help="Repository root")
    ap.add_argument("--map-path", default=DEFAULT_MAP_PATH, help="Source-of-truth map markdown path")
    ap.add_argument(
        "--policy-pack-path",
        default=DEFAULT_POLICY_PACK_PATH,
        help="Machine-readable anti-drift policy pack path",
    )
    ap.add_argument(
        "--policy-schema-path",
        default=DEFAULT_POLICY_SCHEMA_PATH,
        help="Policy pack schema path",
    )
    ap.add_argument(
        "--waiver-path",
        default=DEFAULT_WAIVER_PATH,
        help="Expiry-bounded anti-drift waiver register path",
    )
    ap.add_argument(
        "--waiver-schema-path",
        default=DEFAULT_WAIVER_SCHEMA_PATH,
        help="Waiver register schema path",
    )
    ap.add_argument(
        "--drift-output-path",
        default=DEFAULT_DRIFT_OUTPUT_PATH,
        help="Durable anti-drift artifact output path",
    )
    ap.add_argument("--json", action="store_true", help="Pretty-print JSON")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = resolve_repo_path(DEFAULT_REPO_ROOT, str(args.repo_root))
    map_path = resolve_repo_path(repo_root, str(args.map_path))
    policy_pack_path = resolve_repo_path(repo_root, str(args.policy_pack_path))
    policy_schema_path = resolve_repo_path(repo_root, str(args.policy_schema_path))
    waiver_path = resolve_repo_path(repo_root, str(args.waiver_path))
    waiver_schema_path = resolve_repo_path(repo_root, str(args.waiver_schema_path))
    drift_output_path = resolve_repo_path(repo_root, str(args.drift_output_path))

    payload = evaluate(
        repo_root=repo_root,
        map_path=map_path,
        waiver_path=waiver_path,
        waiver_schema_path=waiver_schema_path,
        policy_pack_path=policy_pack_path,
        policy_schema_path=policy_schema_path,
        drift_output_path=drift_output_path,
    )

    try:
        atomic_write_json(drift_output_path, payload)
    except Exception as exc:
        failures = list(payload.get("failures") or [])
        failures.append(
            {
                "reason_id": "drift_marker_write_failed",
                "check_id": "drift_artifact_written",
                "detail": str(exc),
                "waived": False,
                "waived_by": None,
            }
        )
        payload["failures"] = failures

        reasons = list(payload.get("block_reasons") or [])
        if "drift_marker_write_failed" not in reasons:
            reasons.append("drift_marker_write_failed")
        payload["block_reasons"] = reasons
        payload["block_reason"] = reasons[0] if reasons else "drift_marker_write_failed"
        payload["decision"] = "BLOCK"

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    return 0 if payload.get("decision") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
