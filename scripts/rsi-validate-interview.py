#!/usr/bin/env python3
"""Validate one RSI interview against structured audit evidence.

The default operation is read-only. Invalid reports exit 1. When
``--grill-output`` is supplied, an invalid report also produces a ready-to-run
query file using the existing RSI grill prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rsi_interview import DETAIL_FIELDS, parse_model_json, required_ids

STORE = Path.home() / ".hermes" / "rsi"
AUDIT = STORE / "audit" / "latest.json"
GRILL_PROMPT = STORE / "grill-prompt.txt"
REQUIRED_LIST_FIELDS = (
    "autonomous_failures",
    "incomplete_tasks",
    "incidents",
    "correction_feedback",
    "accounted_session_ids",
)


def _detail_record_matches(field: str, record: Any, required_id: str) -> bool:
    """The documented schema requires an exact ``id`` in both categories."""
    return isinstance(record, dict) and record.get("id") == required_id


def validate_interview(
    profile: str,
    report: Any,
    audit: dict[str, Any],
    *,
    merge_conflicts: list[str] | None = None,
) -> dict[str, Any]:
    required = required_ids(profile, audit)
    errors: list[str] = []
    missing_qualitative: list[str] = []
    profiles = audit.get("profiles")
    if not isinstance(profiles, dict) or not isinstance(profiles.get(profile), dict):
        errors.append("audit has no structured profile slice")
    missing_accounted = list(required["all"])
    missing_detail = {
        "autonomous_failures": list(required["autonomous_failures"]),
        "incomplete_tasks": list(required["incomplete_tasks"]),
    }

    if not isinstance(report, dict):
        errors.append("interview JSON must be an object")
    else:
        if report.get("profile") != profile:
            errors.append(f"profile must equal {profile!r}")

        for field in REQUIRED_LIST_FIELDS:
            if not isinstance(report.get(field), list):
                errors.append(f"{field} must be a list")

        accounted = report.get("accounted_session_ids")
        if isinstance(accounted, list):
            exact_accounted = {item for item in accounted if isinstance(item, str)}
            missing_accounted = [item for item in required["all"] if item not in exact_accounted]

        for field in ("autonomous_failures", "incomplete_tasks"):
            records = report.get(field)
            if isinstance(records, list):
                missing_detail[field] = [
                    item
                    for item in required[field]
                    if not any(
                        _detail_record_matches(field, record, item)
                        for record in records
                    )
                ]
                seen_ids: set[str] = set()
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        errors.append(f"{field}[{index}] must be an object")
                        continue
                    item_id = record.get("id")
                    if not isinstance(item_id, str) or not item_id.strip():
                        errors.append(f"{field}[{index}].id must be a non-empty exact ID")
                        continue
                    if item_id in seen_ids:
                        errors.append(f"{field} contains duplicate exact ID: {item_id}")
                    seen_ids.add(item_id)
                    missing_fields = [
                        key
                        for key in DETAIL_FIELDS[field]
                        if not isinstance(record.get(key), str) or not record[key].strip()
                    ]
                    if missing_fields:
                        missing_qualitative.append(item_id)
                        errors.append(
                            f"{field} id={item_id} missing qualitative fields: "
                            + ", ".join(missing_fields)
                        )

    for conflict in merge_conflicts or []:
        errors.append(f"scaffold merge contradiction: {conflict}")

    if missing_accounted:
        errors.append(
            "accounted_session_ids missing exact audited IDs: "
            + ", ".join(missing_accounted)
        )
    for field in ("autonomous_failures", "incomplete_tasks"):
        if missing_detail[field]:
            errors.append(
                f"{field} missing literal audited IDs: "
                + ", ".join(missing_detail[field])
            )

    return {
        "valid": not errors,
        "profile": profile,
        "required_ids": required["all"],
        "missing_accounted_ids": missing_accounted,
        "missing_detail_ids": missing_detail,
        "missing_qualitative_ids": list(dict.fromkeys(missing_qualitative)),
        "errors": errors,
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = parse_model_json(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"malformed {label} JSON: top level must be an object")
    return value


def _write_grill(path: Path, base_path: Path, result: dict[str, Any]) -> None:
    try:
        base = base_path.read_text(encoding="utf-8").rstrip()
    except OSError:
        base = "RSI audited your chats since the last tick. Your interview did not match that evidence."
    mismatch = json.dumps(
        {
            "profile": result["profile"],
            "validation_errors": result["errors"],
            "required_ids": result["required_ids"],
        },
        sort_keys=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{base}\n\nDETERMINISTIC VALIDATION MISMATCHES:\n{mismatch}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("report", type=Path)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--grill-prompt", type=Path, default=GRILL_PROMPT)
    parser.add_argument("--grill-output", type=Path)
    args = parser.parse_args(argv)

    try:
        audit = _load_object(args.audit, "audit")
    except ValueError as exc:
        result = {
            "valid": False,
            "profile": args.profile,
            "required_ids": [],
            "missing_accounted_ids": [],
            "missing_detail_ids": {
                "autonomous_failures": [],
                "incomplete_tasks": [],
            },
            "errors": [str(exc)],
        }
    else:
        try:
            report = _load_object(args.report, "interview")
        except ValueError as exc:
            required = required_ids(args.profile, audit)
            result = {
                "valid": False,
                "profile": args.profile,
                "required_ids": required["all"],
                "missing_accounted_ids": required["all"],
                "missing_detail_ids": {
                    "autonomous_failures": required["autonomous_failures"],
                    "incomplete_tasks": required["incomplete_tasks"],
                },
                "errors": [str(exc)],
            }
        else:
            result = validate_interview(args.profile, report, audit)

    if not result["valid"] and args.grill_output is not None:
        _write_grill(args.grill_output, args.grill_prompt, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
