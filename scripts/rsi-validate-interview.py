#!/usr/bin/env python3
"""Validate one RSI interview against structured audit evidence.

The default operation is read-only. Invalid reports exit 1. When
``--grill-output`` is supplied, an invalid report also produces a ready-to-run
query file using the existing RSI grill prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

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


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _profile_evidence(profile: str, audit: dict[str, Any]) -> dict[str, Any]:
    profiles = audit.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    evidence = profiles.get(profile)
    return evidence if isinstance(evidence, dict) else {}


def required_ids(profile: str, audit: dict[str, Any]) -> dict[str, list[str]]:
    """Return exact audited IDs grouped by the report field that must cite them.

    Classification is based only on the audit's structured collections and
    lifecycle markers. It deliberately does not inspect titles, summaries,
    errors, or other free text.
    """
    evidence = _profile_evidence(profile, audit)
    autonomous: list[str] = []
    incomplete: list[str] = []

    sessions = evidence.get("session_failures")
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("id") or "").strip()
            hits = item.get("fail_hits")
            lifecycle_needs_input = (
                isinstance(hits, list)
                and "lifecycle:needs_input" in {str(hit) for hit in hits}
            )
            (incomplete if lifecycle_needs_input else autonomous).append(session_id)

    cron_failures = evidence.get("cron_failures")
    if isinstance(cron_failures, list):
        for item in cron_failures:
            if isinstance(item, dict):
                autonomous.append(str(item.get("execution_id") or "").strip())

    kanban_failures = evidence.get("kanban_failures")
    if isinstance(kanban_failures, list):
        for item in kanban_failures:
            if isinstance(item, dict):
                incomplete.append(str(item.get("task_id") or "").strip())

    autonomous = _ordered_unique(autonomous)
    incomplete = _ordered_unique(incomplete)
    return {
        "autonomous_failures": autonomous,
        "incomplete_tasks": incomplete,
        "all": _ordered_unique([*autonomous, *incomplete]),
    }


def _contains_standalone_id(value: Any, required_id: str) -> bool:
    if isinstance(value, str):
        pattern = rf"(?<![A-Za-z0-9_-]){re.escape(required_id)}(?![A-Za-z0-9_-])"
        return re.search(pattern, value) is not None
    if isinstance(value, list):
        return any(_contains_standalone_id(item, required_id) for item in value)
    if isinstance(value, dict):
        return any(_contains_standalone_id(item, required_id) for item in value.values())
    return False


def _detail_record_matches(field: str, record: Any, required_id: str) -> bool:
    if not isinstance(record, dict):
        return False
    if field == "incomplete_tasks":
        return record.get("id") == required_id
    # autonomous_failures records carry only summary/evidence/suggested_fix,
    # so a standalone exact ID in any documented free-text field (including
    # ``summary``) counts. The standalone boundary still rejects
    # suffix/prefixed near-matches.
    return (
        record.get("id") == required_id
        or record.get("execution_id") == required_id
        or _contains_standalone_id(record.get("evidence"), required_id)
        or _contains_standalone_id(record.get("summary"), required_id)
    )


def validate_interview(
    profile: str,
    report: Any,
    audit: dict[str, Any],
) -> dict[str, Any]:
    required = required_ids(profile, audit)
    errors: list[str] = []
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
        "errors": errors,
    }


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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
