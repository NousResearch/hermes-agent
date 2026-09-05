#!/usr/bin/env python3
"""Evidence-derived scaffolding and merge rules for RSI interviews."""
from __future__ import annotations

from typing import Any, NamedTuple


REPORT_LIST_FIELDS = (
    "autonomous_failures",
    "incomplete_tasks",
    "incidents",
    "correction_feedback",
    "accounted_session_ids",
)
DETAIL_FIELDS = {
    "autonomous_failures": ("summary", "evidence", "suggested_fix"),
    "incomplete_tasks": ("title", "summary", "why_incomplete"),
}


class MergeResult(NamedTuple):
    report: dict[str, Any]
    conflicts: list[str]
    missing_qualitative_ids: list[str]


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _profile_evidence(profile: str, audit: dict[str, Any]) -> dict[str, Any]:
    profiles = audit.get("profiles")
    if not isinstance(profiles, dict):
        return {}
    evidence = profiles.get(profile)
    return evidence if isinstance(evidence, dict) else {}


def required_ids(profile: str, audit: dict[str, Any]) -> dict[str, list[str]]:
    """Classify exact IDs using structured audit lifecycle fields only."""
    evidence = _profile_evidence(profile, audit)
    autonomous: list[str] = []
    incomplete: list[str] = []

    sessions = evidence.get("session_failures")
    if isinstance(sessions, list):
        for item in sessions:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("id") or "").strip()
            if not session_id:
                continue
            # Every structurally failed session is an autonomy failure. A
            # needs-input lifecycle marker additionally makes the same run an
            # incomplete task; the two categories are intentionally not
            # mutually exclusive.
            autonomous.append(session_id)
            hits = item.get("fail_hits")
            if isinstance(hits, list) and "lifecycle:needs_input" in {
                str(hit) for hit in hits
            }:
                incomplete.append(session_id)

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


def _items_by_id(items: Any, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get(key) or "").strip()
        if item_id and item_id not in result:
            result[item_id] = item
    return result


def build_scaffold(profile: str, audit: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable ID/category skeleton for one profile report."""
    required = required_ids(profile, audit)
    evidence = _profile_evidence(profile, audit)
    sessions = _items_by_id(evidence.get("session_failures"), "id")
    cron = _items_by_id(evidence.get("cron_failures"), "execution_id")
    kanban = _items_by_id(evidence.get("kanban_failures"), "task_id")

    autonomous = []
    for item_id in required["autonomous_failures"]:
        source = sessions.get(item_id) or cron.get(item_id) or {}
        source_name = str(source.get("source") or source.get("name") or "audited run")
        autonomous.append(
            {
                "id": item_id,
                "summary": "",
                "evidence": "",
                "suggested_fix": "",
                "audit_source": source_name,
            }
        )

    incomplete = []
    for item_id in required["incomplete_tasks"]:
        source = sessions.get(item_id) or kanban.get(item_id) or {}
        incomplete.append(
            {
                "id": item_id,
                "title": str(source.get("title") or ""),
                "summary": "",
                "why_incomplete": "",
            }
        )

    return {
        "profile": profile,
        "autonomous_failures": autonomous,
        "incomplete_tasks": incomplete,
        "incidents": [],
        "correction_feedback": [],
        "accounted_session_ids": list(required["all"]),
    }


def _merge_record_values(
    field: str,
    item_id: str,
    target: dict[str, Any],
    incoming: dict[str, Any],
    seen_values: dict[str, str],
    conflicts: list[str],
) -> None:
    for key in DETAIL_FIELDS[field]:
        value = incoming.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        value = value.strip()
        prior = seen_values.get(key)
        if prior is not None and prior != value:
            message = f"{field} id={item_id} has conflicting {key} values"
            if message not in conflicts:
                conflicts.append(message)
            continue
        seen_values[key] = value
        target[key] = value


def _merge_detail_field(
    field: str,
    scaffold_records: list[dict[str, Any]],
    model_records: Any,
    conflicts: list[str],
    forbidden_ids: set[str],
) -> list[dict[str, Any]]:
    merged = [dict(item) for item in scaffold_records]
    by_id = {item["id"]: item for item in merged}
    seen: dict[str, dict[str, str]] = {item["id"]: {} for item in merged}

    if not isinstance(model_records, list):
        return merged
    for record in model_records:
        if not isinstance(record, dict):
            continue
        item_id = str(record.get("id") or "").strip()
        if item_id in forbidden_ids:
            # An audited ID belongs only to the categories selected by the
            # structured lifecycle classifier. Model prose cannot reclassify it.
            continue
        if not item_id:
            # Preserve malformed model output so validation can grill it; it
            # cannot satisfy or alter any mandatory scaffold record.
            merged.append(dict(record))
            continue
        if item_id not in by_id:
            target = {"id": item_id}
            merged.append(target)
            by_id[item_id] = target
            seen[item_id] = {}
        _merge_record_values(
            field,
            item_id,
            by_id[item_id],
            record,
            seen[item_id],
            conflicts,
        )
    return merged


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def merge_interview(
    profile: str,
    model_report: Any,
    audit: dict[str, Any],
) -> MergeResult:
    """Merge model enrichment into an audit-owned immutable scaffold."""
    scaffold = build_scaffold(profile, audit)
    model = model_report if isinstance(model_report, dict) else {}
    conflicts: list[str] = []
    required = required_ids(profile, audit)

    report = dict(scaffold)
    for field in ("autonomous_failures", "incomplete_tasks"):
        report[field] = _merge_detail_field(
            field,
            scaffold[field],
            model.get(field),
            conflicts,
            set(required["all"]) - set(required[field]),
        )

    for field in ("incidents", "correction_feedback"):
        values = model.get(field)
        report[field] = list(values) if isinstance(values, list) else []

    model_accounted = model.get("accounted_session_ids")
    extras = [item for item in model_accounted if isinstance(item, str)] if isinstance(model_accounted, list) else []
    report["accounted_session_ids"] = _ordered_unique(
        [*scaffold["accounted_session_ids"], *extras]
    )

    missing: list[str] = []
    for field in ("autonomous_failures", "incomplete_tasks"):
        mandatory = set(required[field])
        for record in report[field]:
            if not isinstance(record, dict) or record.get("id") not in mandatory:
                continue
            if not all(_nonempty(record.get(key)) for key in DETAIL_FIELDS[field]):
                missing.append(str(record["id"]))

    return MergeResult(
        report=report,
        conflicts=conflicts,
        missing_qualitative_ids=_ordered_unique(missing),
    )
