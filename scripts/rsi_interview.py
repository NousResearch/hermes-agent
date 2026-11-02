#!/usr/bin/env python3
"""Evidence-derived scaffolding and merge rules for RSI interviews."""
from __future__ import annotations

import json
import re
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
    omitted_admission_ids: list[str] = []


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?[ \t]*```", re.DOTALL)


def parse_model_json(text: str) -> Any:
    """Parse a model response as JSON, tolerating a markdown fence wrapper.

    Models wrap JSON in ``` fences (coder/QA-style) or emit it bare
    (reviewer-style). A fence is only stripped when the bare text is not
    itself valid JSON, so malformed fenced output still fails loudly.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _FENCE_RE.search(stripped)
    if match:
        return json.loads(match.group(1))
    raise json.JSONDecodeError("no JSON object found", stripped[:80], 0)


def looks_like_grill_admissions(model: Any) -> bool:
    """True for the documented grill-admission schema, not the full report."""
    return (
        isinstance(model, dict)
        and isinstance(model.get("admissions"), list)
        and "autonomous_failures" not in model
        and "incomplete_tasks" not in model
    )


def apply_grill_admissions(
    prior_report: dict[str, Any],
    admissions_report: dict[str, Any],
    audit: dict[str, Any],
) -> MergeResult:
    """Merge a grill-admission response into the prior validated report.

    Audit-owned invariants are preserved exactly as in ``merge_interview``:
    IDs and category membership come from the scaffold classifier, and model
    prose can only update qualitative fields on rows that already exist in
    the prior report. An admission whose ``id`` names no prior detail row is
    never allowed to add or reclassify a row; if it names an existing
    ``correction_feedback`` id it updates that entry's qualitative fields,
    and otherwise it is surfaced as a merge conflict so validation keeps
    grilling instead of silently accepting it.

    Field mapping per admission: ``what_happened`` -> ``summary``,
    ``why_misreported`` -> ``evidence`` (autonomous failures) or
    ``why_incomplete`` (incomplete tasks). The admission-level
    ``reporting_sentence`` and ``suggested_fix`` describe the profile's
    reporting behavior, not any single audited row, so they are never
    written into per-row records.
    """
    conflicts: list[str] = []
    omitted: list[str] = []
    profile = str(prior_report.get("profile") or "")
    required = required_ids(profile, audit)

    report = {
        "profile": profile,
        "accounted_session_ids": list(prior_report.get("accounted_session_ids") or []),
    }
    for field in (
        "autonomous_failures",
        "incomplete_tasks",
        "incidents",
        "correction_feedback",
    ):
        report[field] = [dict(row) if isinstance(row, dict) else row
                         for row in (prior_report.get(field) or [])]

    by_id: dict[str, dict[str, Any]] = {}
    feedback_by_id: dict[str, dict[str, Any]] = {}
    for field in ("autonomous_failures", "incomplete_tasks"):
        for row in report[field]:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                by_id.setdefault(row["id"], row)
    for row in report["correction_feedback"]:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            feedback_by_id.setdefault(row["id"], row)

    admissions = admissions_report.get("admissions")
    for admission in admissions if isinstance(admissions, list) else []:
        if not isinstance(admission, dict):
            continue
        item_id = str(admission.get("id") or "").strip()
        target = by_id.get(item_id)
        if target is None:
            # A correction-id admission (e.g. "was c-020 still happening?")
            # answers a qualitative question, not an audited detail row.
            feedback_row = feedback_by_id.get(item_id)
            if feedback_row is not None:
                what = admission.get("what_happened")
                if isinstance(what, str) and what.strip():
                    feedback_row["evidence"] = what.strip()
                why = admission.get("why_misreported")
                if isinstance(why, str) and why.strip():
                    feedback_row["correction"] = why.strip()
                continue
            if item_id:
                # A non-audited id (e.g. the real 2026-09-05 coder payload's
                # "interview" meta-admission) cannot add or reclassify an
                # audit-owned row. Safe omission: recorded in the merge
                # result so it is visible, but never a validation failure.
                omitted.append(item_id)
            continue
        what = admission.get("what_happened")
        if isinstance(what, str) and what.strip():
            target["summary"] = what.strip()
        why = admission.get("why_misreported")
        if isinstance(why, str) and why.strip():
            if "evidence" in target:
                target["evidence"] = why.strip()
            elif "why_incomplete" in target:
                target["why_incomplete"] = why.strip()
        fix = admission.get("suggested_fix")
        # The admission-level suggested_fix is a reporting/SOUL proposal, not
        # a per-row fix, but an empty row fix is exactly the gap grilling
        # exists to close — fill it, never overwrite a prior row's fix.
        if (
            isinstance(fix, str)
            and fix.strip()
            and not _nonempty(target.get("suggested_fix"))
        ):
            target["suggested_fix"] = fix.strip()

    # The documented grill schema carries suggested_fix ONLY at top level
    # (the real 2026-09-05 coder/QA/reviewer payloads had no per-row fix),
    # so per-row filling alone left every failed initial scaffold's
    # autonomous-failure fixes empty. Fill each still-empty mandatory row
    # fix from the top-level proposal — never overwrite a prior row's fix.
    top_fix = admissions_report.get("suggested_fix")
    if isinstance(top_fix, str) and top_fix.strip():
        top_fix = top_fix.strip()
        for field in ("autonomous_failures", "incomplete_tasks"):
            mandatory = set(required[field])
            for row in report[field]:
                if (
                    isinstance(row, dict)
                    and row.get("id") in mandatory
                    and not _nonempty(row.get("suggested_fix"))
                ):
                    row["suggested_fix"] = top_fix

    feedback = admissions_report.get("correction_feedback")
    if isinstance(feedback, list):
        existing_feedback: dict[str, dict[str, Any]] = {}
        for row in report["correction_feedback"]:
            if isinstance(row, dict) and isinstance(row.get("id"), str):
                existing_feedback.setdefault(row["id"], row)
        for row in feedback:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("id") or "").strip()
            if row_id and row_id in existing_feedback:
                existing_feedback[row_id].update(row)
            else:
                report["correction_feedback"].append(row)

    missing: list[str] = []
    for field in ("autonomous_failures", "incomplete_tasks"):
        mandatory = set(required[field])
        for row in report[field]:
            if not isinstance(row, dict) or row.get("id") not in mandatory:
                continue
            if not all(_nonempty(row.get(key)) for key in DETAIL_FIELDS[field]):
                missing.append(str(row["id"]))

    return MergeResult(
        report=report,
        conflicts=conflicts,
        missing_qualitative_ids=_ordered_unique(missing),
        omitted_admission_ids=_ordered_unique(omitted),
    )


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
