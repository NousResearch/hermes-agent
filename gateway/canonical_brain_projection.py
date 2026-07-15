"""Mechanical read model for model-authored Canonical Brain events.

The projector never inspects free-form text for business meaning.  It folds
explicit event types and structured fields into a bounded current-case view.
Hermes/GPT authors task plans, decisions, and verification judgments; this
module only preserves their latest explicit state and runtime-attested
receipts.
"""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any, Iterable, Mapping


ROUTE_BACK_EVENT_TYPES = frozenset({
    "route_back.required",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
})
ROUTE_BACK_TERMINAL_TYPES = frozenset({"route_back.sent", "route_back.blocked"})
TASK_PLAN_EVENT_TYPE = "task.plan.updated"
TASK_VERIFICATION_EVENT_TYPE = "task.verification.recorded"
APPROVAL_RECEIPT_EVENT_TYPE = "approval.capability.recorded"
CAPABILITY_CHECK_EVENT_TYPE = "capability.check.recorded"
DEFAULT_TIMELINE_LIMIT = 20
MAX_TIMELINE_LIMIT = 50
MAX_WORKSPACE_RECEIPTS = 80
MAX_CAPABILITY_RECEIPTS = 64


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return []
        return decoded if isinstance(decoded, list) else []
    return []


def _event_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("occurred_at") or ""), str(row.get("event_id") or ""))


def _last_explicit_mapping(
    ordered: list[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    """Return the last non-empty explicit mapping for ``field``.

    Canonical append rows use ``{}`` when a field is omitted.  A later note
    must therefore not erase the last real status/next action.  Callers that
    need to clear a value record an explicit sentinel such as
    ``{"kind": "none"}``, which remains non-empty and becomes current.
    """
    for row in reversed(ordered):
        value = _mapping(row.get(field))
        if value:
            return dict(value)
    return {}


def _last_explicit_sequence(
    ordered: list[Mapping[str, Any]],
    field: str,
) -> list[Any]:
    for row in reversed(ordered):
        value = _sequence(row.get(field))
        if value:
            return list(value)
    return []


def _last_summary(ordered: list[Mapping[str, Any]]) -> str:
    for row in reversed(ordered):
        status = _mapping(row.get("status"))
        payload = _mapping(row.get("payload"))
        summary = status.get("summary") or payload.get("summary")
        if summary:
            return str(summary)[:500]
    return ""


def _receipt_entry(row: Mapping[str, Any], payload_key: str) -> dict[str, Any]:
    payload = _mapping(row.get("payload"))
    value = _mapping(payload.get(payload_key))
    if not value:
        return {}
    return {
        **dict(value),
        "event_id": str(row.get("event_id") or ""),
        "occurred_at": str(row.get("occurred_at") or ""),
        # Task verification and approval/capability process receipts remain
        # model/audit information.  Only route_back.sent can carry a
        # deterministic external-delivery attestation.
        "runtime_attested": False,
    }


def _plan_revision(row: Mapping[str, Any]) -> int:
    """Return a sortable numeric plan revision, or ``-1`` for legacy rows."""
    plan = _mapping(_mapping(row.get("payload")).get("plan"))
    revision = plan.get("revision")
    if isinstance(revision, bool):
        return -1
    if isinstance(revision, int):
        return revision
    if isinstance(revision, str) and revision.strip().isdigit():
        return int(revision.strip())
    return -1


def _remaining_uses(entry: Mapping[str, Any]) -> int | None:
    remaining = entry.get("remaining_uses_for_command", entry.get("remaining_uses"))
    if isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 0:
        return None
    return remaining


def select_canonical_plan_head(
    plan_rows: list[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Select the unique mechanical head of a plan revision graph.

    Revisions are compared only within one ``plan_id``.  Explicit
    ``supersedes_plan_id`` edges then identify the unique plan lineage head.
    Wall-clock and UUID order are used only as a duplicate-revision fallback;
    they never decide between revisions or supersession branches.
    """
    latest_by_plan_id: dict[str, Mapping[str, Any]] = {}
    edges_by_plan_id: dict[str, set[tuple[str, int | None]]] = defaultdict(set)
    content_by_plan_revision: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in plan_rows:
        plan = _mapping(_mapping(row.get("payload")).get("plan"))
        plan_id = str(plan.get("plan_id") or "").strip()
        if not plan_id:
            return None, "task_plan_missing_plan_id"
        revision = _plan_revision(row)
        if revision >= 0:
            content_by_plan_revision[(plan_id, revision)].add(
                json.dumps(
                    dict(plan),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
        current = latest_by_plan_id.get(plan_id)
        if current is None or (
            _plan_revision(row), _event_sort_key(row)
        ) >= (
            _plan_revision(current), _event_sort_key(current)
        ):
            latest_by_plan_id[plan_id] = row

        supersedes_plan_id = str(plan.get("supersedes_plan_id") or "").strip()
        if not supersedes_plan_id:
            continue
        raw_revision = plan.get("supersedes_plan_revision")
        supersedes_revision = (
            raw_revision
            if isinstance(raw_revision, int) and not isinstance(raw_revision, bool)
            else None
        )
        edges_by_plan_id[plan_id].add((supersedes_plan_id, supersedes_revision))

    if not latest_by_plan_id:
        return None, None
    for (plan_id, revision), contents in content_by_plan_revision.items():
        if len(contents) > 1:
            return None, f"task_plan_revision_content_conflict:{plan_id}:{revision}"
    for plan_id, edges in edges_by_plan_id.items():
        if len(edges) > 1:
            return None, f"task_plan_supersession_changed:{plan_id}"
        predecessor_id, predecessor_revision = next(iter(edges))
        if predecessor_id == plan_id:
            return None, f"task_plan_supersession_cycle:{plan_id}"
        if predecessor_id in latest_by_plan_id:
            selected_predecessor_revision = _plan_revision(
                latest_by_plan_id[predecessor_id]
            )
            if (
                predecessor_revision is None
                or predecessor_revision < 1
                or predecessor_revision != selected_predecessor_revision
            ):
                return None, (
                    "task_plan_supersession_revision_mismatch:"
                    f"{plan_id}:{predecessor_id}"
                )

    superseded_plan_ids = {
        predecessor_id
        for edges in edges_by_plan_id.values()
        for predecessor_id, _ in edges
        if predecessor_id in latest_by_plan_id
    }
    head_ids = sorted(set(latest_by_plan_id) - superseded_plan_ids)
    if len(head_ids) != 1:
        return None, (
            "task_plan_graph_has_no_head"
            if not head_ids
            else "task_plan_graph_has_multiple_heads:" + ",".join(head_ids)
        )
    return latest_by_plan_id[head_ids[0]], None


def _select_latest_plan_row(
    plan_rows: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    row, _ = select_canonical_plan_head(plan_rows)
    return row


def _timeline_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    status = _mapping(row.get("status"))
    payload = _mapping(row.get("payload"))
    return {
        "event_id": str(row.get("event_id") or ""),
        "event_type": str(row.get("event_type") or ""),
        "occurred_at": str(row.get("occurred_at") or ""),
        "summary": str(status.get("summary") or payload.get("summary") or "")[:500],
        "status": dict(status),
        "next_action": dict(_mapping(row.get("next_action"))),
    }


def _task_workspace(ordered: list[Mapping[str, Any]]) -> dict[str, Any]:
    plan_rows = [row for row in ordered if row.get("event_type") == TASK_PLAN_EVENT_TYPE]
    latest_plan_row, plan_projection_error = select_canonical_plan_head(plan_rows)
    plan = dict(_mapping(_mapping(latest_plan_row.get("payload") if latest_plan_row else {}).get("plan")))

    all_verifications = [
        entry
        for entry in (
            _receipt_entry(row, "verification")
            for row in ordered
            if row.get("event_type") == TASK_VERIFICATION_EVENT_TYPE
        )
        if entry
    ]
    approvals = [
        entry
        for entry in (
            _receipt_entry(row, "approval_receipt")
            for row in ordered
            if row.get("event_type") == APPROVAL_RECEIPT_EVENT_TYPE
        )
        if entry
    ][-MAX_WORKSPACE_RECEIPTS:]
    capability_check_events = [
        entry
        for entry in (
            _receipt_entry(row, "capability_receipt")
            for row in ordered
            if row.get("event_type") == CAPABILITY_CHECK_EVENT_TYPE
        )
        if entry
    ]
    minimum_capability_checks: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in capability_check_events:
        key = (
            str(entry.get("approval_id") or ""),
            str(entry.get("command_sha256") or ""),
        )
        remaining = _remaining_uses(entry)
        if not all(key) or remaining is None:
            continue
        current = minimum_capability_checks.get(key)
        current_remaining = _remaining_uses(current or {})
        if current_remaining is None or remaining <= current_remaining:
            minimum_capability_checks[key] = entry
    capability_checks = sorted(
        minimum_capability_checks.values(),
        key=_event_sort_key,
    )[-MAX_CAPABILITY_RECEIPTS:]

    required_verification_ids = {
        str(value)
        for value in _sequence(plan.get("verification_event_ids"))
        if str(value).strip()
    }
    required_verifications = [
        item for item in all_verifications
        if str(item.get("event_id") or "") in required_verification_ids
    ]
    other_verifications = [
        item for item in all_verifications
        if str(item.get("event_id") or "") not in required_verification_ids
    ]
    verifications = sorted(
        required_verifications
        + other_verifications[-max(
            0,
            MAX_WORKSPACE_RECEIPTS - len(required_verifications),
        ):],
        key=_event_sort_key,
    )[-MAX_WORKSPACE_RECEIPTS:]
    passed_verification_ids = {
        item["event_id"]
        for item in verifications
        if str(item.get("outcome") or "") == "passed"
    }
    steps = [dict(item) for item in _sequence(plan.get("steps")) if isinstance(item, Mapping)]
    remaining_step_ids = [
        str(item.get("id") or "")
        for item in steps
        if item.get("id") and item.get("status") in {"pending", "in_progress", "blocked"}
    ]
    completion_receipts_satisfied = None
    if plan.get("state") == "completed":
        completion_receipts_satisfied = bool(required_verification_ids) and (
            required_verification_ids.issubset(passed_verification_ids)
        )

    return {
        "plan_event_id": str(latest_plan_row.get("event_id") or "") if latest_plan_row else None,
        "plan_event_at": str(latest_plan_row.get("occurred_at") or "") if latest_plan_row else None,
        "plan_projection_complete": plan_projection_error is None,
        "plan_projection_error": plan_projection_error,
        "plan": plan,
        "remaining_step_ids": remaining_step_ids,
        "verifications": verifications,
        "approvals": approvals,
        "capability_checks": capability_checks,
        "completion_receipts_satisfied": completion_receipts_satisfied,
        "missing_verification_event_ids": sorted(
            required_verification_ids - passed_verification_ids
        ),
    }


def fold_case_events(
    rows: Iterable[Mapping[str, Any]],
    *,
    timeline_limit: int = DEFAULT_TIMELINE_LIMIT,
) -> list[dict[str, Any]]:
    """Fold exact structured events into a bounded latest per-case state."""
    timeline_limit = max(1, min(int(timeline_limit), MAX_TIMELINE_LIMIT))
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_event_ids: set[str] = set()
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        event_id = str(row.get("event_id") or "").strip()
        if not case_id or (event_id and event_id in seen_event_ids):
            continue
        if event_id:
            seen_event_ids.add(event_id)
        grouped[case_id].append(row)

    projections: list[dict[str, Any]] = []
    for case_id, events in grouped.items():
        ordered = sorted(events, key=_event_sort_key)
        latest = ordered[-1]
        route_events = [row for row in ordered if row.get("event_type") in ROUTE_BACK_EVENT_TYPES]
        latest_route = route_events[-1] if route_events else None
        source_refs: list[dict[str, Any]] = []
        seen_refs: set[tuple[str, str, str]] = set()
        for row in ordered:
            refs = _mapping(_mapping(row.get("source")).get("source_refs"))
            key = (
                str(refs.get("platform") or ""),
                str(refs.get("thread_id") or refs.get("chat_id") or ""),
                str(refs.get("message_id") or refs.get("event_ref") or refs.get("manual_ref") or ""),
            )
            if any(key) and key not in seen_refs:
                seen_refs.add(key)
                source_refs.append(dict(refs))

        route_type = str(latest_route.get("event_type") or "") if latest_route else None
        timeline_rows = ordered[-timeline_limit:]
        timeline = [_timeline_entry(row) for row in timeline_rows]
        cursor = None
        if timeline:
            cursor = {
                "oldest_event_id": timeline[0]["event_id"],
                "oldest_occurred_at": timeline[0]["occurred_at"],
            }

        projections.append({
            "case_id": case_id,
            # Compatibility field: this is the count in the bounded query
            # window, never an assertion about all rows in Cloud SQL.
            "event_count": len(ordered),
            "window_event_count": len(ordered),
            "latest_event_id": str(latest.get("event_id") or ""),
            "latest_event_type": str(latest.get("event_type") or ""),
            "latest_event_at": str(latest.get("occurred_at") or ""),
            "status": _last_explicit_mapping(ordered, "status"),
            "summary": _last_summary(ordered),
            "next_action": _last_explicit_mapping(ordered, "next_action"),
            "actor": _last_explicit_mapping(ordered, "actor"),
            "subject": _last_explicit_mapping(ordered, "subject"),
            "evidence": _last_explicit_sequence(ordered, "evidence"),
            "decision": _last_explicit_mapping(ordered, "decision"),
            "safety": _last_explicit_mapping(ordered, "safety"),
            "route_back": {
                "latest_event_type": route_type,
                "terminal": route_type in ROUTE_BACK_TERMINAL_TYPES if route_type else False,
                "target_ref": dict(_mapping(
                    _mapping(
                        _mapping(latest_route.get("payload") if latest_route else {}).get("route_back")
                    ).get("target_ref")
                )),
                "receipt": dict(
                    _mapping(
                        _mapping(latest_route.get("payload") if latest_route else {}).get("receipt")
                    )
                ),
            },
            "workspace": _task_workspace(ordered),
            "timeline": timeline,
            "timeline_truncated": len(ordered) > len(timeline),
            "cursor": cursor,
            "source_refs": source_refs,
        })

    return sorted(
        projections,
        key=lambda item: (item["latest_event_at"], item["case_id"]),
        reverse=True,
    )


__all__ = [
    "fold_case_events",
    "ROUTE_BACK_EVENT_TYPES",
    "ROUTE_BACK_TERMINAL_TYPES",
    "TASK_PLAN_EVENT_TYPE",
    "TASK_VERIFICATION_EVENT_TYPE",
    "APPROVAL_RECEIPT_EVENT_TYPE",
    "CAPABILITY_CHECK_EVENT_TYPE",
    "select_canonical_plan_head",
]
