"""Mechanical read model for model-authored Canonical Brain events.

The projector never inspects free-form text and never assigns business meaning.
It folds explicit event types and structured fields into current case state.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


ROUTE_BACK_EVENT_TYPES = frozenset({
    "route_back.required",
    "route_back.intent.created",
    "route_back.sent",
    "route_back.blocked",
})
ROUTE_BACK_TERMINAL_TYPES = frozenset({"route_back.sent", "route_back.blocked"})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _event_sort_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row.get("occurred_at") or ""), str(row.get("event_id") or ""))


def fold_case_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fold exact structured events into latest per-case state."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id") or "").strip()
        if case_id:
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
                str(refs.get("message_id") or refs.get("event_ref") or ""),
            )
            if any(key) and key not in seen_refs:
                seen_refs.add(key)
                source_refs.append(dict(refs))

        latest_status = dict(_mapping(latest.get("status")))
        latest_payload = _mapping(latest.get("payload"))
        summary = str(
            latest_status.get("summary")
            or latest_payload.get("summary")
            or ""
        )[:500]
        route_type = str(latest_route.get("event_type") or "") if latest_route else None
        projections.append({
            "case_id": case_id,
            "event_count": len(ordered),
            "latest_event_id": str(latest.get("event_id") or ""),
            "latest_event_type": str(latest.get("event_type") or ""),
            "latest_event_at": str(latest.get("occurred_at") or ""),
            "status": latest_status,
            "summary": summary,
            "next_action": dict(_mapping(latest.get("next_action"))),
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
            "source_refs": source_refs,
        })

    return sorted(
        projections,
        key=lambda item: (item["latest_event_at"], item["case_id"]),
        reverse=True,
    )


__all__ = ["fold_case_events", "ROUTE_BACK_EVENT_TYPES", "ROUTE_BACK_TERMINAL_TYPES"]
