"""Thin Gmail/Calendar action wrappers with write-action confirmation gates."""

from __future__ import annotations

import json
from typing import Any

from agent.local_secretary.write_action_gate import check_write_action


def gmail_search(query: str, *, confirmed: bool = False) -> str:
    gate = check_write_action("gmail_search", confirmed=confirmed)
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "gmail_search",
            "query": query,
            "messages": [],
            "note": "read-only search stub; wire to google-workspace skill for live calls",
        }
    )


def gmail_send(
    to: str,
    subject: str,
    body: str,
    *,
    confirmed: bool = False,
) -> str:
    gate = check_write_action(
        "gmail_send",
        confirmed=confirmed,
        detail=f"to={to} subject={subject}",
    )
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "gmail_send",
            "to": to,
            "subject": subject,
            "sent": True,
        }
    )


def calendar_list(*, range_hint: str = "today", confirmed: bool = False) -> str:
    gate = check_write_action("calendar_list", confirmed=confirmed)
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "calendar_list",
            "range": range_hint,
            "timezone": "Asia/Tokyo",
            "events": [],
        }
    )


def calendar_create(
    summary: str,
    start: str,
    end: str,
    *,
    confirmed: bool = False,
) -> str:
    gate = check_write_action(
        "calendar_create",
        confirmed=confirmed,
        detail=f"summary={summary} start={start} end={end}",
    )
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "calendar_create",
            "summary": summary,
            "start": start,
            "end": end,
        }
    )


def calendar_update(event_id: str, **fields: Any) -> str:
    confirmed = bool(fields.pop("confirmed", False))
    gate = check_write_action(
        "calendar_update",
        confirmed=confirmed,
        detail=f"event_id={event_id}",
    )
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "calendar_update",
            "event_id": event_id,
            "fields": fields,
        }
    )


def calendar_delete(event_id: str, *, confirmed: bool = False) -> str:
    gate = check_write_action(
        "calendar_delete",
        confirmed=confirmed,
        detail=f"event_id={event_id}",
    )
    if not gate.ok:
        return json.dumps(gate.to_json())
    return json.dumps(
        {
            "success": True,
            "action": "calendar_delete",
            "event_id": event_id,
            "deleted": True,
        }
    )
