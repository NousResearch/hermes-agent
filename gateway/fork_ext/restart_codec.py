"""Pure restart-loop failure entry codec helpers."""

from __future__ import annotations

from typing import Any


def decode_restart_failure_entry(value: Any) -> dict:
    if isinstance(value, dict):
        try:
            count = int(value.get("count", 0) or 0)
        except (TypeError, ValueError):
            count = 0
        marks = value.get("replay_marks", [])
        if not isinstance(marks, list):
            marks = []
        clean_marks = []
        for mark in marks:
            try:
                clean_marks.append(float(mark))
            except (TypeError, ValueError):
                continue
        request_ids = value.get("replay_request_ids", [])
        if not isinstance(request_ids, list):
            request_ids = []
        return {
            "count": max(0, count),
            "replay_marks": clean_marks,
            "replay_request_ids": [str(item) for item in request_ids if item],
            "armed": bool(value.get("armed", False)),
        }
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        count = 0
    return {
        "count": max(0, count),
        "replay_marks": [],
        "replay_request_ids": [],
        "armed": False,
    }


def encode_restart_failure_entry(entry: dict) -> Any:
    count = int(entry.get("count", 0) or 0)
    replay_marks = entry.get("replay_marks") or []
    replay_request_ids = entry.get("replay_request_ids") or []
    armed = bool(entry.get("armed", False))
    if replay_marks or replay_request_ids or armed:
        return {
            "count": count,
            "replay_marks": replay_marks,
            "replay_request_ids": replay_request_ids,
            "armed": armed,
        }
    return count
