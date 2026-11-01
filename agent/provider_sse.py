"""Provider error/SSE text parsing shared by the streaming and non-streaming paths.

Moved out of ``chat_completion_helpers`` unchanged: pure text functions, no agent state.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def _status_code_from_value(value: Any) -> Optional[int]:
    if isinstance(value, int) and 100 <= value < 600:
        return value
    if not isinstance(value, str):
        return None
    match = re.search(r"(?:HTTP_STATUS/)?\b([1-5]\d\d)\b", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _parse_provider_sse_events(text: str) -> list[dict]:
    """Parse provider text that looks like Server-Sent Events."""
    events: list[dict] = []
    current = {"event": None, "data": [], "comments": [], "fields": {}}

    def _flush_current():
        nonlocal current
        if any(current.values()):
            status_candidates = list(current["comments"]) + [
                current["fields"][key]
                for key in ("status", "status_code", "http_status")
                if key in current["fields"]
            ]
            events.append({
                "event": current["event"],
                "data": "\n".join(current["data"]),
                "comments": list(current["comments"]),
                "fields": dict(current["fields"]),
                "status_code": next(
                    (s for s in map(_status_code_from_value, status_candidates) if s is not None), None),
            })
        current = {"event": None, "data": [], "comments": [], "fields": {}}

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip("\r")
        if line == "":
            _flush_current()
            continue
        if line.startswith(":"):
            current["comments"].append(line[1:].strip())
            continue

        field, sep, value = line.partition(":")
        if not sep:
            current["fields"][field.strip().lower()] = ""
            continue
        field = field.strip().lower()
        value = value.removeprefix(" ")
        if field == "event":
            current["event"] = value.strip()
        elif field == "data":
            current["data"].append(value)
        else:
            current["fields"][field] = value

    _flush_current()
    return events
