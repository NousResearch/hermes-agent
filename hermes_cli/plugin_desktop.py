"""Public PluginContext capability for requests linked to a live desktop player."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any


_OPERATION_FIELDS = {
    "context": {"type"},
    "jobs.list": {"type"},
    "control": {"type", "context_token", "navigation_generation", "action", "position_seconds"},
    "transcript.start": {"type", "context_token", "language", "range"},
    "download.start": {"type", "context_token", "quality"},
    "job.status": {"type", "job_id"},
    "job.cancel": {"type", "job_id"},
}
_CONTROL_ACTIONS = frozenset({
    "play", "pause", "mute", "unmute", "seek", "pin", "unpin", "fullscreen", "exit-fullscreen",
})
_DOWNLOAD_QUALITIES = frozenset({"best", "720p", "1080p", "audio"})
_TOKEN_MIN, _TOKEN_MAX = 24, 256
logger = logging.getLogger(__name__)


def _valid_range(value: Any) -> bool:
    if not isinstance(value, dict) or not value or set(value) - {"start_seconds", "end_seconds"}:
        return False
    for number in value.values():
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 86_400:
            return False
    return value.get("end_seconds", 86_400) >= value.get("start_seconds", 0)


def _valid_operation(operation: Any) -> bool:
    """Reject non-JSON, open-ended, or ill-typed requests before host routing."""
    if not isinstance(operation, dict) or not all(isinstance(key, str) for key in operation):
        return False
    kind = operation.get("type")
    if not isinstance(kind, str) or kind not in _OPERATION_FIELDS:
        return False
    if set(operation) - _OPERATION_FIELDS[kind]:
        return False
    if kind in {"context", "jobs.list"}:
        return set(operation) == {"type"}
    if kind == "control":
        token, generation, action = (operation.get("context_token"),
                                     operation.get("navigation_generation"), operation.get("action"))
        if not _valid_token(token) or type(generation) is not int or generation < 0:
            return False
        if not isinstance(action, str) or action not in _CONTROL_ACTIONS:
            return False
        position = operation.get("position_seconds")
        if (action == "seek") != (position is not None):
            return False
        return position is None or (
            type(position) in (int, float) and math.isfinite(position) and 0 <= position <= 86_400
        )
    if kind in {"transcript.start", "download.start"} and not _valid_token(operation.get("context_token")):
        return False
    if kind == "transcript.start":
        language = operation.get("language")
        if language is not None and (
            not isinstance(language, str) or not 2 <= len(language) <= 35
            or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language)
        ):
            return False
        segment = operation.get("range")
        return segment is None or _valid_range(segment)
    if kind == "download.start":
        return operation.get("quality") in (None, *_DOWNLOAD_QUALITIES)
    if kind in {"job.status", "job.cancel"}:
        job_id = operation.get("job_id")
        return isinstance(job_id, str) and 1 <= len(job_id) <= 128 and all(
            char.isalnum() or char in "_-" for char in job_id
        )
    return False


def _valid_token(token: Any) -> bool:
    return isinstance(token, str) and _TOKEN_MIN <= len(token) <= _TOKEN_MAX


class DesktopPluginCapability:
    """Capability limited to sink reachability and one session-bound media call."""

    def available(self) -> bool:
        try:
            from tools import desktop_ui
            return bool(desktop_ui.available())
        except (ImportError, AttributeError):
            return False

    def media_request(self, *, session_id: str, operation: dict) -> dict:
        if not isinstance(session_id, str) or not session_id.strip():
            return {"error": "session-unavailable"}
        if not _valid_operation(operation):
            return {"error": "invalid-operation"}
        try:
            # Canonical local turns have no legacy runtime record. Their lease
            # owns the route; an expired/foreign lease must never fall back.
            from gateway.local_player import request
            result = request(session_id, operation)
            if result is None:
                from tui_gateway.server import request_player_agent_for_session
                result = request_player_agent_for_session(session_id, operation)
            parsed = json.loads(result) if isinstance(result, str) else result
            return parsed if isinstance(parsed, dict) else {"error": "player-request-unavailable"}
        except Exception as exc:  # health: allow BLE001 -- tracebacks can expose account URLs or media context
            logger.warning("Plugin desktop media request failed (%s)", type(exc).__name__)
            return {"error": "player-request-unavailable"}
