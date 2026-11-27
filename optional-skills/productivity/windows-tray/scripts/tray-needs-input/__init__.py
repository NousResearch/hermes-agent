# -*- coding: utf-8 -*-
"""tray-needs-input — observer-only plugin.

Mirrors "a turn is parked on a human prompt" (clarify question / approval
confirmation) into ``<hermes-home>/tray-needs-input.json`` so an external
Windows tray helper can light its dot amber. Every hook returns None: this
plugin never blocks a tool, never rewrites args, and never raises into the
approval or tool path.

Payload written (atomic os.replace):
    {"pending": true|false, "kind": "clarify|approval|...", "session_key": str, "ts": epoch}
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

STATE_FILE = "tray-needs-input.json"
_CLARIFY_TOOLS = {"clarify", "ask_clarify"}


def _home() -> str:
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes")


def _write(pending: bool, kind: str, session_key: str = "") -> None:
    payload = {"pending": pending, "kind": kind, "session_key": session_key, "ts": time.time()}
    try:
        home = _home()
        path = os.path.join(home, STATE_FILE)
        fd, tmp = tempfile.mkstemp(dir=home, prefix=".tni-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception:
        pass  # observability only — the tray falls back to its other signals


def _tool_name(kwargs: dict) -> str:
    name = kwargs.get("tool_name") or kwargs.get("function_name") or ""
    return name if isinstance(name, str) else ""


def _current(pending: bool | None = None, kind: str | None = None) -> bool:
    try:
        with open(os.path.join(_home(), STATE_FILE), encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return False
    if pending is not None and bool(d.get("pending")) is not pending:
        return False
    if kind is not None and d.get("kind") != kind:
        return False
    return True


def _on_pre_tool_call(**kwargs: Any):
    if _tool_name(kwargs) in _CLARIFY_TOOLS:
        _write(True, "clarify", str(kwargs.get("session_key") or kwargs.get("session_id") or ""))
    return None  # observer only


def _on_post_tool_call(**kwargs: Any):
    if _tool_name(kwargs) in _CLARIFY_TOOLS and _current(pending=True, kind="clarify"):
        _write(False, "clarify")
    return None


def _on_pre_approval(**kwargs: Any):
    _write(True, "approval", str(kwargs.get("session_key") or ""))
    return None


def _on_post_approval(**kwargs: Any):
    if _current(pending=True, kind="approval"):
        _write(False, "approval", str(kwargs.get("session_key") or ""))
    return None


def _on_session_end(**kwargs: Any):
    if _current(pending=True):
        _write(False, "session_end")
    return None


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_approval_request", _on_pre_approval)
    ctx.register_hook("post_approval_response", _on_post_approval)
    ctx.register_hook("on_session_end", _on_session_end)
