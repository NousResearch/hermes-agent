"""Self-improvement telemetry plugin.

This opt-in plugin records structured, local-only tool-call metrics for later
self-improvement review. It intentionally logs sizes, labels, and risk flags —
not raw tool outputs, prompts, transcripts, or argument values.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

_LOCK = threading.RLock()
_SESSION_STATE: dict[str, dict[str, int]] = {}

LARGE_TOOL_OUTPUT_CHARS = 10_000
VERY_LARGE_TOOL_OUTPUT_CHARS = 40_000


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("on_session_reset", on_session_reset)


def reset_runtime_state() -> None:
    with _LOCK:
        _SESSION_STATE.clear()


def telemetry_dir() -> Path:
    override = os.getenv("HERMES_SELF_IMPROVEMENT_TELEMETRY_DIR", "").strip()
    if override:
        return Path(override)
    return get_hermes_home() / "ops" / "self-improvement-log"


def _session_id(kwargs: dict[str, Any]) -> str:
    return str(kwargs.get("session_id") or kwargs.get("task_id") or "unknown")


def _result_chars(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, str):
        return len(result)
    try:
        return len(json.dumps(result, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(result))


def _args_keys(args: Any) -> list[str]:
    if isinstance(args, dict):
        return sorted(str(key) for key in args.keys())
    return []


def _risk_flags(session_id: str, tool_name: str, args: Any, result_chars: int) -> list[str]:
    flags: list[str] = []
    if result_chars >= LARGE_TOOL_OUTPUT_CHARS:
        flags.append("large_tool_output")
    if result_chars >= VERY_LARGE_TOOL_OUTPUT_CHARS:
        flags.append("very_large_tool_output")

    with _LOCK:
        state = _SESSION_STATE.setdefault(session_id, {})
        if tool_name == "skill_view":
            key = f"skill_view:{_skill_name(args)}"
            state[key] = state.get(key, 0) + 1
            if state[key] >= 2:
                flags.append("duplicate_skill_view")
        if tool_name == "cronjob" and _arg_value(args, "action") == "list":
            key = "cronjob:list"
            state[key] = state.get(key, 0) + 1
            if state[key] >= 2:
                flags.append("repeated_cronjob_list")
    return flags


def _skill_name(args: Any) -> str:
    value = _arg_value(args, "name")
    return value or "unknown"


def _arg_value(args: Any, key: str) -> str:
    if isinstance(args, dict):
        value = args.get(key)
        return str(value) if value is not None else ""
    return ""


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def on_pre_tool_call(**kwargs: Any) -> None:
    # Reserved for future duration tracking once hook payloads expose stable
    # call identifiers. Keep the hook registered so shape discovery can confirm
    # runtime availability without changing behavior.
    return None


def on_post_tool_call(**kwargs: Any) -> None:
    tool_name = str(kwargs.get("tool_name") or kwargs.get("name") or "unknown")
    args = kwargs.get("args") or {}
    session_id = _session_id(kwargs)
    result_chars = _result_chars(kwargs.get("result"))
    payload = {
        "schema_version": 1,
        "kind": "tool_call_metric",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "session_id": session_id,
        "tool_name": tool_name,
        "args_keys": _args_keys(args),
        "duration_ms": int(kwargs.get("duration_ms") or 0),
        "result_chars": result_chars,
        "risk_flags": _risk_flags(session_id, tool_name, args, result_chars),
    }
    _append_jsonl(telemetry_dir() / "context_metrics.jsonl", payload)
    return None


def on_session_end(**kwargs: Any) -> None:
    _SESSION_STATE.pop(_session_id(kwargs), None)
    return None


def on_session_reset(**kwargs: Any) -> None:
    _SESSION_STATE.pop(_session_id(kwargs), None)
    return None
