"""Small, deterministic workflow traces used by ``/learn``.

The trace is deliberately an operation log, not a transcript export: tool results and
message bodies are never included.  This makes it safe to hand to skill authoring while
keeping the reader bounded and profile-local.
"""
from __future__ import annotations

import json
import re
from typing import Any

from hermes_constants import get_hermes_home

DEFAULT_MAX_MESSAGES = 64
MAX_TOOL_CALLS_PER_STEP = 32
MAX_SESSION_ID = 128


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _tool_name(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(call.get("name") or function.get("name") or "unknown").strip() or "unknown"


def _safe_arguments(call: dict[str, Any]) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    args = call.get("arguments", function.get("arguments", {}))
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (TypeError, ValueError):
            parsed = args
    else:
        parsed = args
    def structure(value: Any, key: str = "") -> Any:
        # Argument values are often prompts, file contents, URLs, or shell text.
        # Keep the shape useful for a workflow recipe, but never copy free-form
        # strings into the trace.  In particular, redaction is not a sufficient
        # privacy boundary: arbitrary private text can be non-secret too.
        if isinstance(value, dict):
            safe_keys = {}
            for raw_key, raw_value in list(value.items())[:32]:
                key_text = str(raw_key)
                safe_key = key_text[:64] if re.fullmatch(r"[A-Za-z0-9_.-]+", key_text[:64]) else "<key>"
                safe_keys[safe_key] = structure(raw_value, key_text)
            return safe_keys
        if isinstance(value, list):
            return [structure(v, key) for v in value[:32]]
        if isinstance(value, str):
            return "<string>"
        if value is None:
            return "<null>"
        if isinstance(value, bool):
            return "<boolean>"
        if isinstance(value, int):
            return "<integer>"
        if isinstance(value, float):
            return "<number>"
        return f"<{type(value).__name__}>"
    return _text(structure(parsed))[:1000]


def _markdown_escape(value: Any, limit: int = 160) -> str:
    """Render metadata as inert, bounded Markdown text."""
    text = str(value if value is not None else "")[:limit]
    return (text.replace("\\", "\\\\").replace("`", "\\`")
            .replace("*", "\\*")
            .replace("[", "\\[").replace("]", "\\]")
            .replace("<", "\\<").replace(">", "\\>")
            .replace("\r", " ").replace("\n", " "))


def _trusted_bound_session_id(requested: str | None) -> str | None:
    """Return a requested id only when it is compatible with the task-local binding."""
    requested = str(requested or "").strip() or None
    try:
        from gateway import session_context as _session_context
        bound = _session_context._SESSION_ID.get()
        if bound is not _session_context._UNSET:
            bound = str(bound or "").strip() or None
            if requested != bound:
                return None
    except Exception:
        pass
    return requested


def read_workflow_trace(session_id: str, *, db=None, max_messages: int = DEFAULT_MAX_MESSAGES) -> dict[str, Any]:
    """Read at most *max_messages* rows from one profile's SessionDB.

    The returned records contain user/assistant labels and tool-call metadata only;
    tool rows and all message content are intentionally excluded.
    """
    if not str(session_id or "").strip():
        raise ValueError("session_id is required")
    sid = _trusted_bound_session_id(session_id)
    if not sid:
        raise PermissionError("workflow trace session_id is not the active session")
    # A session id arriving from a caller is not an authority to read another transcript.
    # The gateway binds the active id/profile in ContextVars; only test/ad-hoc DB readers with
    # no binding retain the legacy direct-reader behavior.
    try:
        from gateway import session_context as _session_context
        # Do not consult get_session_env here: it intentionally falls back to the process
        # environment for legacy subprocesses, and that environment is not caller proof.
        _unset = _session_context._UNSET
        _bound_sid = _session_context._SESSION_ID.get()
        _bound_profile = _session_context._SESSION_PROFILE.get()
        active_sid = "" if _bound_sid is _unset else str(_bound_sid or "").strip()
        active_profile = "" if _bound_profile is _unset else str(_bound_profile or "").strip()
    except Exception:
        active_sid = active_profile = ""
    if active_sid and active_sid != sid:
        raise PermissionError("workflow trace session_id is not the active session")
    limit = max(1, min(int(max_messages), DEFAULT_MAX_MESSAGES))
    owns_db = db is None
    if db is None:
        from hermes_state import SessionDB
        db = SessionDB(get_hermes_home() / "state.db")
    try:
        if active_profile:
            owner = getattr(db, "_own_profile_name", lambda: None)()
            if owner and owner != active_profile:
                raise PermissionError("workflow trace profile is not the active profile")
        messages = db.get_messages_as_conversation(
            sid, include_ancestors=True, limit=limit + 1, latest=True, exclude_roles=("tool",)) or []
        truncated = len(messages) > limit
        visible = [m for m in messages if isinstance(m, dict) and m.get("role") != "tool"]
        messages = visible[-limit:]
        steps = []
        for message in messages:
            calls = message.get("tool_calls") or []
            if isinstance(calls, str):
                try:
                    calls = json.loads(calls)
                except (TypeError, ValueError):
                    calls = []
            if not isinstance(calls, list):
                calls = []
            step = {"role": str(message.get("role") or "unknown"), "tools": []}
            for call in calls[:MAX_TOOL_CALLS_PER_STEP]:
                if isinstance(call, dict):
                    step["tools"].append({"name": _tool_name(call), "arguments": _safe_arguments(call)})
            if len(calls) > MAX_TOOL_CALLS_PER_STEP:
                step["tools_truncated"] = True
            steps.append(step)
        return {"session_id": sid, "steps": steps, "truncated": truncated, "message_count": len(messages)}
    finally:
        if owns_db:
            db.close()


def render_workflow_trace(trace: dict[str, Any]) -> str:
    """Render a stable markdown reference; never render tool results."""
    lines = [f"# Workflow trace ({_markdown_escape(trace.get('session_id', ''), MAX_SESSION_ID)})", ""]
    for index, step in enumerate(trace.get("steps") or [], 1):
        lines.append(f"{index}. **{_markdown_escape(step.get('role', 'unknown'), 32)}**")
        for tool in step.get("tools") or []:
            lines.append(
                f"   - tool `{_markdown_escape(tool.get('name', 'unknown'), 80)}` "
                f"args: `{_markdown_escape(tool.get('arguments', '{}'), 1000)}`"
            )
        if step.get("tools_truncated"):
            lines.append("   - _additional tool calls omitted_")
    if trace.get("truncated"):
        lines.append("\n_Trace bounded to the most recent operations._")
    return "\n".join(lines) + "\n"
