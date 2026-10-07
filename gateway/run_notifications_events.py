"""Process-watcher event construction, split out of ``gateway/run_notifications.py``.

One cohesive job: read a finished background process, bound and redact its output, and stamp the
launching watcher's chat route onto the synthetic ``completion`` event that the notification
lanes deliver. Kept in its own module so the notifications module — already over its line cap —
does not grow (root AGENTS.md: offset growth by moving a function into a topical sibling).
"""

from __future__ import annotations

# Chat-route keys copied verbatim off the watcher onto the completion event.
_WATCHER_ROUTE_FIELDS = ("session_key", "platform", "chat_type", "chat_id", "thread_id", "user_id", "user_name")


def build_process_completion_event(watcher: dict, session, session_id: str) -> dict:
    """Build the synthetic ``completion`` event for an agent-notify watcher."""
    from gateway.run import _redact_gateway_user_facing_secrets
    from agent.redact import redact_terminal_output
    from tools.ansi_strip import strip_ansi
    from tools.process_registry import transform_process_output
    _command = getattr(session, "command", "") or ""
    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
    _raw = transform_process_output(_raw, command=_command, returncode=session.exit_code,
                                    task_id=getattr(session, "task_id", "") or "") if _raw else _raw
    _raw = redact_terminal_output(_raw, _command)
    # Keep the last ~2000 chars snapped to a line boundary, with a marker when cut.
    _LIMIT = 2000
    # Truncate at line boundaries so notifications never start mid-line (fixes #23284). Keep the last
    # ~2000 chars but snap to the nearest preceding newline, then prepend a truncation marker when
    # output was cut.
    if len(_raw) > _LIMIT:
        _tail = _raw[-_LIMIT:]
        _nl = _tail.find("\n")
        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
    else:
        _out = _raw
    return {
        "type": "completion",
        "session_id": session_id,
        # The spawning turn's raw task id ("sa-..." for a delegated child): lets every delivery
        # lane apply one subagent-suppression rule and keep the attribution line when surfaced.
        "owner_task_id": getattr(session, "owner_task_id", "") or "",
        **{k: watcher.get(k, "") for k in _WATCHER_ROUTE_FIELDS},
        "message_id": str(watcher.get("message_id") or "").strip() or None,
        "started_at": getattr(session, "started_at", None),
        "command": _redact_gateway_user_facing_secrets(_command),
        "exit_code": session.exit_code,
        "completion_reason": getattr(session, "completion_reason", "exited"),
        "termination_source": getattr(session, "termination_source", ""),
        "output": _redact_gateway_user_facing_secrets(_out),
        # Spawning session-db id: lets pre-flight drop this completion if the user /new'd first.
        "parent_session_id": (
            watcher.get("parent_session_id") or getattr(session, "parent_session_id", "") or ""
        ),
    }
