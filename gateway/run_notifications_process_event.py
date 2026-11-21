"""Synthetic process-completion event payloads for gateway watchers."""

def build_process_completion_event(watcher: dict, session, session_id: str) -> dict:
    """Build the synthetic ``completion`` event for an agent-notify watcher."""
    from gateway.run import _redact_gateway_user_facing_secrets
    from gateway.run_notifications import _WATCHER_ROUTE_FIELDS
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
        **{k: watcher.get(k, "") for k in _WATCHER_ROUTE_FIELDS},
        "owner_task_id": getattr(session, "owner_task_id", "") or "",
        "task_id": getattr(session, "task_id", "") or "",
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
