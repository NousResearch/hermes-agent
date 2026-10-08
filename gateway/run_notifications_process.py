"""Process completion event construction and human-facing notification formatting.

Redactors and display helpers are read from ``gateway.run`` at call time, preserving the facade seam.
"""

from __future__ import annotations

import time

from agent.i18n import t

# Routing fields copied verbatim from a process watcher onto its synthetic completion event.
_WATCHER_ROUTE_FIELDS = ("session_key", "platform", "chat_type", "chat_id", "thread_id", "user_id", "user_name")


class GatewayProcessNotificationMixin:
    """Build process notices for the gateway notification delivery paths."""

    @staticmethod
    def _redacted_output_tail(session, limit: int) -> str:
        """Last ``limit`` chars of process output through the secret redactors (unconditional floor)."""
        from gateway.run import _redact_gateway_user_facing_secrets
        from tools.ansi_strip import strip_ansi
        from tools.process_registry import transform_process_output
        new_output = strip_ansi(session.output_buffer[-limit:]) if session.output_buffer else ""
        if new_output:
            from agent.redact import redact_terminal_output
            _command = getattr(session, "command", "") or ""
            new_output = transform_process_output(new_output, command=_command, returncode=session.exit_code,
                                                  task_id=getattr(session, "task_id", "") or "")
            new_output = redact_terminal_output(new_output, _command)
            # redact_terminal_output() is unforced (raw when security.redact_secrets is off); this goes
            # straight to the adapter, so apply the same unconditional floor as agent-notify.
            new_output = _redact_gateway_user_facing_secrets(new_output)
        return new_output

    @staticmethod
    def _build_process_completion_event(watcher: dict, session, session_id: str) -> dict:
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

    def _format_process_final_message(self, session_id: str, session, notify_mode: str) -> str:
        """Human-facing completion message. Every mode shares the one-line status header; the
        raw-output modes (all/result/error) append the bounded output tail under it instead of the
        old bracketed ``[Background process proc_… finished~ …]`` debug wrapper (#54266)."""
        from gateway.run import _format_concise_process_notification, _redact_gateway_user_facing_secrets
        new_output = self._redacted_output_tail(session, 1000)
        _started = getattr(session, "started_at", None)
        _dur = max(0.0, time.time() - _started) if isinstance(_started, (int, float)) else None
        command = _redact_gateway_user_facing_secrets(getattr(session, "command", "") or "")
        if notify_mode == "concise":
            return _format_concise_process_notification(session_id, command, session.exit_code, new_output,
                                                        duration_seconds=_dur)
        header = _format_concise_process_notification(session_id, command, session.exit_code, "", duration_seconds=_dur)
        return t("gateway.background.final_output", header=header, output=new_output.strip()) if new_output.strip() else header

    def _format_process_running_message(self, session) -> str:
        from gateway.run import _redact_gateway_user_facing_secrets, _shorten_command_for_display
        new_output = self._redacted_output_tail(session, 500)
        short_cmd = _shorten_command_for_display(_redact_gateway_user_facing_secrets(getattr(session, "command", "") or ""))
        header = t("gateway.background.still_running") + (f" — `{short_cmd}`" if short_cmd else "")
        return t("gateway.background.recent_output", header=header, output=new_output.strip()) if new_output.strip() else header
