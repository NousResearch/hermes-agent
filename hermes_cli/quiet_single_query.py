"""Quiet ``hermes chat -Q`` helpers: bind this session's key and resume nested notifies.

Bot Mode delivers a local DM as ``hermes -p <bot> chat -Q --query-file``. Interactive
chat binds ``set_current_session_key(self.session_id)`` around the turn; the quiet
path did not, so a nested ``message_agent`` notify inherited the dispatcher's
``HERMES_SESSION_KEY`` and never woke the recipient. Quiet also printed and exited
after one turn, so a nested teammate reply that finished during the one-shot linger
was never injected as a follow-up.
"""

from __future__ import annotations

from typing import Any, Callable

# Nested A→B→C is one extra turn; this caps a runaway message_agent chain.
_MAX_QUIET_NOTIFY_ROUNDS = 8


def bind_quiet_session_key(session_id: str):
    """Bind the approval/session key to *this* quiet session, not an inherited parent env."""
    from tools.approval_context import reset_current_session_key, set_current_session_key

    token = set_current_session_key(session_id or "default")
    return token, reset_current_session_key


def continue_quiet_notify_completions(
    session_id: str,
    run_turn: Callable[[str], Any],
    *,
    owns_event=None,
    max_rounds: int = _MAX_QUIET_NOTIFY_ROUNDS,
) -> Any:
    """Linger for ``notify_on_complete`` work, then run owned completion texts as follow-up turns.

    Returns the last ``run_turn`` result, or ``None`` when nothing owned completed.
    """
    from tools.process_registry import process_registry

    last: Any = None
    key = session_id or ""
    for _ in range(max(int(max_rounds), 0)):
        process_registry.wait_for_pending_completions(None)
        drained = process_registry.drain_notifications(session_key=key, owns_event=owns_event)
        texts = [
            text for event, text in drained
            if event.get("type", "completion") == "completion" and text
        ]
        if not texts:
            return last
        last = run_turn("\n\n".join(texts))
    return last
