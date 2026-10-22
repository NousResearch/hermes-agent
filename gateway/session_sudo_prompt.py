"""The masked sudo password prompt for a local viewer's turn.

Main's TUI sidecar registered ``set_sudo_password_callback`` per session (``tui_gateway/agent_callbacks.py``):
the terminal tool's sudo rewrite asked the renderer through a ``sudo`` server→client request and the
Ink card answered with ``{value}``. On the session authority no callback was registered, so the tool
took the headless path (``sudo: a terminal is required``). Here the owner sends the same ``sudo``
request frame (``tui_gateway/contracts/server_requests.py``) to the viewers attached to the session's
event stream and blocks the agent thread until one answers, the turn is interrupted, or it times out.
The password is never published, journaled or logged: it travels only in the viewer's response frame.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import uuid

logger = logging.getLogger(__name__)

SUDO_TIMEOUT_S = 120  # main: ``_ask("sudo", ..., timeout=120)``
_POLL_S = 0.25


class _Pending:
    __slots__ = ("event", "result", "session_id", "value")

    def __init__(self, session_id):
        self.event = threading.Event()
        self.session_id = session_id
        self.value = ""
        self.result = None


_lock = threading.Lock()
_open: dict[str, _Pending] = {}


def is_response_frame(frame) -> bool:
    """A viewer's answer to one of the owner's requests: a string ``srq-`` id, a result/error, no method."""
    return (isinstance(frame, dict) and "method" not in frame and isinstance(frame.get("id"), str)
            and frame["id"].startswith("srq-") and ("result" in frame or "error" in frame))


def resolve(frame, *, session_ok=lambda session_id: True) -> bool:
    """Settle the open request *frame* answers; False when nothing (here) waits for that id.
    ``session_ok`` lets the caller refuse an answer from a connection not entitled to that session."""
    with _lock:
        pending = _open.get(frame["id"])
        if pending is None or not session_ok(pending.session_id):
            return False
        _open.pop(frame["id"], None)
    result = frame.get("result") if "error" not in frame else None
    value = result.get("value") if isinstance(result, dict) else ""
    pending.value = value if isinstance(value, str) else ""
    pending.result = result if isinstance(result, dict) else None
    pending.event.set()
    return True


def ask_sudo(events, session_id, command, *, cancelled=lambda: False, timeout=SUDO_TIMEOUT_S) -> str:
    """Send one ``sudo`` request to the session's attached viewers and wait for its ``value``.
    "" when nobody is attached, the viewer skipped, the turn was interrupted, or it timed out."""
    return ask_viewer(events, session_id, "sudo", {"command": command}, cancelled=cancelled, timeout=timeout)


def ask_viewer(events, session_id, method, params, *, cancelled=lambda: False, timeout=SUDO_TIMEOUT_S) -> str:
    """Send one one-string server→client request (``sudo``, ``secret``: ``ValueResult``) to the
    session's attached viewers and wait for its ``value``; "" on skip, interrupt, timeout or no viewer."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout)
    return pending.value if pending is not None else ""


def ask_viewer_result(events, session_id, method, params, *, cancelled=lambda: False, timeout=SUDO_TIMEOUT_S):
    """The same round trip for a structured answer (``clarify``: ``ClarifyResult``): the viewer's
    ``result`` dict, or None on interrupt, timeout, an error response or no attached viewer."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout)
    return pending.result if pending is not None else None


def _ask(events, session_id, method, params, *, cancelled, timeout):
    if not events.fanout.transports():
        return None
    request_id = f"srq-{uuid.uuid4().hex[:12]}"
    pending = _Pending(session_id)
    with _lock:
        _open[request_id] = pending
    events.fanout.write({"jsonrpc": "2.0", "id": request_id, "method": method,
                         "params": {"session_id": session_id, **params}})
    waited = 0.0
    reason = "timeout"
    while not pending.event.wait(_POLL_S):
        waited += _POLL_S
        if cancelled():
            reason = "interrupted"
            break
        if timeout is not None and waited >= timeout:
            break
    with _lock:
        still_open = _open.pop(request_id, None) is pending
    if still_open:
        # The same teardown main's renderers key on (``request.cancel``): the card closes and a
        # timed-out password prompt says so.
        with contextlib.suppress(Exception):
            events.publish(session_id, {"id": request_id, "method": method, "reason": reason},
                           event_type="request.cancel")
        return None
    return pending


@contextlib.contextmanager
def sudo_prompt_scope(turn, agent):
    """Install the sudo callback on this (agent) thread for a local viewer's turn; restore after."""
    from gateway.run_turn_progress import local_viewer_turn
    if not local_viewer_turn(turn):
        yield
        return
    from tools.terminal_tool import _get_sudo_password_callback, set_sudo_password_callback
    authority, session_id, _generation = turn._approval_owner

    def callback():
        from gateway.run import _redact_approval_command
        from tools.terminal_tool_sudo import get_sudo_prompt_command
        live = authority.sessions.get(session_id)
        if live is None:
            return ""
        return ask_sudo(live.event_stream, session_id, _redact_approval_command(get_sudo_prompt_command()),
                        cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)))

    previous = _get_sudo_password_callback()
    set_sudo_password_callback(callback)
    try:
        yield
    finally:
        set_sudo_password_callback(previous)
