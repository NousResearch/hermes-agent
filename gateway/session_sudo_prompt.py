"""The masked sudo password prompt for a local viewer's turn.

Main's TUI sidecar registered ``set_sudo_password_callback`` per session (``tui_gateway/agent_callbacks.py``):
the terminal tool's sudo rewrite asked the renderer through a ``sudo`` server→client request and the
Ink card answered with ``{value}``. On the session authority no callback was registered, so the tool
took the headless path (``sudo: a terminal is required``). Here the owner sends the same ``sudo``
request frame (``tui_gateway/contracts/server_requests.py``) to the viewers attached to the session's
event stream and blocks the agent thread until one answers, the turn is interrupted, or it times out.
The password is never published, journaled or logged: it travels only in the viewer's response frame.
The request registry itself is ``gateway/session_server_requests.py``.
"""
from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)

SUDO_TIMEOUT_S = 120  # main: ``_ask("sudo", ..., timeout=120)``


def ask_sudo(events, session_id, command, *, cancelled=lambda: False, timeout=SUDO_TIMEOUT_S,
             wait_for_viewer=False) -> str:
    """Send one ``sudo`` request to the session's attached viewers and wait for its ``value``.
    "" when nobody is attached (unless *wait_for_viewer*: see ``ask_viewer``), the viewer skipped,
    the turn was interrupted, or it timed out."""
    from gateway.session_server_requests import ask_viewer
    return ask_viewer(events, session_id, "sudo", {"command": command}, cancelled=cancelled, timeout=timeout,
                      wait_for_viewer=wait_for_viewer)


@contextlib.contextmanager
def sudo_prompt_scope(turn, agent):
    """Install the sudo callback on this (agent) thread for a local viewer's turn; restore after."""
    from gateway.run_turn_progress import local_viewer_turn
    if not local_viewer_turn(turn):
        yield
        return
    from gateway.session_finite import finite_turn_required
    from tools.terminal_tool import _get_sudo_password_callback, set_sudo_password_callback
    authority, session_id, _generation = turn._approval_owner
    # A viewer that reconnects gets the card through ``open_requests``; a finite turn's (``chat -q``)
    # client detached on purpose and nobody can answer, so that one stays headless at once.
    wait = finite_turn_required() is not True

    def callback():
        from gateway.run import _redact_approval_command
        from tools.terminal_tool_sudo import get_sudo_prompt_command
        live = authority.sessions.get(session_id)
        if live is None:
            return ""
        return ask_sudo(live.event_stream, session_id, _redact_approval_command(get_sudo_prompt_command()),
                        cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)),
                        timeout=SUDO_TIMEOUT_S, wait_for_viewer=wait)

    previous = _get_sudo_password_callback()
    set_sudo_password_callback(callback)
    try:
        yield
    finally:
        set_sudo_password_callback(previous)
