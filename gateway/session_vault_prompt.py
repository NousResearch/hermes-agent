"""The browser vault's prompts (unlock, save login, one-time code) for a local viewer's turn.

Main's TUI sidecar registered the three vault callbacks per session (``tui_gateway/agent_callbacks.py``):
``browser_vault`` asked the renderer through ``vault.unlock_prompt``, ``vault.save_login`` and ``vault.code``
server→client requests, and the Desktop and Ink cards answered with ``{value}``. On the session authority no
callback was registered, so every Desktop chat reported ``prompt_unavailable`` ("this session cannot ask the
user") and a site's emailed or texted code could not be entered at all. Here the owner sends the same request
frames (``tui_gateway/contracts/server_requests.py``) to the viewers attached to the session's event stream and
blocks the agent thread until one answers, the turn is interrupted, or it times out. Passwords and codes are
never published, journaled or logged: they travel only in the viewer's response frame. The request registry
itself is ``gateway/session_server_requests.py``.
"""
from __future__ import annotations

import contextlib
import json
import logging

logger = logging.getLogger(__name__)

# Main's sidecar deadlines (``tui_gateway/agent_callbacks.py``).
UNLOCK_TIMEOUT_S = 120
SAVE_LOGIN_TIMEOUT_S = 180
CODE_TIMEOUT_S = 180


def release_unlocks(session_id) -> None:
    """The session closed: lock the password managers it unlocked, as the sidecar's session end does
    (``tui_gateway/session_lifecycle.py`` ``_lock_vault_managers``). Other sessions keep theirs."""
    try:
        from agent.vault_backends import unlock
        unlock.release_session(session_id)
    except Exception:  # health: allow BLE001 -- a close must finish even if the lock fails
        logger.debug("vault manager lock on session close failed", exc_info=True)


@contextlib.contextmanager
def vault_prompt_scope(turn, agent):
    """Install the vault callbacks on this (agent) thread for a local viewer's turn; restore after.
    Tool workers inherit them through ``tools.thread_context.propagate_context_to_thread``."""
    from gateway.run_turn_progress import local_viewer_turn
    if not local_viewer_turn(turn):
        yield
        return
    from agent.vault_backends import unlock
    authority, session_id, _generation = turn._approval_owner

    def ask(method, params, timeout):
        from gateway.session_server_requests import ask_viewer
        live = authority.sessions.get(session_id)
        if live is None:
            return ""
        return ask_viewer(live.event_stream, session_id, method, params, timeout=timeout,
                          cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)))

    def unlock_prompt(backend, display_name):
        return ask("vault.unlock_prompt", {"backend": backend, "display_name": display_name}, UNLOCK_TIMEOUT_S)

    def save_login(origin, site):
        # The card sends identifier + password as one JSON string; it goes straight to the vault store.
        raw = ask("vault.save_login", {"origin": origin, "site": site}, SAVE_LOGIN_TIMEOUT_S)
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            return None
        return data if isinstance(data, dict) and data.get("password") else None

    def code_prompt(site, hint):
        return ask("vault.code", {"site": site, "hint": hint}, CODE_TIMEOUT_S)

    previous = (unlock.get_unlock_prompt_callback(), unlock.get_save_login_prompt_callback(),
                unlock.get_code_prompt_callback(), unlock.get_current_session_id())
    unlock.set_unlock_prompt_callback(unlock_prompt)
    unlock.set_save_login_prompt_callback(save_login)
    unlock.set_code_prompt_callback(code_prompt)
    unlock.set_current_session_id(session_id)  # an unlock made on this turn is released when this session closes
    try:
        yield
    finally:
        unlock.set_unlock_prompt_callback(previous[0])
        unlock.set_save_login_prompt_callback(previous[1])
        unlock.set_code_prompt_callback(previous[2])
        unlock.set_current_session_id(previous[3])
