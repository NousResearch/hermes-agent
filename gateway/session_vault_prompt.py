"""Masked vault requests for canonical local viewers; answers never enter the event journal."""
from __future__ import annotations

import contextlib
import json

UNLOCK_TIMEOUT_S = 120
ENTRY_TIMEOUT_S = 180


@contextlib.contextmanager
def vault_prompt_scope(turn, agent):
    """Bind the local turn's vault callbacks and owner, restoring every slot on exit."""
    from gateway.run_turn_progress import local_viewer_turn
    from gateway.session_cli_turn import classic_cli_turn
    from tools.approval_context import _no_user_can_answer
    if not local_viewer_turn(turn) or classic_cli_turn(turn) or _no_user_can_answer():
        yield
        return

    from agent.vault_backends import unlock
    authority, session_id, generation = turn._approval_owner
    live = authority.sessions.get(session_id)
    active = True

    def cancelled():
        return (not active or bool(getattr(agent, '_interrupt_requested', False))
                or live is None or authority.sessions.get(session_id) is not live
                or authority.executing.get(session_id) != generation
                or not live.event_stream.fanout.transports())

    def ask(method, params, timeout):
        from gateway.session_server_requests import ask_viewer
        # Check both sides of the wait: an answer may race Stop, close, or turn teardown.
        if cancelled():
            return ''
        value = ask_viewer(live.event_stream, session_id, method, params,
                           cancelled=cancelled, timeout=timeout)
        return '' if cancelled() else value

    def unlock_cb(backend, display_name):
        return ask('vault.unlock_prompt', {'backend': backend, 'display_name': display_name}, UNLOCK_TIMEOUT_S)

    def save_login_cb(origin, site):
        raw = ask('vault.save_login', {'origin': origin, 'site': site}, ENTRY_TIMEOUT_S)
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            return None
        if not isinstance(data, dict) or not all(
                isinstance(data.get(key), str) and data[key] for key in ('identifier', 'password')):
            return None
        return {key: data[key] for key in ('identifier', 'password')}

    def code_cb(site, hint):
        return ask('vault.code', {'site': site, 'hint': hint}, ENTRY_TIMEOUT_S)

    previous = (unlock.get_unlock_prompt_callback(), unlock.get_save_login_prompt_callback(),
                unlock.get_code_prompt_callback(), unlock.get_current_session_id())
    unlock.set_current_session_id(session_id)
    unlock.set_unlock_prompt_callback(unlock_cb)
    unlock.set_save_login_prompt_callback(save_login_cb)
    unlock.set_code_prompt_callback(code_cb)
    try:
        yield
    finally:
        active = False
        unlock.set_unlock_prompt_callback(previous[0])
        unlock.set_save_login_prompt_callback(previous[1])
        unlock.set_code_prompt_callback(previous[2])
        unlock.set_current_session_id(previous[3])
