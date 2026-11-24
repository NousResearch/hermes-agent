"""The masked skill-secret prompt for a local viewer's turn.

Main's classic CLI registered ``set_secret_capture_callback`` in process: a skill whose
``required_environment_variables`` are missing opened the masked "Skill Setup Required" panel and the
value was saved to the profile ``.env`` (``save_env_value_secure``) without ever reaching the model.
On the session authority the skill runs in the gateway, where a gateway surface only got the
"secret entry is not available" hint. For a local viewer's turn the owner now sends main's ``secret``
server→client request (``tui_gateway/contracts/server_requests.py``: ``SecretRequestParams`` →
``ValueResult``) to the attached viewers, saves a non-empty answer in this profile's ``.env`` here,
and returns main's callback result. The value travels only in the viewer's response frame.
Ephemeral ``request_secure_input`` uses the same masked request but never the persistence
callback; its turn-scoped answer is accepted only for the live session generation.
"""
from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)

SECRET_TIMEOUT_S = 120  # main's CLI secret panel deadline (hermes_cli/callbacks.py)


def _skipped(var_name, reason, message):
    return {"success": True, "reason": reason, "stored_as": var_name, "validated": False,
            "skipped": True, "message": message}


@contextlib.contextmanager
def secret_prompt_scope(turn, agent):
    """Bind this turn's secret-capture callback (a ContextVar, so parallel tool workers inherit it)."""
    from gateway.run_turn_progress import local_viewer_turn
    from tools.secure_input_tool import bind_secure_input_callback
    if not local_viewer_turn(turn):
        with bind_secure_input_callback(None):
            yield
        return
    from tools.skills_tool import bind_turn_secret_capture
    authority, session_id, _generation = turn._approval_owner

    def callback(var_name, prompt, metadata=None):
        from gateway.session_server_requests import ask_viewer
        live = authority.sessions.get(session_id)
        if live is None:
            return _skipped(var_name, "cancelled", "Secret setup was skipped.")
        params = {"env_var": var_name, "prompt": prompt, **({"metadata": metadata} if metadata else {})}
        value = ask_viewer(live.event_stream, session_id, "secret", params, timeout=SECRET_TIMEOUT_S,
                           cancelled=lambda: bool(getattr(agent, "_interrupt_requested", False)))
        if not value:
            return _skipped(var_name, "cancelled", "Secret setup was skipped.")
        from hermes_cli.config import save_env_value_secure
        return {**save_env_value_secure(var_name, value), "skipped": False,
                "message": "Secret stored securely. The secret value was not exposed to the model."}

    def ephemeral_callback(purpose, prompt, metadata=None):
        from gateway.session_server_requests import ask_viewer
        live = authority.sessions.get(session_id)

        def current_owner():
            row = authority.db.get_session(session_id)
            return (live is not None and authority.sessions.get(session_id) is live
                    and row is not None and row["runtime_generation"] == _generation
                    and not getattr(agent, "_interrupt_requested", False))

        if not current_owner():
            return ""
        value = ask_viewer(live.event_stream, session_id, "secret",
                           {"env_var": purpose, "prompt": prompt, "metadata": metadata or {}},
                           timeout=SECRET_TIMEOUT_S, cancelled=lambda: not current_owner())
        return value if current_owner() else ""

    with bind_turn_secret_capture(callback), bind_secure_input_callback(ephemeral_callback):
        yield
