"""Plugin slash-command lookup and run for ``command.dispatch`` / ``slash.exec`` (``methods_tools``).

Bodies are rebound onto server.py's globals (method_ctx.bind_module) and reference them bare.
"""

from __future__ import annotations

from .method_ctx import bind_module


def _plugin_command_handler(name: str):
    try:
        return _tools_mod("hermes_agent.hermes_cli.plugins").get_plugin_command_handler(name)
    except Exception:
        return None


def _run_plugin_command(handler, arg: str, session=None, sid: str = "") -> str:
    """Run a plugin slash-command handler under the session's ``HERMES_SESSION_*`` binding.

    Plugin handlers read ``get_session_env()`` for the chat/session they serve; these RPCs run on
    the socket/worker thread where nothing upstream binds it (only the turn path does), so a handler
    saw ``""`` or the launch process's inherited values. Same class as the messaging gateway's
    #108698; ``_set_session_context`` is the turn path's own seam."""
    plugins = _tools_mod("hermes_agent.hermes_cli.plugins")
    invoke = _tools_mod("hermes_agent.hermes_cli.plugin_command_dispatch").invoke_plugin_command
    key = str(session.get("session_key") or "") if session else ""
    tokens = _set_session_context(key, cwd=str(session.get("cwd") or "")) if session else []
    context = {"source": None, "session_key": key, "platform": _session_source(session),
               "session_id": getattr(session.get("agent"), "session_id", None) or key if session else ""}
    # ctx.ask_choice: the ``clarify`` server request this backend's clarify tool sends.
    choices = _tools_mod("hermes_agent.hermes_cli.plugin_choices")
    try:
        with choices.bind_asker(choices.clarify_asker(lambda questions: _clarify_block(sid, questions))):
            return str(plugins.resolve_plugin_command_result(invoke(handler, arg, **context)) or "")
    finally:
        _clear_session_context(tokens)


def _dispatch_plugin_in_profile_scope(name: str, arg: str, session, sid: str = "", profile: str | None = None) -> str | None:
    """Run plugin command *name* of the chat's (sessionless: the named) profile with its secrets; None
    when it has none (#133244). Discovery runs ``register()``, so it stays in the secret scope; a loaded
    registry without *name* skips the bind, and a failing bind never fails a built-in."""
    with _session_home_scope(session, profile=profile):
        if _tools_mod("hermes_agent.hermes_cli.plugins").plugin_command_absent_nowait(name):
            return None
    try:
        scope = _session_profile_runtime_scope(session or {"profile_home": str(_profile_home(profile) or "") or None})
        scope.__enter__()
    except Exception:
        if _is_registry_command(name):
            return None
        raise
    try:
        handler = _plugin_command_handler(name)
        return None if handler is None else _run_plugin_command(handler, arg, session, sid)
    finally:
        scope.__exit__(None, None, None)


def register(server) -> None:
    bind_module(globals(), server)
