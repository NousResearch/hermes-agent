"""The gateway binds HERMES_SESSION_ID so agent-initiated session renames can find the id.

Terminal-tool subprocesses (``hermes sessions rename <id> "<title>"`` run by the agent
mid-turn) need the CURRENT session's durable id. The gateway binds the session context
through ``set_session_vars``; before the fix the call omitted ``session_id``, so the
ContextVar stayed ``_UNSET`` and the subprocess env bridge (which treats an engaged-but-
unset var as "no session in this task") STRIPPED ``HERMES_SESSION_ID`` from every terminal
child — the id was live in the gateway but never reached the tool that needed it.

These tests pin the binding seam end to end: the runner's ``_set_session_env`` (the
production binder) through ``_inject_session_context_env`` (the production bridge) to the
child env, without spawning a real subprocess.
"""

import os
from types import SimpleNamespace

import pytest

import gateway.session_context as sc
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import _VAR_MAP, clear_session_vars, set_session_vars
from tools.environments.local import _inject_session_context_env

SESSION_VARS = list(_VAR_MAP.keys())


@pytest.fixture(autouse=True)
def _isolate_session_context(monkeypatch):
    """Clean ContextVar + os.environ + engaged-latch slate per test, restored."""
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(sc._UNSET)
    sc._session_context_engaged = False
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _context(session_id="sess-turn-1"):
    """A SessionContext shaped like the gateway's, with the source fields _set_session_env reads."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-101", chat_type="group", chat_name=None, thread_id=None,
        user_id="u1", user_id_alt=None, user_name="tester", message_id=None,
        profile="", scope_id="", parent_chat_id="",
    )
    return SimpleNamespace(source=source, session_key="agent:main:telegram:group:-101",
                           session_id=session_id)


def test_runner_set_session_env_binds_the_session_id():
    """The production binder passes the context's session_id into set_session_vars, so the
    HERMES_SESSION_ID ContextVar is bound (not _UNSET) for the whole turn."""
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}

    tokens = runner._set_session_env(_context(session_id="sess-turn-1"))
    try:
        assert sc.get_session_env("HERMES_SESSION_ID") == "sess-turn-1"
        assert sc.get_session_env("HERMES_SESSION_KEY") == "agent:main:telegram:group:-101"
    finally:
        clear_session_vars(tokens)


def test_bound_session_id_reaches_the_terminal_child_env():
    """End to end across the bridge: the gateway-bound session id is what a terminal-tool
    subprocess sees, even when a foreign concurrent session's id sits in os.environ (the
    engaged bridge must prefer the bound value, never the process-global mirror)."""
    os.environ["HERMES_SESSION_ID"] = "FOREIGN-CONCURRENT-ID"
    tokens = set_session_vars(session_id="sess-turn-1",
                              session_key="agent:main:telegram:group:-101",
                              platform="telegram", chat_id="-101")
    try:
        child_env = {"PATH": "/usr/bin"}
        _inject_session_context_env(child_env)
        assert child_env["HERMES_SESSION_ID"] == "sess-turn-1"
    finally:
        clear_session_vars(tokens)


def test_unbound_session_id_is_stripped_not_inherited():
    """The pre-fix behaviour this guards against regressing into silently: engaged context
    WITHOUT a session_id binding must strip the var rather than leak the foreign global."""
    sc._session_context_engaged = True
    os.environ["HERMES_SESSION_ID"] = "FOREIGN-CONCURRENT-ID"

    child_env = {"PATH": "/usr/bin", "HERMES_SESSION_ID": "FOREIGN-CONCURRENT-ID"}
    _inject_session_context_env(child_env)

    assert "HERMES_SESSION_ID" not in child_env
