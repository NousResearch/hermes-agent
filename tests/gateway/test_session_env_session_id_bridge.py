"""dt-570 (behemoth): HERMES_SESSION_ID must reach the terminal tool on every
turn, not only on the turn that constructed the AIAgent.

_set_session_env binds every HERMES_SESSION_* ContextVar per message; it never
bound session_id, so _SESSION_ID was explicitly "" for the turn. AIAgent.__init__
sets it (set_current_session_id) only inside the executor-thread context of the
turn that built the agent; every later turn on a cached agent copies the handler
context where the var is "" — and _inject_session_context_env treats an
explicitly-empty ContextVar as authoritative, exporting HERMES_SESSION_ID="" to
the shell even though the os.environ mirror holds the right id.
"""
import os
from contextvars import copy_context

from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.platforms.base import Platform
from gateway.session_context import get_session_env
from tools.environments.local import _inject_session_context_env


def _context(session_id: str) -> SessionContext:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="8073053805",
        chat_name="Alexandru Bonea",
        chat_type="dm",
        user_id="8073053805",
        user_name="Alexandru Bonea",
        thread_id="18167",
    )
    return SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:8073053805:18167",
        session_id=session_id,
    )


def test_set_session_env_binds_session_id(monkeypatch):
    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    tokens = runner._set_session_env(_context("20260829_151644_229db1da"))
    try:
        assert get_session_env("HERMES_SESSION_ID") == "20260829_151644_229db1da"
        assert os.getenv("HERMES_SESSION_ID") is None  # contextvar path only
    finally:
        runner._clear_session_env(tokens)


def test_reused_agent_turn_bridges_session_id_to_subprocess(monkeypatch):
    """A turn on a cached agent: only _set_session_env runs (no AIAgent
    construction). The shell env must still carry the session id."""
    runner = object.__new__(GatewayRunner)
    # The os.environ mirror written by an earlier turn's set_current_session_id.
    monkeypatch.setenv("HERMES_SESSION_ID", "20260829_151644_229db1da")

    def turn():
        tokens = runner._set_session_env(_context("20260829_151644_229db1da"))
        try:
            env = {}
            _inject_session_context_env(env)
            return env
        finally:
            runner._clear_session_env(tokens)

    # The gateway runs the turn in a copy of the handler context (executor).
    env = copy_context().run(turn)
    assert env.get("HERMES_SESSION_THREAD_ID") == "18167"
    assert env.get("HERMES_SESSION_ID") == "20260829_151644_229db1da"


def test_session_id_stays_task_local_across_sessions(monkeypatch):
    """Two concurrent sessions never see each other's id (the leak guard the
    bridge exists for still holds with session_id bound)."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.setenv("HERMES_SESSION_ID", "foreign_session")

    def turn(sid):
        tokens = runner._set_session_env(_context(sid))
        try:
            env = {}
            _inject_session_context_env(env)
            return env.get("HERMES_SESSION_ID")
        finally:
            runner._clear_session_env(tokens)

    assert copy_context().run(turn, "sess_a") == "sess_a"
    assert copy_context().run(turn, "sess_b") == "sess_b"
