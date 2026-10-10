"""Conversation boundaries must drop the session's terminal cwd record.

The record is keyed by the SESSION KEY — the stable chat lane, reused across generations — so it
outlived the conversation that wrote it: a ``cd`` from an earlier conversation overrode the
profile's configured ``terminal.cwd`` in a later one (accountant incident, 2026-09-28: the env
snapshot showed ``/home/ec2-user/Developer/finances`` while the command inside it resolved
relative paths against ``$HOME``).

``/new`` routes through the conversation-boundary funnel (``_clear_conversation_scope``) along
with /resume, auto-reset and the compression-exhausted reset; the terminal cwd record is
conversation-scoped state and is cleared there, under the same key it is written under.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import tools.terminal_tool as tt
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key

STALE = "/home/ec2-user"


def _source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(session_key: str, session_id: str = "sess-1"):
    """A GatewayRunner wired just far enough to run ``/new`` (mirrors test_session_model_reset)."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._background_tasks = set()
    session_entry = SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    runner._session_db = None
    runner._agent_cache_lock = None
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""
    return runner


@pytest.fixture(autouse=True)
def _clean_terminal_state(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})


def _configured_terminal_cwd() -> str:
    return tt._get_env_config()["cwd"]


def test_new_drops_the_sessions_terminal_cwd_record():
    """After /new the next command resolves the configured cwd, not the old conversation's ``cd``."""
    source = _source("c1")
    session_key = build_session_key(source)
    other_key = build_session_key(_source("c2"))
    runner = _make_runner(session_key)

    tt.record_session_cwd(session_key, STALE)
    tt.record_session_cwd(other_key, STALE)

    asyncio.run(runner._handle_reset_command(MessageEvent(text="/new", source=source, message_id="m1")))

    assert tt.get_session_cwd(session_key) is None
    assert tt.get_session_cwd(other_key) == STALE  # another lane keeps its own record
    assert tt._resolve_command_cwd(
        workdir=None, default_cwd=_configured_terminal_cwd(), session_key=session_key
    ) == _configured_terminal_cwd()


def test_reset_clears_only_its_own_profile_lane(tmp_path, monkeypatch):
    """A multiplexed gateway serves both profiles: one lane's /new never clears the other's.

    Two homes, A → B → A: the record is keyed by the profile-namespaced session key, and each
    profile's own configured ``terminal.cwd`` is what its lane falls back to.
    """
    from agent.secret_scope import set_multiplex_active
    from gateway.run import _profile_runtime_scope

    launch = tmp_path / "launch"
    home_a = launch / "profiles" / "accountant"
    home_b = launch / "profiles" / "assistant"
    finances_a = tmp_path / "finances-a"
    finances_b = tmp_path / "finances-b"
    for d in (home_a, home_b, finances_a, finances_b):
        d.mkdir(parents=True)
    (home_a / "config.yaml").write_text(f"terminal:\n  cwd: {finances_a}\n")
    (home_b / "config.yaml").write_text(f"terminal:\n  cwd: {finances_b}\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))

    source_a = _source("c1")
    source_b = _source("c2")
    key_a = build_session_key(source_a, profile="accountant")
    key_b = build_session_key(source_b, profile="assistant")
    assert key_a != key_b and "accountant" in key_a and "assistant" in key_b
    runner_a = _make_runner(key_a, session_id="sess-a")
    runner_b = _make_runner(key_b, session_id="sess-b")

    set_multiplex_active(True)
    try:
        # Each lane's `cd` is recorded under its own profile scope — the record key is
        # profile-qualified, so a write outside a scope belongs to no lane.
        with _profile_runtime_scope(home_a, prepared_secret_scope={}):
            tt.record_session_cwd(key_a, STALE)
            assert tt.get_session_cwd(key_a) == STALE
        with _profile_runtime_scope(home_b, prepared_secret_scope={}):
            tt.record_session_cwd(key_b, STALE)
            assert Path(_configured_terminal_cwd()) == finances_b
            asyncio.run(runner_b._handle_reset_command(
                MessageEvent(text="/new", source=source_b, message_id="m1")
            ))
            assert tt.get_session_cwd(key_b) is None
            assert tt._resolve_command_cwd(
                workdir=None, default_cwd=_configured_terminal_cwd(), session_key=key_b
            ) == str(finances_b)
        # Back to lane A: B's reset must not have touched it.
        with _profile_runtime_scope(home_a, prepared_secret_scope={}):
            assert Path(_configured_terminal_cwd()) == finances_a
            assert tt.get_session_cwd(key_a) == STALE
            assert tt._resolve_command_cwd(
                workdir=None, default_cwd=_configured_terminal_cwd(), session_key=key_a
            ) == STALE
            asyncio.run(runner_a._handle_reset_command(
                MessageEvent(text="/new", source=source_a, message_id="m2")
            ))
            assert tt.get_session_cwd(key_a) is None
            assert tt._resolve_command_cwd(
                workdir=None, default_cwd=_configured_terminal_cwd(), session_key=key_a
            ) == str(finances_a)
    finally:
        set_multiplex_active(False)
