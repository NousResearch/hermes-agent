"""Tests for ``GatewayRunner._resolve_turn_agent``.

Phase 6 of the multi-agent gateway refactor; revised in the
chat-level-binding follow-up.  This is the choke point that decides
which agent (Hermes profile) handles a given turn — it must honour
``@<name>`` mentions, fall back to the chat's persisted binding (set
by ``/profile <name>``), and finally to the default agent.

The signature is ``(message, source)`` — session_key is intentionally
NOT a parameter because under the multi-agent model the session_key
depends on the agent resolved here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.agent_registry import reset_default_registry
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    (root / "profiles").mkdir()
    (root / "profiles" / "coder").mkdir()
    (root / "profiles" / "data-sci").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reset_default_registry()
    yield root
    reset_default_registry()


@pytest.fixture
def runner_stub(fake_root, tmp_path):
    """Lightweight stand-in for GatewayRunner.

    The real __init__ has ~60 parameters and pulls in adapters / SQLite /
    hooks. We only need ``session_store`` so we can exercise the resolver
    in isolation. Borrowing the unbound method keeps coverage on the
    production code path.
    """

    class _Stub:
        pass

    stub = _Stub()
    stub.config = GatewayConfig()
    stub.session_store = SessionStore(tmp_path / "sessions", stub.config)
    return stub


def _resolve(runner_stub, message: str, source):
    """Invoke the production method bound to the stub."""
    return GatewayRunner._resolve_turn_agent(runner_stub, message, source)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user1",
        user_name="alice",
    )


class TestNoMentionFallsBackToChatBinding:
    def test_default_when_chat_has_no_binding(self, runner_stub):
        name, _home, msg = _resolve(runner_stub, "hello", _source())
        assert name == "default"
        assert msg == "hello"

    def test_falls_back_to_chat_binding(self, runner_stub):
        runner_stub.session_store.set_chat_agent(_source(), "coder")
        name, _home, msg = _resolve(runner_stub, "do the thing", _source())
        assert name == "coder"
        assert msg == "do the thing"

    def test_unknown_stored_agent_falls_back_to_default(self, runner_stub):
        # Force an invalid value via direct mutation — simulates a profile
        # that was deleted while the binding persisted its name.
        from gateway.session import build_chat_key

        chat_key = build_chat_key(_source())
        runner_stub.session_store._chat_bindings_loaded = True
        runner_stub.session_store._chat_bindings[chat_key] = "ghost"
        name, _home, msg = _resolve(runner_stub, "hi", _source())
        assert name == "default"
        assert msg == "hi"


class TestMentionOverridesStored:
    def test_mention_takes_priority(self, runner_stub):
        runner_stub.session_store.set_chat_agent(_source(), "coder")
        name, _home, msg = _resolve(
            runner_stub, "@data-sci analyze X", _source()
        )
        assert name == "data-sci"
        assert msg == "analyze X"

    def test_mention_does_not_mutate_binding(self, runner_stub):
        runner_stub.session_store.set_chat_agent(_source(), "coder")
        _resolve(runner_stub, "@data-sci ping", _source())
        # Chat binding is unchanged — @ is per-turn only
        assert runner_stub.session_store.get_chat_agent(_source()) == "coder"

    def test_unknown_mention_passes_through(self, runner_stub):
        runner_stub.session_store.set_chat_agent(_source(), "coder")
        name, _home, msg = _resolve(
            runner_stub, "@nonexistent hello", _source()
        )
        # Unknown mentions are NOT routed — text passes through and the
        # chat binding wins.
        assert name == "coder"
        assert msg == "@nonexistent hello"


class TestHomePath:
    def test_default_home_is_hermes_root(self, runner_stub, fake_root):
        _name, home, _msg = _resolve(runner_stub, "hi", _source())
        assert home == fake_root

    def test_named_agent_home_is_profile_dir(self, runner_stub, fake_root):
        runner_stub.session_store.set_chat_agent(_source(), "coder")
        _name, home, _msg = _resolve(runner_stub, "hi", _source())
        assert home == fake_root / "profiles" / "coder"

    def test_mentioned_agent_home(self, runner_stub, fake_root):
        _name, home, _msg = _resolve(runner_stub, "@data-sci foo", _source())
        assert home == fake_root / "profiles" / "data-sci"
