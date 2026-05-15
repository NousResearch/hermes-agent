"""Integration tests for the gateway ``/profile`` slash command.

After the agent-vs-profile reconciliation, ``/profile`` is the single
multi-agent entry point:

  /profile               → show the session's active profile + host info
  /profile ls            → list every available profile
  /profile <name>        → bind this session to <name>
  /profile default       → reset to the default profile

It replaces the read-only ``/profile`` and the separate ``/agent``
command that existed during early refactor iterations.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import pytest

from gateway.agent_registry import reset_default_registry
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Lay out ~/.hermes with default + coder + data-sci."""
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


def _make_event(text: str) -> MessageEvent:
    src = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user1",
        user_name="alice",
    )
    return MessageEvent(text=text, source=src)


class _RunnerStub:
    """Minimal stub of GatewayRunner exposing only what _handle_profile_command needs."""

    def __init__(self, sessions_dir: Path):
        self.config = GatewayConfig()
        self.session_store = SessionStore(sessions_dir, self.config)
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = Lock()


def _call_handler(runner: _RunnerStub, event: MessageEvent) -> str:
    from gateway.run import GatewayRunner

    handler = GatewayRunner._handle_profile_command
    return asyncio.get_event_loop().run_until_complete(
        handler(runner, event)  # type: ignore[arg-type]
    )


@pytest.fixture
def runner(fake_root, tmp_path):
    return _RunnerStub(tmp_path / "sessions")


class TestRegistry:
    def test_profile_registered(self):
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command

        cmd = resolve_command("profile")
        assert cmd is not None
        assert cmd.name == "profile"
        assert "profile" in GATEWAY_KNOWN_COMMANDS

    def test_agent_command_removed(self):
        """The standalone /agent command was merged into /profile."""
        from hermes_cli.commands import resolve_command

        # /agent should no longer resolve — only /agents (plural, running tasks)
        cmd = resolve_command("agent")
        assert cmd is None or cmd.name != "agent"


class TestBareProfile:
    def test_bare_shows_default(self, runner):
        result = _call_handler(runner, _make_event("/profile"))
        assert "Active profile" in result
        assert "default" in result
        # Hint to discover more options
        assert "/profile ls" in result

    def test_bare_shows_current_after_switch(self, runner):
        # Initialise the session record first.
        _call_handler(runner, _make_event("/profile"))
        _call_handler(runner, _make_event("/profile coder"))
        result = _call_handler(runner, _make_event("/profile"))
        assert "coder" in result
        # Default is no longer the active marker
        assert "Active profile (this session): coder" in result


class TestProfileList:
    def test_ls_lists_all(self, runner):
        result = _call_handler(runner, _make_event("/profile ls"))
        assert "Available profiles:" in result
        assert "default" in result
        assert "coder" in result
        assert "data-sci" in result
        assert "Active: default" in result

    def test_list_alias_same_as_ls(self, runner):
        result = _call_handler(runner, _make_event("/profile list"))
        assert "Available profiles:" in result
        assert "coder" in result


class TestProfileSwitch:
    def test_switch_to_known_profile(self, runner):
        _call_handler(runner, _make_event("/profile"))
        result = _call_handler(runner, _make_event("/profile coder"))
        assert "Switched profile to: coder" in result

        source = _make_event("ignored").source
        # New multi-agent model: the binding lives at chat-level so
        # next-message resolution finds it regardless of session_key.
        assert runner.session_store.get_chat_agent(source) == "coder"

    def test_switch_invalidates_old_agent_cache_slot(self, runner):
        # The cache key is per (chat, agent) now — switching from
        # default to coder evicts default's slot (the prior binding),
        # not coder's (which is independent and fresh).
        _call_handler(runner, _make_event("/profile"))
        source = _make_event("ignored").source
        default_key = runner.session_store._generate_session_key(
            source, agent_name="default"
        )
        runner._agent_cache[default_key] = ("sentinel-agent", "sig-123")

        _call_handler(runner, _make_event("/profile coder"))
        assert default_key not in runner._agent_cache, (
            "switching away from default must evict its cache slot so a "
            "later switch back doesn't reuse a stale construction"
        )

    def test_switch_to_unknown_profile(self, runner):
        result = _call_handler(runner, _make_event("/profile nope"))
        assert "Unknown profile" in result
        assert "nope" in result
        assert "coder" in result  # available list referenced

    def test_already_on_target_short_circuits(self, runner):
        _call_handler(runner, _make_event("/profile"))
        _call_handler(runner, _make_event("/profile coder"))
        result = _call_handler(runner, _make_event("/profile coder"))
        assert "Already on profile 'coder'" in result

    def test_switch_to_default_resets(self, runner):
        _call_handler(runner, _make_event("/profile"))
        _call_handler(runner, _make_event("/profile coder"))
        result = _call_handler(runner, _make_event("/profile default"))
        assert "Switched profile to: default" in result
        source = _make_event("ignored").source
        assert runner.session_store.get_chat_agent(source) == "default"
