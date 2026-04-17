"""Tests for MemoryProvider.on_session_end() dispatch on gateway session expiry (#11205).

Verifies that the gateway's flush path invokes the memory provider lifecycle
hook on the live cached agent's memory manager when a session ends — matching
the contract the CLI graceful shutdown path already honors.
"""

import sys
import threading
import types
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    """gateway.run imports dotenv at module level; stub it so tests run without the package."""
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


_TRANSCRIPT = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi there"},
    {"role": "user", "content": "remember: my name is Alice"},
    {"role": "assistant", "content": "Got it, Alice!"},
]


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner.adapters = {}
    runner.hooks = MagicMock()
    runner.session_store = MagicMock()
    runner.session_store.load_transcript.return_value = _TRANSCRIPT
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    return runner


def _patched_flush_env(monkeypatch):
    """Return a context-manager-friendly dict of patches for the flush path."""
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


class TestOnSessionEndDispatch:
    """The flush path must dispatch on_session_end to the cached agent's memory manager."""

    def test_on_session_end_called_for_cached_agent(self, monkeypatch):
        """When an agent is in _agent_cache, its memory_manager.on_session_end fires."""
        runner = _make_runner()
        cached_agent = MagicMock()
        cached_agent._memory_manager = MagicMock()
        runner._agent_cache["agent:main:slack:dm:42"] = (cached_agent, "sig")

        _patched_flush_env(monkeypatch)
        with (
            patch.object(runner, "_resolve_session_agent_runtime",
                         return_value=("test-model", {"api_key": "k"})),
            patch.object(runner, "_cleanup_agent_resources"),
        ):
            runner._flush_memories_for_session("session_xyz", "agent:main:slack:dm:42")

        cached_agent._memory_manager.on_session_end.assert_called_once()
        # Messages should be cleaned (role + content only) from the transcript
        called_msgs = cached_agent._memory_manager.on_session_end.call_args.args[0]
        assert called_msgs == _TRANSCRIPT

    def test_on_session_end_called_for_running_agent(self, monkeypatch):
        """If the agent is mid-turn (in _running_agents), still dispatch the hook."""
        runner = _make_runner()
        running_agent = MagicMock()
        running_agent._memory_manager = MagicMock()
        runner._running_agents["agent:main:telegram:dm:9"] = running_agent

        _patched_flush_env(monkeypatch)
        with (
            patch.object(runner, "_resolve_session_agent_runtime",
                         return_value=("test-model", {"api_key": "k"})),
            patch.object(runner, "_cleanup_agent_resources"),
        ):
            runner._flush_memories_for_session("session_run", "agent:main:telegram:dm:9")

        running_agent._memory_manager.on_session_end.assert_called_once()

    def test_no_dispatch_when_no_cached_agent(self, monkeypatch):
        """When no agent is cached for the session_key, the hook isn't called (no-op)."""
        runner = _make_runner()
        _patched_flush_env(monkeypatch)
        with (
            patch.object(runner, "_resolve_session_agent_runtime",
                         return_value=("test-model", {"api_key": "k"})),
            patch.object(runner, "_cleanup_agent_resources"),
        ):
            # Should not raise
            runner._flush_memories_for_session("session_no_agent", "agent:main:slack:dm:99")

    def test_provider_failure_does_not_break_flush(self, monkeypatch):
        """A misbehaving on_session_end shouldn't block the bespoke flush agent."""
        runner = _make_runner()
        cached_agent = MagicMock()
        cached_agent._memory_manager = MagicMock()
        cached_agent._memory_manager.on_session_end.side_effect = RuntimeError("boom")
        runner._agent_cache["agent:main:discord:dm:1"] = (cached_agent, "sig")

        _patched_flush_env(monkeypatch)
        captured_agent = {}

        def _fake_ai_agent(*args, **kwargs):
            agent = MagicMock()
            captured_agent["instance"] = agent
            return agent

        sys.modules["run_agent"].AIAgent = _fake_ai_agent

        with (
            patch.object(runner, "_resolve_session_agent_runtime",
                         return_value=("test-model", {"api_key": "k"})),
            patch.object(runner, "_cleanup_agent_resources"),
            patch.dict("sys.modules", {"tools.memory_tool": MagicMock()}),
        ):
            # Must not raise even though provider hook blew up
            runner._flush_memories_for_session("session_brk", "agent:main:discord:dm:1")

        # The bespoke flush agent should still have run
        assert "instance" in captured_agent
        captured_agent["instance"].run_conversation.assert_called_once()

    def test_cron_session_skips_session_end_too(self, monkeypatch):
        """Cron sessions skip the entire flush — including on_session_end."""
        runner = _make_runner()
        cached_agent = MagicMock()
        cached_agent._memory_manager = MagicMock()
        runner._agent_cache["agent:main:cron:job:1"] = (cached_agent, "sig")

        runner._flush_memories_for_session("cron_daily_20260417", "agent:main:cron:job:1")

        cached_agent._memory_manager.on_session_end.assert_not_called()

    def test_no_session_key_skips_lookup(self, monkeypatch):
        """When session_key is not provided, no cached-agent lookup occurs."""
        runner = _make_runner()
        # Even if there's a stray entry, it shouldn't be matched without a key.
        cached_agent = MagicMock()
        cached_agent._memory_manager = MagicMock()
        runner._agent_cache["agent:main:slack:dm:foo"] = (cached_agent, "sig")

        _patched_flush_env(monkeypatch)
        with (
            patch.object(runner, "_resolve_session_agent_runtime",
                         return_value=("test-model", {"api_key": "k"})),
            patch.object(runner, "_cleanup_agent_resources"),
        ):
            runner._flush_memories_for_session("session_nokey", None)

        cached_agent._memory_manager.on_session_end.assert_not_called()
