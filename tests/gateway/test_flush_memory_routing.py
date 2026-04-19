"""Regression: post-turn/background flush-agent construction must carry the
session's ``model.routes`` bundle forward.

If ``_flush_memories_for_session`` calls ``_resolve_session_agent_runtime``
without a ``source``, ``_build_routing_context`` returns ``{}`` and
``apply_route`` short-circuits at ``smart_model_routing.py``. The flush
agent then inherits ``model.default`` + ``model.base_url`` from config
instead of the owner/hub_peer route — so a routed turn whose main agent
ran at ``(slate-3, litellm-3)`` spawns a flush agent at
``(slate-1, litellm-1)``, and its auxiliary compression / memory calls
can end up 401'ing against an integration key scoped to a different
model.

These tests pin the plumbing: the source is recovered from the session
entry when the caller omits it, and forwarded through to the resolver
so ``model.routes`` can fire.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_dotenv(monkeypatch):
    fake = types.ModuleType("dotenv")
    fake.load_dotenv = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "dotenv", fake)


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
    return runner


_TRANSCRIPT = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "a real conversation"},
    {"role": "assistant", "content": "acknowledged"},
]


class TestFlushSourceRecovery:
    def _seed(self, monkeypatch):
        tmp_agent = MagicMock()
        fake_run_agent = types.ModuleType("run_agent")
        fake_run_agent.AIAgent = MagicMock(return_value=tmp_agent)
        monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

        runner = _make_runner()
        runner.session_store.load_transcript.return_value = _TRANSCRIPT
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("slate-1", {"api_key": "k", "base_url": "u", "provider": "custom"})
        )
        runner._cleanup_agent_resources = MagicMock()
        return runner, tmp_agent

    def test_recovers_source_from_session_entry_when_not_passed(self, monkeypatch):
        runner, _ = self._seed(monkeypatch)
        entry = MagicMock()
        entry.origin = MagicMock(name="source_from_entry")
        runner.session_store._entries = {"agent:main:telegram:dm:123": entry}

        runner._flush_memories_for_session(
            "session_abc", session_key="agent:main:telegram:dm:123"
        )

        runner._resolve_session_agent_runtime.assert_called_once()
        _, kwargs = runner._resolve_session_agent_runtime.call_args
        assert kwargs.get("source") is entry.origin, (
            "source must be recovered from session_store._entries so "
            "apply_route can fire model.routes for the flush agent"
        )
        assert kwargs.get("session_key") == "agent:main:telegram:dm:123"

    def test_prefers_explicit_source_over_store_lookup(self, monkeypatch):
        runner, _ = self._seed(monkeypatch)
        stale_entry = MagicMock()
        stale_entry.origin = MagicMock(name="stale_source")
        runner.session_store._entries = {"agent:main:telegram:dm:123": stale_entry}

        explicit = MagicMock(name="explicit_source")
        runner._flush_memories_for_session(
            "session_abc",
            session_key="agent:main:telegram:dm:123",
            source=explicit,
        )

        _, kwargs = runner._resolve_session_agent_runtime.call_args
        assert kwargs.get("source") is explicit

    def test_missing_session_key_does_not_crash(self, monkeypatch):
        runner, _ = self._seed(monkeypatch)
        runner.session_store._entries = {}
        # No session_key, no source — still resolves (source=None), doesn't raise
        runner._flush_memories_for_session("session_abc")
        runner._resolve_session_agent_runtime.assert_called_once()
        _, kwargs = runner._resolve_session_agent_runtime.call_args
        assert kwargs.get("source") is None

    def test_async_wrapper_forwards_source(self, monkeypatch):
        import asyncio

        runner = _make_runner()
        runner._flush_memories_for_session = MagicMock()

        entry_source = MagicMock(name="entry_source")
        asyncio.run(
            runner._async_flush_memories(
                "session_abc", "agent:main:telegram:dm:123", entry_source
            )
        )
        # ``run_in_executor`` passes positional args; verify all three made it through.
        _args, _ = runner._flush_memories_for_session.call_args
        assert _args == ("session_abc", "agent:main:telegram:dm:123", entry_source)
