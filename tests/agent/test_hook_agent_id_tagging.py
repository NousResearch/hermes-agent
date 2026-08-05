"""Lifecycle hooks carry ``agent_id`` only when an agent profile is bound.

Multi-agent installs want to know *which* agent fired a ``pre_tool_call`` or
``subagent_stop`` hook.  Single-agent installs have no routed profile, and
their hook payloads must stay byte-identical to what observers and plugins
already receive — an ``agent_id: None`` key on every payload is a silent
contract change for every existing consumer.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.profile import AgentProfile, use_profile


@pytest.fixture(autouse=True)
def _no_leaked_profile():
    """Never leak an active profile into later tests in this session."""
    import agent.profile as _live

    yield
    _live._current_agent_profile.set(None)


class TestPreToolCallAgentId:
    def _observed_payload(self, monkeypatch):
        from hermes_cli import observability
        from hermes_cli.plugins import get_pre_tool_call_directive

        observed = []
        monkeypatch.setattr(
            observability,
            "observe_lifecycle",
            lambda hook_name, **kwargs: observed.append((hook_name, kwargs)),
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.invoke_hook",
            lambda hook_name, **kwargs: [],
        )
        get_pre_tool_call_directive(
            "write_file",
            {"path": "README.md"},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
        )
        assert len(observed) == 1
        return observed[0][1]

    def test_no_profile_bound_omits_agent_id(self, monkeypatch):
        assert "agent_id" not in self._observed_payload(monkeypatch)

    def test_bound_profile_tags_payload(self, monkeypatch):
        with use_profile(AgentProfile(id="coder")):
            payload = self._observed_payload(monkeypatch)
        assert payload["agent_id"] == "coder"


class TestSubagentStopAgentId:
    def _invoke(self, monkeypatch):
        from tools import delegate_tool

        hook = Mock()
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
        parent = SimpleNamespace(
            session_id="parent-1",
            _current_turn_id="turn-1",
            _memory_manager=None,
        )
        child = SimpleNamespace(session_id="child-1")
        delegate_tool._finalize_child_results(
            [
                {
                    "task_index": 0,
                    "status": "completed",
                    "summary": "done",
                    "duration_seconds": 0.25,
                    "_child_role": "leaf",
                    "_child_cost_usd": 0.0,
                }
            ],
            [{"goal": "do a thing"}],
            [(0, {"goal": "do a thing"}, child)],
            parent,
        )
        hook.assert_called_once()
        return hook.call_args.kwargs

    def test_no_profile_bound_omits_agent_id(self, monkeypatch):
        assert "agent_id" not in self._invoke(monkeypatch)

    def test_bound_profile_tags_payload(self, monkeypatch):
        with use_profile(AgentProfile(id="researcher")):
            kwargs = self._invoke(monkeypatch)
        assert kwargs["agent_id"] == "researcher"
