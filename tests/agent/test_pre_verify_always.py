"""Regression coverage for `agent.pre_verify_always` — the opt-in that lets
`pre_verify` hooks run on turns that edited no files, plus the preflush that
makes the current turn's tool results readable by those hooks.

The gate itself lives in `agent/turn_stop_gates.py`; the config reader lives in
`agent/verify_hooks.py` and is covered alongside the other agent-config readers
in `tests/agent/test_verify_hooks.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import turn_stop_gates, verify_hooks


@pytest.fixture
def agent():
    return SimpleNamespace(
        _turn_file_mutation_paths=set(),
        _resolved_is_coding=False,
        session_id="pre-verify-test",
        platform="cli",
        model="test/model",
        _flush_messages_to_session_db=MagicMock(),
    )


def _wire(monkeypatch, *, always: bool, hook_result="keep going", has_hook=True):
    """Patch the three lazily-imported collaborators the gate resolves."""
    calls = {}

    def _get_message(**kwargs):
        calls.update(kwargs)
        calls["invoked"] = True
        return hook_result

    monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda *a, **k: always)
    monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda *a, **k: 3)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: has_hook)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_pre_verify_continue_message", _get_message
    )
    return calls


class TestGateScope:
    def test_no_edits_and_option_off_does_not_invoke_hook(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=False)
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 0) is None
        assert "invoked" not in calls

    def test_no_edits_and_option_on_invokes_hook(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=True)
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 0) == "keep going"
        assert calls["invoked"] is True
        # The hook still receives changed_paths so edit-only hooks can bail out.
        assert calls["changed_paths"] == []

    def test_file_edit_behaviour_unchanged_when_option_off(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=False)
        agent._turn_file_mutation_paths = {"src/app.py"}
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 0) == "keep going"
        assert calls["changed_paths"] == ["src/app.py"]

    def test_hook_returning_none_produces_no_nudge(self, agent, monkeypatch):
        _wire(monkeypatch, always=True, hook_result=None)
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 0) is None

    def test_no_registered_hook_short_circuits(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=True, has_hook=False)
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 0) is None
        assert "invoked" not in calls

    def test_attempt_budget_still_bounds_the_gate(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=True)
        assert turn_stop_gates._pre_verify_nudge(agent, "done", 3) is None
        assert "invoked" not in calls


class TestPreflush:
    def test_current_turn_is_flushed_before_the_hook_runs(self, agent, monkeypatch):
        """A hook deciding on what this turn already did reads it from the session
        store, so the turn must be persisted before the hook is asked."""
        order = []
        agent._flush_messages_to_session_db = MagicMock(
            side_effect=lambda *a, **k: order.append("flush")
        )

        def _get_message(**kwargs):
            order.append("hook")
            return "keep going"

        monkeypatch.setattr(verify_hooks, "pre_verify_always", lambda *a, **k: True)
        monkeypatch.setattr(verify_hooks, "max_verify_nudges", lambda *a, **k: 3)
        monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: True)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_verify_continue_message", _get_message
        )

        messages = [{"role": "tool", "content": "result"}]
        turn_stop_gates._pre_verify_nudge(
            agent, "done", 0, messages=messages, conversation_history=None
        )
        assert order == ["flush", "hook"]
        agent._flush_messages_to_session_db.assert_called_once_with(messages, None)

    def test_flush_failure_does_not_end_the_turn(self, agent, monkeypatch):
        calls = _wire(monkeypatch, always=True)
        agent._flush_messages_to_session_db = MagicMock(
            side_effect=RuntimeError("db locked")
        )
        nudge = turn_stop_gates._pre_verify_nudge(
            agent, "done", 0, messages=[{"role": "tool"}], conversation_history=None
        )
        assert nudge == "keep going"
        assert calls["invoked"] is True

    def test_no_messages_means_no_flush(self, agent, monkeypatch):
        _wire(monkeypatch, always=True)
        turn_stop_gates._pre_verify_nudge(agent, "done", 0)
        agent._flush_messages_to_session_db.assert_not_called()


class TestConfigReader:
    def test_default_is_off(self):
        assert verify_hooks.pre_verify_always({}) is False
        assert verify_hooks.pre_verify_always({"agent": {}}) is False

    def test_reads_truthy_values(self):
        assert verify_hooks.pre_verify_always({"agent": {"pre_verify_always": True}})
        assert verify_hooks.pre_verify_always({"agent": {"pre_verify_always": "yes"}})

    def test_reads_falsey_values(self):
        assert not verify_hooks.pre_verify_always({"agent": {"pre_verify_always": False}})
        assert not verify_hooks.pre_verify_always({"agent": {"pre_verify_always": "off"}})
