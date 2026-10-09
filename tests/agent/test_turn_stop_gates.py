"""Scope of the ``pre_verify`` gate in ``agent/turn_stop_gates.py``.

Edited-code turns by default; text-response stops (no file edits) only when
``agent.pre_verify_text_stops`` is on. The hook itself scopes what it wants to judge.
"""

from __future__ import annotations

from agent import turn_stop_gates


class _Agent:
    _turn_file_mutation_paths: set = set()
    session_id = "s1"
    platform = "cli"
    model = "m"
    _resolved_is_coding = False


def _wire(monkeypatch, *, hook: bool, text_stops: bool, message: str = "keep going"):
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda name: hook and name == "pre_verify")
    monkeypatch.setattr("agent.verify_hooks.max_verify_nudges", lambda config=None: 3)
    monkeypatch.setattr("agent.verify_hooks.pre_verify_text_stops", lambda config=None: text_stops)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_pre_verify_continue_message", lambda **kw: message
    )


def test_text_stop_fires_hook_when_opted_in(monkeypatch):
    _wire(monkeypatch, hook=True, text_stops=True)
    assert turn_stop_gates._pre_verify_nudge(_Agent(), "reply", 0) == "keep going"


def test_text_stop_keeps_gate_closed_by_default(monkeypatch):
    _wire(monkeypatch, hook=True, text_stops=False)
    assert turn_stop_gates._pre_verify_nudge(_Agent(), "reply", 0) is None


def test_edited_turn_still_fires_without_optin(monkeypatch):
    _wire(monkeypatch, hook=True, text_stops=False)
    agent = _Agent()
    agent._turn_file_mutation_paths = {"x.py"}
    assert turn_stop_gates._pre_verify_nudge(agent, "reply", 0) == "keep going"


def test_no_hook_no_nudge(monkeypatch):
    _wire(monkeypatch, hook=False, text_stops=True)
    assert turn_stop_gates._pre_verify_nudge(_Agent(), "reply", 0) is None
