"""Agent adapters: the profile and durable step clock come from the session row."""

from types import SimpleNamespace

import scheduler.agent_hooks as hooks


class _DB:
    def __init__(self, row):
        self._row = row

    def get_session(self, session_id):
        return self._row


class _Engine:
    active = True

    def __init__(self):
        self.ensure_kwargs = {}
        self.turn_pre_kwargs = {}

    def ensure_session(self, session_id, **kwargs):
        self.ensure_kwargs = kwargs

    def turn_pre(self, session_id, **kwargs):
        self.turn_pre_kwargs = kwargs


def test_turn_start_seeds_profile_and_step_clock_from_the_session_row(monkeypatch):
    engine = _Engine()
    monkeypatch.setattr(hooks, "_engine", lambda: engine)
    agent = SimpleNamespace(
        session_id="s", _user_turn_count=3, _current_turn_id="turn-a",
        _session_db=_DB({"step_seq": 17, "profile_name": "work"}),
    )
    hooks.on_turn_start(agent)
    assert engine.ensure_kwargs["step_seq"] == 17
    assert engine.ensure_kwargs["profile_name"] == "work"
    assert engine.ensure_kwargs["turn_seq"] == 2
    assert engine.turn_pre_kwargs["turn_seq"] == 3
    assert engine.turn_pre_kwargs["turn_id"] == "turn-a"
