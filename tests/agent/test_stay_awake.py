"""Invariant tests for the stay-awake inhibitor (agent/stay_awake.py).

Contract under test:
- disabled → zero side effects (no process spawned, no OS call);
- turn_scope refcounts a single shared inhibitor across nested/concurrent turns
  (Windows SetThreadExecutionState is not nestable, so exactly one enter/exit pair
  must wrap the whole overlap);
- config default keeps the feature off.

Host-specific spawn behaviour (caffeinate/systemd-inhibit argv) is intentionally
not tested by faking ``platform.system()`` — repo policy tests OS behaviour on
that OS only.
"""
from unittest.mock import patch

import agent.stay_awake as stay_awake
from agent.stay_awake import StayAwake, turn_scope


def test_disabled_is_complete_noop():
    with patch("agent.stay_awake.subprocess.Popen") as popen:
        with StayAwake(enabled=False) as sa:
            assert sa._process is None
        popen.assert_not_called()


def test_turn_scope_refcounts_one_shared_inhibitor(monkeypatch):
    events = []

    class Fake:
        def __init__(self, enabled=False):
            pass

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *a):
            events.append("exit")

    monkeypatch.setattr(stay_awake, "StayAwake", Fake)
    with turn_scope(enabled=True):
        with turn_scope(enabled=True):
            assert events == ["enter"]  # nested turn reuses the live inhibitor
        assert events == ["enter"]  # inner exit must NOT release it
    assert events == ["enter", "exit"]  # last turn out releases exactly once


def test_turn_scope_defaults_off_from_config(monkeypatch):
    monkeypatch.setattr(stay_awake, "_config_enabled", lambda: False)

    class Explode:
        def __init__(self, enabled=False):
            raise AssertionError("inhibitor must not start when disabled")

    monkeypatch.setattr(stay_awake, "StayAwake", Explode)
    with turn_scope():
        pass
