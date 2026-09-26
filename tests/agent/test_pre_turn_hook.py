"""Generic ``pre_turn`` short-circuit seam: a plugin-provided result dict replaces the turn.

Covers the hook dispatch contract declared in hermes_cli/plugins.py (first non-None dict
returned by any pre_turn callback is used AS the turn result; None falls through).
"""

from types import SimpleNamespace
from unittest.mock import Mock
import pytest


def _patch_hooks(monkeypatch, results):
    import hermes_cli.lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, "has_hook", lambda name: name == "pre_turn")
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kw: list(results))


def test_pre_turn_returned_dict_short_circuits_the_turn(monkeypatch):
    from agent.conversation_loop import _run_conversation_turn

    short = {"final_response": "exact", "substituted": True, "completed": True}
    _patch_hooks(monkeypatch, [None, short])
    gate = Mock(side_effect=AssertionError("normal path should not run"))
    monkeypatch.setattr("agent.conversation_loop.begin_fast_mode_turn", gate)
    result = _run_conversation_turn(SimpleNamespace(), "only reply: exact")
    assert result is short
    gate.assert_not_called()


def test_pre_turn_none_results_fall_through_to_normal_path(monkeypatch):
    from agent.conversation_loop import _run_conversation_turn

    _patch_hooks(monkeypatch, [None, None])

    class NormalRouteReached(Exception):
        pass

    gate = Mock(side_effect=NormalRouteReached)
    monkeypatch.setattr("agent.conversation_loop.begin_fast_mode_turn", gate)
    with pytest.raises(NormalRouteReached):
        _run_conversation_turn(SimpleNamespace(), "普通问候")
    gate.assert_called_once()


def test_pre_turn_hook_failure_is_fail_open(monkeypatch):
    from agent.conversation_loop import _run_conversation_turn
    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "has_hook", lambda name: True)
    monkeypatch.setattr(
        lifecycle, "invoke_hook", lambda name, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    class NormalRouteReached(Exception):
        pass

    gate = Mock(side_effect=NormalRouteReached)
    monkeypatch.setattr("agent.conversation_loop.begin_fast_mode_turn", gate)
    with pytest.raises(NormalRouteReached):
        _run_conversation_turn(SimpleNamespace(), "普通问候")
    gate.assert_called_once()