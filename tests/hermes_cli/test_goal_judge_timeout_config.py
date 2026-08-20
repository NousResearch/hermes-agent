"""`auxiliary.goal_judge.timeout` must actually reach the judge (#91022).

The key is declared in DEFAULT_CONFIG (60s) and surfaces in the auxiliary
config UI, but the judge path hardcoded `DEFAULT_JUDGE_TIMEOUT = 30.0` and
never read it — a user raising the timeout for a slow-but-healthy reasoning
endpoint got no effect, and the loop auto-paused on misleading transport
failures pointing at provider/key. The reader mirrors `_goal_judge_max_tokens`
(max_tokens is wired, timeout was not).
"""

import pytest

import hermes_cli.goals as goals
from hermes_cli.goals import DEFAULT_JUDGE_TIMEOUT, _goal_judge_timeout


def _patch_config(monkeypatch, cfg):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)


def test_reader_returns_configured_timeout(monkeypatch):
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"timeout": 120}}})
    assert _goal_judge_timeout() == 120.0


def test_reader_falls_back_when_key_absent(monkeypatch):
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"max_tokens": 512}}})
    assert _goal_judge_timeout() == DEFAULT_JUDGE_TIMEOUT


def test_reader_falls_back_on_non_positive_or_garbage(monkeypatch):
    for bad in (0, -5, "fast", None):
        _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"timeout": bad}}})
        assert _goal_judge_timeout() == DEFAULT_JUDGE_TIMEOUT


def test_judge_goal_resolves_timeout_from_config(monkeypatch):
    """The wiring that matters: a caller that passes no timeout (all four
    production call sites) must have the configured value reach call_llm."""
    captured = {}

    class _FakeAux:
        @staticmethod
        def call_llm(*args, **kwargs):
            captured.update(kwargs)
            raise TimeoutError("simulated transport timeout")

    import sys

    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _FakeAux)
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"timeout": 120}}})

    verdict, _, _, _, transport_failed = goals.judge_goal(
        "do the thing", "here is a substantive response to evaluate"
    )

    assert captured.get("timeout") == 120.0
    # Fail-open contract preserved: a timeout is still a transport failure.
    assert verdict == "continue"
    assert transport_failed is True


def test_judge_goal_explicit_timeout_wins(monkeypatch):
    captured = {}

    class _FakeAux:
        @staticmethod
        def call_llm(*args, **kwargs):
            captured.update(kwargs)
            raise TimeoutError("simulated transport timeout")

    import sys

    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", _FakeAux)
    _patch_config(monkeypatch, {"auxiliary": {"goal_judge": {"timeout": 120}}})

    goals.judge_goal("g", "r", timeout=7.5)
    assert captured.get("timeout") == 7.5
