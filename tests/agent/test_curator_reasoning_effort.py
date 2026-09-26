"""The curator review fork must honor auxiliary.curator.reasoning_effort.

The desktop and ``hermes model`` write that key. The review used to build its
agent from the global ``agent.reasoning_effort`` only. See #122379.
"""

import sys
import types

import agent.curator as curator


def _capture_review(monkeypatch, config):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop")

        def close(self):
            pass

    fake = types.ModuleType("run_agent")
    fake.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake)
    monkeypatch.setattr(
        curator, "_resolve_review_provider",
        lambda: ({}, "claude-opus-5-5", "anthropic", {}),
    )
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    result = curator._run_llm_review("x")
    return captured, result


def test_curator_review_uses_auxiliary_reasoning_effort(monkeypatch):
    captured, result = _capture_review(monkeypatch, {
        "agent": {"reasoning_effort": "low"},
        "auxiliary": {"curator": {"provider": "anthropic", "reasoning_effort": "high"}},
    })

    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}
    assert "stop" in result["error"]


def test_curator_review_none_disables_reasoning(monkeypatch):
    captured, _result = _capture_review(monkeypatch, {
        "agent": {"reasoning_effort": "high"},
        "auxiliary": {"curator": {"reasoning_effort": "none"}},
    })

    assert captured["reasoning_config"] == {"enabled": False}


def test_curator_review_falls_back_to_global_effort_when_unset(monkeypatch):
    captured, _result = _capture_review(monkeypatch, {
        "agent": {"reasoning_effort": "low"},
        "auxiliary": {"curator": {"provider": "anthropic"}},
    })

    assert captured["reasoning_config"] == {"enabled": True, "effort": "low"}
