"""Tests for the four agent.* system-prompt injection toggles (issue #37253).

``help_guidance`` / ``profile_hint`` / ``timestamp_line`` / ``environment_hints``
must flow from config.yaml's ``agent:`` section onto ``agent._<key>`` — default
True, individually overridable — through the same default-on boolean gate loop
that carries ``task_completion_guidance`` et al. The prompt-side suppression is
covered by tests/agent/test_system_prompt.py::TestInjectionToggles; this file
pins the config wiring itself so a key dropped from the gate loop fails here.
"""
from types import SimpleNamespace

from agent.agent_init import _apply_agent_section

TOGGLE_KEYS = ("help_guidance", "profile_hint", "timestamp_line", "environment_hints")


def _apply(**agent_cfg):
    # run_budget_seconds is the one agent attribute _apply_agent_section reads
    # before it writes; None lets the config path populate it.
    agent = SimpleNamespace(run_budget_seconds=None)
    # Skip the off-thread env-probe warm: nothing here needs the probe, and
    # the warm would spawn subprocesses inside the unit test.
    agent_cfg.setdefault("environment_probe", False)
    _apply_agent_section(agent, {"agent": agent_cfg})
    return agent


def test_toggles_default_true():
    agent = _apply()
    for key in TOGGLE_KEYS:
        assert getattr(agent, f"_{key}") is True, key


def test_all_toggles_disabled_together():
    agent = _apply(**{key: False for key in TOGGLE_KEYS})
    for key in TOGGLE_KEYS:
        assert getattr(agent, f"_{key}") is False, key


def test_each_toggle_is_independent():
    for key in TOGGLE_KEYS:
        agent = _apply(**{key: False})
        assert getattr(agent, f"_{key}") is False, key
        for other in TOGGLE_KEYS:
            if other != key:
                assert getattr(agent, f"_{other}") is True, other
