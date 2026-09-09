"""Unit coverage for ``agent/model_switching.py`` — adaptive model switching between tool
iterations.

The initial model (``model.default: auto``) is chosen by the semantic router; this module
decides, purely from per-round signals and an opt-in config ladder, when the loop may
escalate to a harder configured model or descend back to a cheaper one. Switches are
scheduled only at generation boundaries (decided after a tool round, applied at the start
of the next iteration) so a generation is never switched mid-stream and the transcript is
never mutated — prompt-cache and role-alternation invariants hold.

These are behaviour contracts, not snapshots: they pin the escalation/descent *relations*
(config → decision) and never freeze model ids or counts that are expected to change.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.model_switching import (
    ModelSwitchingSettings,
    SwitchState,
    apply_pending_model_switch,
    decide_next_model,
    observe_tool_round,
    resolve_model_switching_settings,
    reset_model_switching_turn,
)


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def test_disabled_by_default():
    for config in (None, {}, {"agent": {}}, {"agent": {"model_switching": {}}}):
        settings = resolve_model_switching_settings(config)
        assert settings.enabled is False
        assert settings.escalate_to is None
        assert settings.descend_to is None


def test_reads_escalate_and_descend_targets():
    settings = resolve_model_switching_settings(
        {"agent": {"model_switching": {"escalate_to": "hard-model", "descend_to": "cheap-model"}}}
    )
    assert settings.enabled is True
    assert settings.escalate_to == "hard-model"
    assert settings.descend_to == "cheap-model"


def test_thresholds_coerce_numeric_strings_and_keep_defaults():
    settings = resolve_model_switching_settings(
        {"agent": {"model_switching": {"escalate_to": "h", "escalate_after_failures": "4"}}}
    )
    assert settings.escalate_after_failures == 4
    assert settings.descend_after_successes == 2  # untouched default
    assert settings.max_switches == 3


def test_typo_threshold_warns_and_falls_back(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        settings = resolve_model_switching_settings(
            {"agent": {"model_switching": {"escalate_to": "h", "escalate_after_failures": "oops"}}}
        )
    assert settings.escalate_after_failures == 3
    assert len(caplog.records) == 1


def test_non_positive_threshold_warns_and_falls_back(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        settings = resolve_model_switching_settings(
            {"agent": {"model_switching": {"escalate_to": "h", "descend_after_successes": 0}}}
        )
    assert settings.descend_after_successes == 2
    assert len(caplog.records) == 1


def test_non_string_escalate_target_disables_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        settings = resolve_model_switching_settings(
            {"agent": {"model_switching": {"escalate_to": 42}}}
        )
    assert settings.enabled is False
    assert len(caplog.records) >= 1


def test_malformed_section_never_raises(caplog):
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        result = resolve_model_switching_settings({"agent": "nonsense"})
    assert result.enabled is False
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        result = resolve_model_switching_settings({"agent": {"model_switching": "nonsense"}})
    assert result.enabled is False
    assert len(caplog.records) >= 1


# ---------------------------------------------------------------------------
# decide_next_model — the pure escalation/descent core
# ---------------------------------------------------------------------------

def test_disabled_settings_never_switch():
    state = SwitchState(consecutive_failures=5, consecutive_successes=5)
    target, new_state = decide_next_model(
        ModelSwitchingSettings(), state, current_model="auto", round_failed=True
    )
    assert target is None
    assert new_state == state  # untouched


def test_escalates_after_consecutive_failures():
    settings = ModelSwitchingSettings(escalate_to="hard", escalate_after_failures=3)
    state = SwitchState()
    target = None
    for _ in range(2):
        target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target is None  # below threshold
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target == "hard"
    assert state.escalated is True


def test_clean_round_resets_failure_streak():
    settings = ModelSwitchingSettings(escalate_to="hard", escalate_after_failures=3)
    state = SwitchState()
    for _ in range(2):
        _, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    _, state = decide_next_model(settings, state, current_model="auto", round_failed=False)
    # clean round reset the streak; two failures now are below the 3-failure threshold
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target is None
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target is None  # still only 2 consecutive failures
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target == "hard"  # 3rd consecutive failure escalates


def test_no_repeat_escalation_when_already_escalated():
    settings = ModelSwitchingSettings(escalate_to="hard", escalate_after_failures=1)
    state = SwitchState()
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target == "hard"
    target, state = decide_next_model(settings, state, current_model="hard", round_failed=True)
    assert target is None


def test_descend_after_clean_successes():
    settings = ModelSwitchingSettings(
        escalate_to="hard", descend_to="cheap", escalate_after_failures=1, descend_after_successes=2
    )
    state = SwitchState()
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target == "hard"
    target, state = decide_next_model(settings, state, current_model="hard", round_failed=False)
    assert target is None
    target, state = decide_next_model(settings, state, current_model="hard", round_failed=False)
    assert target == "cheap"
    assert state.escalated is False


def test_descend_requires_prior_escalation():
    settings = ModelSwitchingSettings(
        escalate_to="hard", descend_to="cheap", descend_after_successes=1
    )
    state = SwitchState()
    target, _ = decide_next_model(settings, state, current_model="auto", round_failed=False)
    assert target is None


def test_failure_resets_success_streak_blocks_descend():
    settings = ModelSwitchingSettings(
        escalate_to="hard", descend_to="cheap", escalate_after_failures=1, descend_after_successes=2
    )
    state = SwitchState()
    _, state = decide_next_model(settings, state, current_model="auto", round_failed=True)  # -> hard
    _, state = decide_next_model(settings, state, current_model="hard", round_failed=False)  # cs=1
    _, state = decide_next_model(settings, state, current_model="hard", round_failed=True)   # cs reset
    # One clean round after the failure is below the 2-clean descend threshold.
    target, state = decide_next_model(settings, state, current_model="hard", round_failed=False)  # cs=1
    assert target is None
    # The second consecutive clean round completes the fresh streak -> descend.
    target, _ = decide_next_model(settings, state, current_model="hard", round_failed=False)  # cs=2
    assert target == "cheap"


def test_no_op_when_already_on_escalation_target():
    settings = ModelSwitchingSettings(escalate_to="hard", escalate_after_failures=1)
    state = SwitchState()
    target, _ = decide_next_model(settings, state, current_model="hard", round_failed=True)
    assert target is None


def test_max_switches_caps_total_switches():
    settings = ModelSwitchingSettings(
        escalate_to="hard", descend_to="cheap",
        escalate_after_failures=1, descend_after_successes=1, max_switches=1,
    )
    state = SwitchState()
    target, state = decide_next_model(settings, state, current_model="auto", round_failed=True)
    assert target == "hard"
    # switches budget exhausted — no descend back
    target, state = decide_next_model(settings, state, current_model="hard", round_failed=False)
    assert target is None


# ---------------------------------------------------------------------------
# Loop integration: observe after a tool round, apply at the next iteration
# ---------------------------------------------------------------------------

def _fake_agent(**overrides):
    defaults = dict(
        model="auto", provider="deepseek", api_key="k",
        base_url="http://127.0.0.1:2100/v1", api_mode="chat_completions",
        _model_switching_settings=ModelSwitchingSettings(escalate_to="hard", escalate_after_failures=1),
        _model_switch_state=None, _adaptive_model_switch_pending=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _failing_round():
    return [
        {"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]},
        {"role": "tool", "name": "terminal", "tool_call_id": "1",
         "content": '{"exit_code": 1, "error": "boom"}'},
    ]


def _clean_round():
    return [
        {"role": "assistant", "tool_calls": [{"function": {"name": "terminal"}}]},
        {"role": "tool", "name": "terminal", "tool_call_id": "1",
         "content": '{"exit_code": 0}'},
    ]


def test_observe_schedules_escalation_after_failure():
    agent = _fake_agent()
    observe_tool_round(agent, _failing_round())
    assert agent._adaptive_model_switch_pending == "hard"


def test_observe_clean_round_does_not_schedule():
    agent = _fake_agent()
    observe_tool_round(agent, _clean_round())
    assert agent._adaptive_model_switch_pending is None


def test_observe_ignores_when_disabled():
    agent = _fake_agent(_model_switching_settings=ModelSwitchingSettings())
    observe_tool_round(agent, _failing_round())
    assert agent._adaptive_model_switch_pending is None
    assert agent._model_switch_state is None  # state never advanced


def test_turn_reset_restores_original_model_after_descent_disabled():
    agent = _fake_agent(
        model="hard",
        _adaptive_model_switch_original_model="auto",
        _model_switch_state=SwitchState(escalated=True, switches=1),
        _adaptive_model_switch_pending="expert",
    )
    reset_model_switching_turn(agent)
    assert agent.model == "auto"
    assert agent._adaptive_model_switch_original_model == "auto"
    assert agent._model_switch_state is None
    assert agent._adaptive_model_switch_pending is None


def test_apply_pending_model_switch_changes_only_request_model(monkeypatch):
    cached_prompt = "byte-stable system prompt"
    primary_runtime = {"model": "auto", "provider": "custom"}
    agent = _fake_agent(
        provider="custom",
        _adaptive_model_switch_pending="hard",
        _cached_system_prompt=cached_prompt,
        _primary_runtime=primary_runtime,
    )

    def forbidden_switch_model(*args, **kwargs):
        raise AssertionError("adaptive routing must not invoke the cache-breaking full switch")

    monkeypatch.setattr("agent.agent_runtime_helpers.switch_model", forbidden_switch_model)
    assert apply_pending_model_switch(agent) is True
    assert agent.model == "hard"
    assert agent._cached_system_prompt == cached_prompt
    assert agent._primary_runtime is primary_runtime
    assert agent._adaptive_model_switch_pending is None


def test_apply_pending_accepts_custom_chat_completions_alias():
    agent = _fake_agent(
        provider="openai-compatible",
        api_mode="chat_completions",
        _adaptive_model_switch_pending="hard",
    )
    assert apply_pending_model_switch(agent) is True
    assert agent.model == "hard"


def test_apply_pending_rejects_non_custom_transport(caplog):
    agent = _fake_agent(provider="deepseek", _adaptive_model_switch_pending="hard")
    with caplog.at_level(logging.WARNING, logger="agent.model_switching"):
        assert apply_pending_model_switch(agent) is False
    assert agent.model == "auto"
    assert agent._adaptive_model_switch_pending is None
    assert "custom endpoint" in caplog.text


def test_apply_pending_noop_without_pending():
    agent = _fake_agent(_adaptive_model_switch_pending=None)
    assert apply_pending_model_switch(agent) is False


