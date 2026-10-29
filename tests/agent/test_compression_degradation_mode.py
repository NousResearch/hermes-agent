"""Compression degradation mode: pinned fallback model + degraded-model guard.

A compression-route outage must cost one bounded attempt + one fallback try, then degrade
quietly on the cooldown ladder — never a no-cooldown immediate retry into the main model
every event. The fallback target order is:
``auxiliary.compression.fallback_model`` (pinned, aux-policy preserved) → main model —
and NEVER a main model listed on ``agent.compression_degraded_models``.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _compression_degraded_models,
    _compression_fallback_model,
)


def _compressor(model="main-model", summary_model="quota-exhausted-aux"):
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model=model, provider="main-provider", quiet_mode=True,
            base_url="https://main.example/v1", api_key="sk-MAIN-SECRET",
            summary_model_override=summary_model,
        )


def _access_error():
    error = Exception("Error code: 403 - model is not available in the current token plan")
    error.status_code = 403
    return error


def _ok_response(text="summary body"):
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = text
    return ok


def _no_config(monkeypatch):
    """No fallback pin, no degraded list: the default route."""
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    monkeypatch.setattr("agent.auxiliary_task_config._get_auxiliary_task_config", lambda task, _seen=frozenset(): {})


def _pin(monkeypatch, fallback_model="", degraded=None):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"compression_degraded_models": degraded}})
    monkeypatch.setattr(
        "agent.auxiliary_task_config._get_auxiliary_task_config",
        lambda task, _seen=frozenset(): {"fallback_model": fallback_model} if task == "compression" else {},
    )


TURNS = [
    {"role": "user", "content": "do something"},
    {"role": "assistant", "content": "ok"},
]


def test_default_fallback_still_reaches_main_but_arms_cooldown(monkeypatch):
    _no_config(monkeypatch)
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _ok_response()]) as mock_call:
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "summary body" in summary
    first, second = mock_call.call_args_list
    assert first.kwargs.get("model") == "quota-exhausted-aux"
    assert (second.kwargs.get("model"), second.kwargs.get("provider")) == ("main-model", "main-provider")
    # The success cleared the arm — healthy fallbacks are transparent.
    assert compressor._summary_failure_cooldown_until == 0.0


def test_default_fallback_arms_ladder_when_main_retry_also_fails(monkeypatch):
    _no_config(monkeypatch)
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _access_error()]):
        summary = compressor._generate_summary(TURNS)

    assert summary is None
    # The un-cooldown-forever behavior is the failure shape this guards: the arm must hold
    # the next events off.
    assert compressor._summary_failure_cooldown_until > 0.0
    assert compressor._last_summary_error


def test_pinned_fallback_model_wins_over_main(monkeypatch):
    _pin(monkeypatch, fallback_model="pinned-helper-model")
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _ok_response("summary via pinned fb")]) as mock_call:
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "summary via pinned fb" in summary
    first, second = mock_call.call_args_list
    assert first.kwargs.get("model") == "quota-exhausted-aux"
    # The retry rides the compression task route with the pinned model — NOT the main runtime.
    assert second.kwargs.get("model") == "pinned-helper-model"
    assert second.kwargs.get("provider") != "main-provider"
    assert compressor.summary_model == "pinned-helper-model"
    # The fb success also cleared the fallback arm.
    assert compressor._summary_failure_cooldown_until == 0.0


def test_degraded_guard_never_falls_back_to_barred_main(monkeypatch):
    _pin(monkeypatch, degraded=["main-model"])
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error()]) as mock_call:
        summary = compressor._generate_summary(TURNS)

    # Degraded, and critically: NO second call — the aux failure is the only attempt.
    assert summary is None
    assert len(mock_call.call_args_list) == 1
    assert compressor._summary_failure_cooldown_until > 0.0
    assert "degraded guard" in (compressor._last_summary_error or "")


def test_pinned_fallback_shields_even_a_barred_main(monkeypatch):
    _pin(monkeypatch, fallback_model="pinned-helper-model", degraded=["main-model"])
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _ok_response("fb summary")]) as mock_call:
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "fb summary" in summary
    assert mock_call.call_args_list[1].kwargs.get("model") == "pinned-helper-model"


def test_fallen_back_pin_is_not_retried_in_a_loop(monkeypatch):
    _pin(monkeypatch, fallback_model="pinned-helper-model")
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _access_error(), _access_error()]):
        summary = compressor._generate_summary(TURNS)

    # Aux fail → pinned-fb fail → generic transient arm: exactly two attempts, no ping-pong.
    assert summary is None
    assert compressor._summary_model_fallen_back is True
    assert compressor._summary_failure_cooldown_until > 0.0


def test_degraded_pin_is_ignored_and_main_used_when_healthy(monkeypatch):
    _pin(monkeypatch, fallback_model="known-bad-model", degraded=["known-bad-model"])
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=[_access_error(), _ok_response("main summary")]) as mock_call:
        summary = compressor._generate_summary(TURNS)

    assert summary is not None and "main summary" in summary
    assert (mock_call.call_args_list[1].kwargs.get("model"), mock_call.call_args_list[1].kwargs.get("provider")) == (
        "main-model", "main-provider",
    )


def test_config_helpers_parse_the_documented_keys(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"compression_degraded_models": ["known-bad-model", "  ", 5]}},
    )
    assert _compression_degraded_models() == ["known-bad-model", "5"]

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"compression_degraded_models": "known-bad-model"}},
    )
    assert _compression_degraded_models() == ["known-bad-model"]

    monkeypatch.setattr("hermes_cli.config.load_config_readonly", dict)
    assert _compression_degraded_models() == []

    monkeypatch.setattr(
        "agent.auxiliary_task_config._get_auxiliary_task_config",
        lambda task, _seen=frozenset(): {"fallback_model": " pinned-helper-model "},
    )
    assert _compression_fallback_model() == "pinned-helper-model"
    monkeypatch.setattr(
        "agent.auxiliary_task_config._get_auxiliary_task_config",
        lambda task, _seen=frozenset(): {},
    )
    assert _compression_fallback_model() == ""


def test_arm_floor_holds_across_repeated_events(monkeypatch):
    """Repeated degraded events back off ≥300s each — never a flat-60s immediate-retry cadence."""
    import time as _time

    _pin(monkeypatch, degraded=["main-model"])
    remaining = []
    compressor = _compressor()
    with patch("agent.context_compressor.call_llm", side_effect=_access_error()):
        for _ in range(3):
            compressor._generate_summary(TURNS, bypass_cooldown=True)
            remaining.append(compressor._summary_failure_cooldown_until - _time.monotonic())
    # Record () takes the max, so the first arm (300s) must survive the later generic 60s arms.
    assert all(299.0 <= r <= 301.0 for r in remaining), remaining


def test_success_after_fallback_clears_the_arm(monkeypatch):
    _no_config(monkeypatch)
    compressor = _compressor()
    with patch("agent.context_compressor.call_llm", side_effect=[_access_error()]):
        compressor._generate_summary(TURNS)
    assert compressor._summary_failure_cooldown_until > 0.0
    # A fresh compressor turn (state is per-compressor-instance) that succeeds must clear.
    with patch("agent.context_compressor.call_llm", side_effect=[_ok_response()]):
        summary = compressor._generate_summary(TURNS, bypass_cooldown=True)
    assert summary is not None
    assert compressor._summary_failure_cooldown_until == 0.0


@pytest.mark.parametrize("scenario", ["no-config", "pin"])
def test_guard_tolerates_config_read_failures(monkeypatch, scenario):
    def _boom():
        raise RuntimeError("config unavailable")

    if scenario == "no-config":
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _boom)
        monkeypatch.setattr("agent.auxiliary_task_config._get_auxiliary_task_config", _boom)
    compressor = _compressor()
    fallback = compressor._fallback_to_main_for_compression(_access_error(), "failed")
    # Fail-closed to the legacy behavior: fall back to main, never crash the compression lane.
    assert fallback == ""
    assert compressor.summary_model == ""
