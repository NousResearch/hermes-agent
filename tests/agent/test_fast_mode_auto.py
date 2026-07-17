from types import SimpleNamespace

import pytest

from agent.fast_mode import (
    DEFAULT_FAST_AUTO_ON_SECONDS,
    begin_fast_mode_turn,
    effective_request_overrides,
    invalidate_fast_mode_turn,
    normalize_fast_auto_on_seconds,
    revalidate_fast_mode_request,
)


def _agent(model="gpt-5.4", **overrides):
    values = {
        "model": model,
        "service_tier": "auto",
        "fast_auto_on_seconds": 60,
        "request_overrides": {"unrelated": "preserved"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value", [None, "", 0, -1, float("inf"), True])
def test_invalid_auto_cutoff_uses_default(value):
    assert normalize_fast_auto_on_seconds(value) == DEFAULT_FAST_AUTO_ON_SECONDS


def test_auto_fast_is_active_through_cutoff_then_removed():
    agent = _agent(request_overrides={"service_tier": "priority", "unrelated": 1})
    begin_fast_mode_turn(agent, now=100.0)

    assert effective_request_overrides(agent, now=160.0) == {
        "service_tier": "priority",
        "unrelated": 1,
    }
    assert effective_request_overrides(agent, now=160.001) == {"unrelated": 1}


def test_auto_fast_resets_for_each_user_turn():
    agent = _agent()
    begin_fast_mode_turn(agent, now=100.0)
    assert "service_tier" not in effective_request_overrides(agent, now=161.0)

    begin_fast_mode_turn(agent, now=200.0)
    assert effective_request_overrides(agent, now=200.0)["service_tier"] == "priority"


def test_cold_fast_only_opens_on_first_logical_session_turn():
    agent = _agent(service_tier="cold", _session_messages=[])

    begin_fast_mode_turn(agent, [], now=100.0)
    assert effective_request_overrides(agent, now=100.0)["service_tier"] == "priority"

    agent._session_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]
    begin_fast_mode_turn(agent, None, now=200.0)
    assert effective_request_overrides(agent, now=200.0) == {
        "unrelated": "preserved"
    }


def test_cold_fast_explicit_empty_history_starts_new_logical_session():
    agent = _agent(
        service_tier="cold",
        _session_messages=[
            {"role": "user", "content": "previous session"},
            {"role": "assistant", "content": "previous reply"},
        ],
    )

    begin_fast_mode_turn(agent, [], now=100.0)

    assert effective_request_overrides(agent, now=100.0)["service_tier"] == "priority"


def test_cold_fast_stays_off_when_fresh_process_resumes_persisted_history():
    agent = _agent(service_tier="cold", _session_messages=[])
    persisted_history = [
        {"role": "user", "content": "before restart"},
        {"role": "assistant", "content": "persisted reply"},
    ]

    begin_fast_mode_turn(agent, persisted_history, now=100.0)

    assert agent._fast_mode_turn_started_at is None
    assert effective_request_overrides(agent, now=100.0) == {
        "unrelated": "preserved"
    }


def test_cold_fast_treats_system_only_history_as_first_turn():
    agent = _agent(service_tier="cold", _session_messages=[])

    begin_fast_mode_turn(
        agent, [{"role": "system", "content": "session setup"}], now=100.0
    )

    assert effective_request_overrides(agent, now=160.0)["service_tier"] == "priority"
    assert effective_request_overrides(agent, now=160.001) == {
        "unrelated": "preserved"
    }


@pytest.mark.parametrize("role", ["assistant", "tool"])
def test_cold_fast_conservatively_rejects_partial_prior_transcripts(role):
    agent = _agent(service_tier="cold", _session_messages=[])

    begin_fast_mode_turn(agent, [{"role": role, "content": "prior"}], now=100.0)

    assert effective_request_overrides(agent, now=100.0) == {
        "unrelated": "preserved"
    }


def test_cold_fast_does_not_lazy_start_without_a_turn_boundary():
    agent = _agent(service_tier="cold")

    assert effective_request_overrides(agent, now=100.0) == {
        "unrelated": "preserved"
    }


def test_live_dynamic_mode_change_cannot_reuse_prior_turn_clock():
    agent = _agent(service_tier="auto", _session_messages=[])
    begin_fast_mode_turn(agent, [], now=100.0)

    agent.service_tier = "cold"

    assert effective_request_overrides(agent, now=110.0) == {
        "unrelated": "preserved"
    }


def test_invalidated_dynamic_mode_waits_for_next_turn_boundary():
    agent = _agent(service_tier="auto", _session_messages=[])
    begin_fast_mode_turn(agent, [], now=100.0)
    invalidate_fast_mode_turn(agent)

    assert effective_request_overrides(agent, now=110.0) == {
        "unrelated": "preserved"
    }


def test_dispatch_revalidation_strips_openai_fast_after_cutoff(monkeypatch):
    agent = _agent(
        service_tier="auto",
        api_mode="chat_completions",
        _session_messages=[],
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    built_kwargs = {"model": "gpt-5.4", "service_tier": "priority"}
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)

    dispatched = revalidate_fast_mode_request(agent, built_kwargs)

    assert "service_tier" not in dispatched


def test_dispatch_revalidation_preserves_middleware_override(monkeypatch):
    agent = _agent(
        service_tier="auto",
        api_mode="chat_completions",
        request_overrides={"timeout": 999, "service_tier": "priority"},
        _session_messages=[],
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    middleware_kwargs = {
        "model": "gpt-5.4",
        "timeout": 5,
        "service_tier": "priority",
    }
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 110.0)

    dispatched = revalidate_fast_mode_request(agent, middleware_kwargs)

    assert dispatched["timeout"] == 5
    assert dispatched["service_tier"] == "priority"


def test_dispatch_revalidation_strips_anthropic_fast_metadata_after_cutoff(
    monkeypatch,
):
    agent = _agent(
        model="anthropic/claude-opus-4.6",
        service_tier="auto",
        api_mode="anthropic_messages",
        _anthropic_base_url="https://api.anthropic.com",
        _is_anthropic_oauth=False,
        _oauth_1m_beta_disabled=False,
        _session_messages=[],
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    built_kwargs = {
        "model": "claude-opus-4-6",
        "extra_body": {"speed": "fast", "unrelated": 1},
        "extra_headers": {
            "anthropic-beta": "interleaved-thinking-2025-05-14,fast-mode-2026-02-01",
            "x-extra": "preserved",
        },
    }
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)

    dispatched = revalidate_fast_mode_request(agent, built_kwargs)

    assert dispatched["extra_body"] == {"unrelated": 1}
    assert dispatched["extra_headers"] == {
        "anthropic-beta": "interleaved-thinking-2025-05-14",
        "x-extra": "preserved",
    }


def test_dispatch_revalidation_preserves_middleware_anthropic_beta(monkeypatch):
    agent = _agent(
        model="anthropic/claude-opus-4.6",
        service_tier="auto",
        api_mode="anthropic_messages",
        _anthropic_base_url="https://api.anthropic.com",
        _is_anthropic_oauth=False,
        _oauth_1m_beta_disabled=False,
        _session_messages=[],
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    middleware_kwargs = {
        "model": "claude-opus-4-6",
        "extra_headers": {
            "anthropic-beta": "plugin-feature-2099-01-01",
            "x-plugin": "preserved",
        },
    }
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 110.0)

    dispatched = revalidate_fast_mode_request(agent, middleware_kwargs)

    betas = dispatched["extra_headers"]["anthropic-beta"].split(",")
    assert "plugin-feature-2099-01-01" in betas
    assert "fast-mode-2026-02-01" in betas
    assert dispatched["extra_headers"]["x-plugin"] == "preserved"


def test_auto_fast_uses_anthropic_speed_override():
    agent = _agent(model="anthropic/claude-opus-4.6")
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=20.0) == {
        "speed": "fast",
        "unrelated": "preserved",
    }
    assert effective_request_overrides(agent, now=71.0) == {"unrelated": "preserved"}


def test_explicit_fast_mode_is_unchanged():
    agent = _agent(
        service_tier="priority",
        request_overrides={"service_tier": "priority", "unrelated": 1},
    )
    begin_fast_mode_turn(agent, now=100.0)

    assert agent._fast_mode_turn_started_at is None
    assert effective_request_overrides(agent, now=1000.0) == {
        "service_tier": "priority",
        "unrelated": 1,
    }


def test_unsupported_model_never_adds_a_fast_override():
    agent = _agent(model="openrouter/some-unsupported-model")
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=10.0) == {"unrelated": "preserved"}
