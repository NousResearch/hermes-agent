from types import SimpleNamespace

from agent.conversation_loop import (
    _CODEX_ZERO_EVENT_BREAKERS,
    _clear_codex_zero_event_breaker,
    _codex_zero_event_breaker_remaining,
    _codex_zero_event_breaker_threshold,
    _is_codex_zero_event_failfast_target,
    _record_codex_zero_event_failure,
)


def _make_agent(*, parent_session_id=None):
    return SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        _parent_session_id=parent_session_id,
    )


def setup_function():
    _CODEX_ZERO_EVENT_BREAKERS.clear()


def test_breaker_only_targets_openai_codex_gpt_5_5():
    good = _make_agent()
    bad = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.4",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        _parent_session_id=None,
    )
    assert _is_codex_zero_event_failfast_target(good) is True
    assert _is_codex_zero_event_failfast_target(bad) is False


def test_breaker_opens_after_two_failures_for_normal_session():
    agent = _make_agent()
    assert _codex_zero_event_breaker_threshold(agent) == 2
    assert _codex_zero_event_breaker_remaining(agent) == 0
    _record_codex_zero_event_failure(agent)
    assert _codex_zero_event_breaker_remaining(agent) == 0
    _record_codex_zero_event_failure(agent)
    assert _codex_zero_event_breaker_remaining(agent) > 0


def test_breaker_is_stricter_for_child_sessions():
    agent = _make_agent(parent_session_id="parent-123")
    assert _codex_zero_event_breaker_threshold(agent) == 1
    _record_codex_zero_event_failure(agent)
    assert _codex_zero_event_breaker_remaining(agent) > 0
    _clear_codex_zero_event_breaker(agent)
    assert _codex_zero_event_breaker_remaining(agent) == 0
