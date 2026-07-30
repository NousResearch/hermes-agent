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
        "provider": "openai-api",
        "api_mode": "chat_completions",
        "base_url": "https://api.openai.com/v1",
        "service_tier": "auto",
        "fast_auto_on_seconds": 60,
        "request_overrides": {"unrelated": "preserved"},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("value", [None, "", 0, -1, float("inf"), True])
def test_invalid_auto_cutoff_uses_default(value):
    assert normalize_fast_auto_on_seconds(value) == DEFAULT_FAST_AUTO_ON_SECONDS


def test_auto_fast_is_inclusive_at_cutoff_then_expires():
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


@pytest.mark.parametrize(
    ("history", "eligible"),
    [
        ([], True),
        ([{"role": "system", "content": "setup"}], True),
        ([{"role": "user", "content": "prior"}], False),
        ([{"role": "assistant", "content": "partial"}], False),
        ([{"role": "tool", "content": "partial"}], False),
    ],
)
def test_cold_eligibility_uses_persisted_logical_transcript(history, eligible):
    agent = _agent(service_tier="cold")
    begin_fast_mode_turn(agent, history, now=100.0)
    effective = effective_request_overrides(agent, now=100.0)
    assert ("service_tier" in effective) is eligible


def test_live_mode_change_or_invalidation_cannot_reuse_turn_clock():
    agent = _agent()
    begin_fast_mode_turn(agent, [], now=100.0)
    agent.service_tier = "cold"
    assert effective_request_overrides(agent, now=110.0) == {"unrelated": "preserved"}
    agent.service_tier = "auto"
    invalidate_fast_mode_turn(agent)
    assert effective_request_overrides(agent, now=110.0) == {"unrelated": "preserved"}


def test_normal_and_fast_modes_remain_backwards_compatible():
    normal = _agent(service_tier=None, request_overrides={"timeout": 9})
    fast = _agent(
        service_tier="priority",
        request_overrides={"service_tier": "priority", "timeout": 9},
    )
    begin_fast_mode_turn(normal, now=100.0)
    begin_fast_mode_turn(fast, now=100.0)
    assert effective_request_overrides(normal, now=1000.0) == {"timeout": 9}
    assert effective_request_overrides(fast, now=1000.0) == {
        "service_tier": "priority",
        "timeout": 9,
    }


def test_unsupported_model_never_adds_fast_metadata():
    agent = _agent(model="openrouter/some-unsupported-model")
    begin_fast_mode_turn(agent, now=10.0)
    assert effective_request_overrides(agent, now=10.0) == {"unrelated": "preserved"}


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("custom", "https://proxy.example/v1"),
        ("openai-api", "https://api.openai.com.attacker.test/v1"),
        ("openai-api", "https://proxy.example/api.openai.com/v1"),
    ],
)
def test_dynamic_openai_priority_requires_native_runtime_identity(
    provider, base_url
):
    agent = _agent(provider=provider, base_url=base_url)
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=10.0) == {
        "unrelated": "preserved"
    }


def test_dynamic_openai_priority_remains_enabled_on_native_endpoint():
    agent = _agent()
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=10.0) == {
        "service_tier": "priority",
        "unrelated": "preserved",
    }


def test_dynamic_anthropic_metadata_is_not_emitted_for_bedrock():
    agent = _agent(
        model="anthropic/claude-opus-4.6",
        provider="bedrock",
        api_mode="bedrock_converse",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    begin_fast_mode_turn(agent, now=10.0)

    assert effective_request_overrides(agent, now=10.0) == {
        "unrelated": "preserved"
    }


def test_dispatch_revalidation_preserves_middleware_fields(monkeypatch):
    agent = _agent(api_mode="chat_completions")
    begin_fast_mode_turn(agent, [], now=100.0)
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)
    dispatched = revalidate_fast_mode_request(
        agent,
        {"model": "gpt-5.4", "service_tier": "priority", "timeout": 5},
    )
    assert dispatched == {"model": "gpt-5.4", "timeout": 5}
    assert agent.request_overrides == {"unrelated": "preserved"}


def test_nonstreaming_dispatch_revalidates_at_transport_boundary(monkeypatch):
    from agent.chat_completion_helpers import _dispatch_nonstreaming_api_request

    captured = {}
    agent = _agent(
        api_mode="chat_completions",
        provider="openai",
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)

    class _Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return "response"

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    result = _dispatch_nonstreaming_api_request(
        agent,
        {"model": "gpt-5.4", "service_tier": "priority", "timeout": 5},
        make_client=lambda *_args, **_kwargs: client,
    )
    assert result == "response"
    assert captured == {"model": "gpt-5.4", "timeout": 5}


def test_streaming_entry_revalidates_before_direct_dispatch(monkeypatch):
    from agent.chat_completion_helpers import interruptible_streaming_api_call

    captured = {}
    agent = _agent(
        api_mode="chat_completions",
        provider="openai",
        platform="cron",
        _interrupt_requested=False,
    )
    agent._interruptible_api_call = lambda kwargs: captured.update(kwargs) or "response"
    begin_fast_mode_turn(agent, [], now=100.0)
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)

    result = interruptible_streaming_api_call(
        agent,
        {"model": "gpt-5.4", "service_tier": "priority", "timeout": 5},
    )
    assert result == "response"
    assert captured == {"model": "gpt-5.4", "timeout": 5}


def test_dispatch_revalidation_preserves_anthropic_metadata(monkeypatch):
    agent = _agent(
        model="anthropic/claude-opus-4.6",
        api_mode="anthropic_messages",
        _anthropic_base_url="https://api.anthropic.com",
        _is_anthropic_oauth=False,
        _oauth_1m_beta_disabled=False,
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    built = {
        "model": "claude-opus-4-6",
        "extra_body": {"speed": "fast", "unrelated": 1},
        "extra_headers": {
            "anthropic-beta": "plugin-feature,fast-mode-2026-02-01",
            "x-plugin": "preserved",
        },
    }
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)
    dispatched = revalidate_fast_mode_request(agent, built)
    assert dispatched["extra_body"] == {"unrelated": 1}
    assert dispatched["extra_headers"] == {
        "anthropic-beta": "plugin-feature",
        "x-plugin": "preserved",
    }


def test_dispatch_revalidation_strips_fast_metadata_from_bedrock(monkeypatch):
    agent = _agent(
        model="anthropic/claude-opus-4.6",
        provider="bedrock",
        api_mode="bedrock_converse",
        base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )
    begin_fast_mode_turn(agent, [], now=100.0)
    monkeypatch.setattr("agent.fast_mode.time.monotonic", lambda: 160.001)

    dispatched = revalidate_fast_mode_request(
        agent,
        {
            "modelId": "anthropic.claude-opus-4-6",
            "speed": "fast",
            "service_tier": "priority",
            "inferenceConfig": {"maxTokens": 100},
        },
    )

    assert dispatched == {
        "modelId": "anthropic.claude-opus-4-6",
        "inferenceConfig": {"maxTokens": 100},
    }


def test_policy_never_mutates_prompt_or_message_roles():
    agent = _agent()
    messages = [
        {"role": "system", "content": "stable prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "working"},
        {"role": "tool", "content": "result"},
    ]
    before = [dict(message) for message in messages]
    begin_fast_mode_turn(agent, [], now=100.0)
    effective_request_overrides(agent, now=110.0)
    revalidate_fast_mode_request(
        agent, {"messages": messages, "service_tier": "priority"}
    )
    assert messages == before
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
