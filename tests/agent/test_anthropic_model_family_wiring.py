"""Live primary-request wiring for Anthropic opaque model family hints."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import build_api_kwargs


def _provider(base_url, model, family):
    return {
        "base_url": base_url,
        "models": {
            model: {"anthropic_model_family": family},
        },
    }


def _agent(custom_providers):
    transport = MagicMock()
    transport.build_kwargs.side_effect = lambda **kwargs: kwargs
    agent = SimpleNamespace(
        api_mode="anthropic_messages",
        tools=[],
        model="ep-primary",
        base_url="https://primary.example/anthropic",
        _custom_providers=custom_providers,
        max_tokens=4096,
        reasoning_config={"enabled": True, "effort": "high"},
        _is_anthropic_oauth=False,
        _oauth_1m_beta_disabled=False,
        request_overrides={},
        context_compressor=None,
        _get_transport=lambda: transport,
        _prepare_anthropic_messages_for_api=lambda messages: messages,
        _anthropic_preserve_dots=lambda: False,
    )
    return agent, transport


def test_primary_request_forwards_current_family():
    providers = [
        _provider(
            "https://primary.example/anthropic",
            "ep-primary",
            "claude-opus-4-6",
        )
    ]
    agent, transport = _agent(providers)

    result = build_api_kwargs(
        agent,
        [{"role": "user", "content": "hi"}],
    )

    assert result["anthropic_model_family"] == "claude-opus-4-6"
    assert transport.build_kwargs.call_count == 1


def test_primary_request_re_resolves_after_runtime_route_change():
    providers = [
        _provider(
            "https://primary.example/anthropic",
            "ep-primary",
            "claude-opus-4-6",
        ),
        _provider(
            "https://switched.example/anthropic",
            "ep-switched",
            "claude-fable-5",
        ),
    ]
    agent, transport = _agent(providers)

    build_api_kwargs(agent, [{"role": "user", "content": "first"}])
    agent.model = "ep-switched"
    agent.base_url = "https://switched.example/anthropic"
    build_api_kwargs(agent, [{"role": "user", "content": "second"}])

    families = [
        call.kwargs["anthropic_model_family"]
        for call in transport.build_kwargs.call_args_list
    ]
    assert families == ["claude-opus-4-6", "claude-fable-5"]


def test_missing_cached_provider_list_does_not_read_config_on_request():
    agent, transport = _agent([])
    del agent._custom_providers

    with patch("hermes_cli.config.get_compatible_custom_providers") as loader:
        build_api_kwargs(agent, [{"role": "user", "content": "hi"}])

    loader.assert_not_called()
    assert transport.build_kwargs.call_args.kwargs[
        "anthropic_model_family"
    ] is None
