from types import SimpleNamespace

from agent.chat_completion_helpers import build_api_kwargs
from hermes_cli.config import _normalize_custom_provider_entry


class _RecordingTransport:
    def build_kwargs(self, **kwargs):
        return kwargs


def test_custom_provider_cache_capability_reaches_chat_transport():
    provider = _normalize_custom_provider_entry({
        "name": "cliproxy",
        "base_url": "https://cliproxy.example/v1",
        "supports_prompt_cache_key": True,
    })
    assert provider is not None

    agent = SimpleNamespace(
        api_mode="chat_completions",
        tools=[],
        model="deepseek-v4-flash",
        base_url=provider["base_url"],
        provider="custom:cliproxy",
        _base_url_lower=provider["base_url"].lower(),
        _custom_providers=[provider],
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        session_id="stable-session",
        max_tokens=None,
        reasoning_config=None,
        request_overrides={},
        _ollama_num_ctx=None,
        _max_tokens_param=lambda value: {"max_tokens": value},
        openrouter_min_coding_score=None,
        _get_transport=lambda: _RecordingTransport(),
        _is_qwen_portal=lambda: False,
        _is_openrouter_url=lambda: False,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 30,
        _supports_reasoning_extra_body=lambda: False,
        _github_models_reasoning_extra_body=lambda: None,
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}])

    assert kwargs["supports_prompt_cache_key"] is True


def test_bare_custom_provider_cache_capability_reaches_profile_path():
    """Bare provider='custom' takes the CustomProfile path, not the legacy
    path.  The config-level supports_prompt_cache_key must still propagate
    through _build_kwargs_from_profile via the params override."""
    provider = _normalize_custom_provider_entry({
        "name": "cliproxy",
        "base_url": "https://cliproxy.example/v1",
        "supports_prompt_cache_key": True,
    })
    assert provider is not None

    agent = SimpleNamespace(
        api_mode="chat_completions",
        tools=[],
        model="deepseek-v4-flash",
        base_url=provider["base_url"],
        provider="custom",
        _base_url_lower=provider["base_url"].lower(),
        _custom_providers=[provider],
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        session_id="stable-session",
        max_tokens=None,
        reasoning_config=None,
        request_overrides={},
        _ollama_num_ctx=None,
        _max_tokens_param=lambda value: {"max_tokens": value},
        openrouter_min_coding_score=None,
        _get_transport=lambda: _RecordingTransport(),
        _is_qwen_portal=lambda: False,
        _is_openrouter_url=lambda: False,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 30,
        _supports_reasoning_extra_body=lambda: False,
        _github_models_reasoning_extra_body=lambda: None,
    )

    kwargs = build_api_kwargs(agent, [{"role": "user", "content": "hello"}])

    assert kwargs["supports_prompt_cache_key"] is True
