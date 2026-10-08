"""Unregistered providers on a local endpoint reuse the custom wire profile."""

from __future__ import annotations

from types import SimpleNamespace

from agent.chat_completion_helpers import _build_chat_completions_kwargs


def _agent(provider: str, base_url: str):
    captured = {}

    class Transport:
        def build_kwargs(self, **kwargs):
            captured.update(kwargs)
            return kwargs

    agent = SimpleNamespace(
        provider=provider,
        base_url=base_url,
        model="local-model",
        _base_url_lower=base_url.lower(),
        _base_url_hostname="127.0.0.1",
        session_id="s",
        max_tokens=None,
        _ollama_num_ctx=None,
        openrouter_min_coding_score=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=None,
        provider_data_collection=None,
        _get_transport=lambda: Transport(),
        _is_qwen_portal=lambda: False,
        _is_openrouter_url=lambda: False,
        _prepare_messages_for_non_vision_model=lambda msgs: msgs,
        _resolved_api_call_timeout=lambda: 30,
        _max_tokens_param=lambda *args, **kwargs: {},
        _supports_reasoning_extra_body=lambda: False,
        _captured=captured,
    )
    return agent


def test_local_unregistered_provider_uses_custom_profile():
    agent = _agent("omlx-profile", "http://127.0.0.1:8095/v1")
    _build_chat_completions_kwargs(
        agent, [{"role": "user", "content": "hi"}], None, None, {}, None,
    )
    assert agent._captured["provider_profile"].name == "custom"


def test_remote_unregistered_provider_stays_on_the_legacy_path():
    agent = _agent("omlx-profile", "https://example.com/v1")
    _build_chat_completions_kwargs(
        agent, [{"role": "user", "content": "hi"}], None, None, {}, None,
    )
    assert "provider_profile" not in agent._captured


def test_registered_provider_is_not_replaced_on_a_local_endpoint():
    agent = _agent("openrouter", "http://127.0.0.1:11434/v1")
    _build_chat_completions_kwargs(
        agent, [{"role": "user", "content": "hi"}], None, None, {}, None,
    )
    assert agent._captured["provider_profile"].name == "openrouter"
