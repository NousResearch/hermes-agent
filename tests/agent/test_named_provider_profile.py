"""Regression coverage for bare configured custom provider profiles (#119681)."""

from __future__ import annotations

import yaml

from agent.chat_completion_helpers import _build_chat_completions_kwargs
from agent.transports.chat_completions import ChatCompletionsTransport
from providers.base import ProviderProfile


class _Agent:
    """Small chat-completions caller using a bare configured provider name."""

    provider = "opencodex"
    model = "relay-model"
    base_url = "https://relay.example/v1"
    _base_url_lower = base_url
    _base_url_hostname = "relay.example"
    session_id = "test-session"
    max_tokens = None
    _ollama_num_ctx = None
    openrouter_min_coding_score = None
    providers_allowed = None
    providers_ignored = None
    providers_order = None
    provider_sort = None
    provider_require_parameters = None
    provider_data_collection = None
    provider_routing = None

    def _get_transport(self):
        return ChatCompletionsTransport()

    def _is_qwen_portal(self):
        return False

    def _is_openrouter_url(self):
        return False

    def _prepare_messages_for_non_vision_model(self, messages):
        return messages

    def _resolved_api_call_timeout(self):
        return None

    def _max_tokens_param(self, *_args, **_kwargs):
        return None

    def _supports_reasoning_extra_body(self):
        return False


def _configure_named_provider(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "opencodex", "default": "relay-model"},
        "providers": {"opencodex": {"base_url": "https://relay.example/v1"}},
    }))


def test_bare_named_provider_uses_custom_reasoning_profile(tmp_path, monkeypatch):
    _configure_named_provider(tmp_path, monkeypatch)

    kwargs = _build_chat_completions_kwargs(
        _Agent(), [{"role": "user", "content": "hi"}], [],
        {"enabled": True, "effort": "high"}, None, None,
    )

    assert kwargs["reasoning_effort"] == "high"
    assert "reasoning" not in kwargs.get("extra_body", {})


def test_bare_named_provider_preserves_disabled_reasoning(tmp_path, monkeypatch):
    _configure_named_provider(tmp_path, monkeypatch)

    kwargs = _build_chat_completions_kwargs(
        _Agent(), [{"role": "user", "content": "hi"}], [],
        {"enabled": False}, None, None,
    )

    assert kwargs["reasoning_effort"] == "none"


def test_bare_named_provider_keeps_an_explicitly_registered_profile(tmp_path, monkeypatch):
    _configure_named_provider(tmp_path, monkeypatch)
    import providers

    # Keep this registration local: configured bare names must fall back only
    # when they have no dedicated profile.
    providers.get_provider_profile("custom")
    monkeypatch.setattr(providers, "_REGISTRY", dict(providers._REGISTRY))
    monkeypatch.setattr(providers, "_ALIASES", dict(providers._ALIASES))

    class DedicatedProfile(ProviderProfile):
        def build_api_kwargs_extras(self, **_kwargs):
            return {}, {"registered_profile": True}

    providers.register_provider(DedicatedProfile(name="opencodex"))
    kwargs = _build_chat_completions_kwargs(
        _Agent(), [{"role": "user", "content": "hi"}], [],
        {"enabled": True, "effort": "high"}, None, None,
    )

    assert kwargs["registered_profile"] is True
    assert "reasoning_effort" not in kwargs
