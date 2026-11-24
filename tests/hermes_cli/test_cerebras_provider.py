"""Cerebras provider setup and request compatibility."""

from io import BytesIO

import pytest

from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_cli.models import provider_model_ids
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli import urllib_security
from providers import get_provider_profile


def test_cerebras_resolves_profile_credentials_and_picker_models(monkeypatch):
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-test-key")

    def open_catalog(request, *, timeout):
        assert request.full_url == "https://api.cerebras.ai/v1/models"
        assert request.get_header("Authorization") == "Bearer cerebras-test-key"
        return BytesIO(b'{"data":[{"id":"gpt-oss-120b"},{"id":"dedicated-endpoint"}]}')

    monkeypatch.setattr(urllib_security, "open_credentialed_url", open_catalog)

    runtime = resolve_runtime_provider(requested="cerebras")
    assert runtime["provider"] == "cerebras"
    assert runtime["api_mode"] == "chat_completions"
    assert runtime["base_url"] == "https://api.cerebras.ai/v1"
    assert runtime["api_key"] == "cerebras-test-key"
    assert set(provider_model_ids("cerebras")) == {"qwen-3.8-27b", "gpt-oss-120b", "dedicated-endpoint"}


@pytest.mark.parametrize(
    ("model", "reasoning", "effort"),
    [
        ("qwen-3.8-27b", {"enabled": False}, "none"),
        ("gpt-oss-120b", {"enabled": False}, None),
        ("gpt-oss-120b", {"enabled": True, "effort": "xhigh"}, "high"),
        ("dedicated-endpoint", {"enabled": True, "effort": "high"}, None),
    ],
)
def test_cerebras_sends_only_supported_reasoning_efforts(model, reasoning, effort):
    profile = get_provider_profile("cerebras")
    kwargs = ChatCompletionsTransport().build_kwargs(
        model,
        [{"role": "user", "content": "Hello"}],
        [],
        provider_name="cerebras",
        base_url=profile.base_url,
        provider_profile=profile,
        reasoning_config=reasoning,
        supports_reasoning=True,
    )

    assert kwargs.get("reasoning_effort") == effort
    assert "reasoning" not in kwargs.get("extra_body", {})
