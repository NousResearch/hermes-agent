"""Unit tests for the CommandCode model provider plugin."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def commandcode_module():
    import model_tools  # noqa: F401  # triggers bundled provider discovery

    return importlib.import_module("plugins.model-providers.commandcode.provider")


@pytest.fixture
def commandcode_profile(commandcode_module):
    import providers

    profile = providers.get_provider_profile("commandcode")
    assert profile is not None, "commandcode provider profile must be registered"
    return profile


@pytest.fixture
def commandcode_anthropic_profile(commandcode_module):
    import providers

    profile = providers.get_provider_profile("commandcode-anthropic")
    assert profile is not None, "commandcode-anthropic profile must be registered"
    return profile


class TestCommandCodeProfile:
    def test_profile_metadata(self, commandcode_profile):
        assert commandcode_profile.name == "commandcode"
        assert commandcode_profile.base_url == "https://api.commandcode.ai/provider/v1"
        assert commandcode_profile.models_url == "https://api.commandcode.ai/provider/v1/models"
        assert commandcode_profile.default_aux_model == "Qwen/Qwen3.6-Plus"

    def test_context_length_overrides_cover_catalog_snapshot(self, commandcode_profile):
        overrides = commandcode_profile.context_length_overrides
        assert len(overrides) >= 20
        assert overrides["deepseek/deepseek-v4-pro"] == 1_000_000
        assert overrides["deepseek/deepseek-v4-flash"] == 1_000_000
        assert overrides["Qwen/Qwen3.6-Plus"] == 200_000

    @pytest.mark.parametrize(
        ("requested", "resolved", "context_length"),
        [
            ("deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro", 1_000_000),
            ("deepseek-v4-pro", "deepseek/deepseek-v4-pro", 1_000_000),
            ("commandcode:deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro", 1_000_000),
            ("Qwen3.6-Plus", "Qwen/Qwen3.6-Plus", 200_000),
        ],
    )
    def test_known_models_resolve_from_context_overrides(
        self,
        commandcode_profile,
        requested,
        resolved,
        context_length,
    ):
        assert commandcode_profile.resolve_model_id(requested) == resolved
        assert commandcode_profile.get_context_length(requested) == context_length

    def test_reasoning_config_passes_through_like_openai_compat(self, commandcode_profile):
        extra_body, top_level = commandcode_profile.build_api_kwargs_extras(
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert extra_body == {"reasoning": {"enabled": True, "effort": "high"}}
        assert top_level == {}

    def test_fetch_models_updates_live_context_length_cache(self, commandcode_profile, commandcode_module, monkeypatch):
        monkeypatch.setattr(commandcode_module, "_MODEL_CACHE", None)
        monkeypatch.setattr(commandcode_module, "_LIVE_CONTEXT_LENGTH_OVERRIDES", {})

        payload = {
            "data": [
                {
                    "id": "deepseek/deepseek-v4-pro",
                    "context_length": 1_000_000,
                },
                {
                    "id": "org/new-model",
                    "context_length": 262_144,
                },
            ]
        }

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload).encode()

        def fake_urlopen(request, timeout):
            assert request.full_url == commandcode_profile.models_url
            assert timeout == 8.0
            return _FakeResponse()

        monkeypatch.setattr(commandcode_module.urllib.request, "urlopen", fake_urlopen)

        assert commandcode_profile.fetch_models() == [
            "deepseek/deepseek-v4-pro",
            "org/new-model",
        ]
        assert commandcode_profile.get_context_length("org/new-model") == 262_144


class TestCommandCodeAnthropicProfile:
    def test_profile_metadata(self, commandcode_anthropic_profile):
        assert commandcode_anthropic_profile.name == "commandcode-anthropic"
        assert commandcode_anthropic_profile.api_mode == "anthropic_messages"
        assert commandcode_anthropic_profile.base_url == "https://api.commandcode.ai/provider"
        assert commandcode_anthropic_profile.models_url == "https://api.commandcode.ai/provider/v1/models"

    def test_shim_adapts_messages_api_to_chat_completions(self):
        anthropic_shim = importlib.import_module(
            "plugins.model-providers.commandcode.anthropic_shim"
        )
        CommandCodeAnthropicShim = anthropic_shim.CommandCodeAnthropicShim

        observed: dict[str, Any] = {}

        class _FakeCompletions:
            def create(self, **kwargs):
                observed.update(kwargs)
                return SimpleNamespace(
                    id="chatcmpl_123",
                    model=kwargs["model"],
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="pong", tool_calls=[]),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions())
        )
        shim = CommandCodeAnthropicShim(fake_client, default_model="deepseek/deepseek-v4-pro")

        response = shim.messages.create(
            system="Be terse.",
            messages=[{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
            tools=[
                {
                    "name": "lookup",
                    "description": "Look something up",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="auto",
            max_tokens=128,
        )

        assert observed["model"] == "deepseek/deepseek-v4-pro"
        assert observed["messages"][0] == {"role": "system", "content": "Be terse."}
        assert observed["messages"][1] == {"role": "user", "content": "ping"}
        assert observed["tools"][0]["function"]["name"] == "lookup"
        assert response.stop_reason == "end_turn"
        assert response.content[0].type == "text"
        assert response.content[0].text == "pong"
