"""Tests for native local Ollama provider behavior."""

from __future__ import annotations

import json
from unittest.mock import patch

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import (
    _PROVIDER_ALIASES,
    is_local_ollama_base_url,
    probe_ollama_native_models,
    provider_model_ids,
    validate_requested_model,
)


def test_ollama_registered_as_first_class_provider():
    pconfig = PROVIDER_REGISTRY["ollama"]
    assert pconfig.id == "ollama"
    assert pconfig.name == "Ollama"
    assert pconfig.inference_base_url == "http://localhost:11434/v1"


def test_ollama_aliases_resolve_to_native_provider():
    assert resolve_provider("ollama") == "ollama"
    assert resolve_provider("ollama-launch") == "ollama"
    assert resolve_provider("ollama_launch") == "ollama"
    assert resolve_provider("ollama-local") == "ollama"


def test_models_aliases_map_ollama_variants():
    assert _PROVIDER_ALIASES["ollama"] == "ollama"
    assert _PROVIDER_ALIASES["ollama-launch"] == "ollama"
    assert _PROVIDER_ALIASES["ollama_launch"] == "ollama"
    assert _PROVIDER_ALIASES["ollama-local"] == "ollama"


def test_is_local_ollama_base_url():
    assert is_local_ollama_base_url("http://localhost:11434/v1") is True
    assert is_local_ollama_base_url("http://127.0.0.1:11434") is True
    assert is_local_ollama_base_url("https://ollama.com/v1") is False
    assert is_local_ollama_base_url("http://localhost:8000/v1") is False


def test_provider_model_ids_ollama_prefers_openai_models():
    with (
        patch("hermes_cli.models._get_custom_base_url", return_value="http://localhost:11434/v1"),
        patch("hermes_cli.models.fetch_api_models", return_value=["qwen3:latest"]),
        patch("hermes_cli.models.fetch_ollama_native_models", return_value=["llama3.2:latest"]) as native_mock,
    ):
        models = provider_model_ids("ollama")

    assert models == ["qwen3:latest"]
    native_mock.assert_not_called()


def test_provider_model_ids_ollama_falls_back_to_native_tags():
    with (
        patch("hermes_cli.models._get_custom_base_url", return_value="http://localhost:11434/v1"),
        patch("hermes_cli.models.fetch_api_models", return_value=None),
        patch("hermes_cli.models.fetch_ollama_native_models", return_value=["qwen3:latest", "llama3.2:latest"]),
    ):
        models = provider_model_ids("ollama")

    assert models == ["qwen3:latest", "llama3.2:latest"]


def test_provider_model_ids_ollama_uses_seed_models_when_discovery_empty():
    with (
        patch("hermes_cli.models._get_custom_base_url", return_value="http://localhost:11434/v1"),
        patch("hermes_cli.models.fetch_api_models", return_value=None),
        patch("hermes_cli.models.fetch_ollama_native_models", return_value=[]),
    ):
        models = provider_model_ids("ollama")

    assert models == ["kimi-k2.5:cloud", "glm-5.1:cloud"]


def test_probe_ollama_native_models_reads_api_tags():
    class _Resp:
        def __init__(self, payload: dict):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(self._payload).encode("utf-8")

    with patch(
        "hermes_cli.models.urllib.request.urlopen",
        return_value=_Resp({"models": [{"name": "qwen3:latest"}, {"model": "llama3.2:latest"}]}),
    ):
        probe = probe_ollama_native_models("http://localhost:11434/v1")

    assert probe["models"] == ["qwen3:latest", "llama3.2:latest"]
    assert probe["probed_url"] == "http://localhost:11434/api/tags"


def test_validate_requested_model_ollama_native_fallback():
    with (
        patch(
            "hermes_cli.models.probe_api_models",
            return_value={
                "models": None,
                "probed_url": "http://localhost:11434/v1/models",
                "resolved_base_url": "http://localhost:11434/v1",
                "suggested_base_url": None,
                "used_fallback": False,
            },
        ),
        patch(
            "hermes_cli.models.probe_ollama_native_models",
            return_value={
                "models": ["qwen3:latest", "llama3.2:latest"],
                "probed_url": "http://localhost:11434/api/tags",
                "resolved_base_url": "http://localhost:11434/v1",
                "native_base_url": "http://localhost:11434",
                "used_fallback": False,
            },
        ),
    ):
        result = validate_requested_model(
            "qwen3:latest",
            "ollama",
            base_url="http://localhost:11434/v1",
        )

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is True


def test_model_picker_lists_ollama_without_api_key():
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("hermes_cli.models.fetch_api_models", return_value=None),
        patch("hermes_cli.models.fetch_ollama_native_models", return_value=[]),
    ):
        providers = list_authenticated_providers(current_provider="openrouter")

    ollama = next((p for p in providers if p["slug"] == "ollama"), None)
    assert ollama is not None, "ollama should appear in /model picker without API key"
    assert ollama["total_models"] >= 2
    assert "kimi-k2.5:cloud" in ollama["models"]
    assert "glm-5.1:cloud" in ollama["models"]
    assert "warning" in ollama
    assert "No local Ollama models found" in ollama["warning"]


def test_model_picker_warns_when_local_ollama_unreachable():
    with (
        patch("agent.models_dev.fetch_models_dev", return_value={}),
        patch("hermes_cli.models.fetch_api_models", return_value=None),
        patch("hermes_cli.models.fetch_ollama_native_models", return_value=None),
    ):
        providers = list_authenticated_providers(current_provider="openrouter")

    ollama = next((p for p in providers if p["slug"] == "ollama"), None)
    assert ollama is not None
    assert "warning" in ollama
    assert "Make sure Ollama is running" in ollama["warning"]
