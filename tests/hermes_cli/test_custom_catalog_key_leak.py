"""``_custom_catalog`` must gate which env API key it sends by the custom endpoint's host, exactly
like the runtime chat path (``hermes_cli.runtime_provider``) already does for GHSA-76xc-57q6-vm5m.
Before the fix it tried ``CUSTOM_API_KEY``/``OPENAI_API_KEY``/``OPENROUTER_API_KEY`` unconditionally,
so opening ``/model`` with ``provider: custom`` and a third-party ``base_url`` sent the user's
OpenRouter/OpenAI key to that host.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import hermes_cli.models as mod


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("CUSTOM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OLLAMA_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def _capture_fetch_api_models():
    seen = []

    def _fake(api_key, base_url, **kwargs):
        seen.append((api_key, base_url))
        return ["m"]

    return seen, _fake


def test_third_party_base_url_does_not_receive_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    seen, fake = _capture_fetch_api_models()
    with patch.object(mod, "_get_custom_base_url", return_value="https://third-party.example/v1"), \
         patch.object(mod, "_get_model_config_dict", return_value={}), \
         patch.object(mod, "fetch_api_models", fake):
        mod._custom_catalog("custom", False)
    assert seen == [("", "https://third-party.example/v1")]


def test_openrouter_base_url_receives_openrouter_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    seen, fake = _capture_fetch_api_models()
    with patch.object(mod, "_get_custom_base_url", return_value="https://openrouter.ai/api/v1"), \
         patch.object(mod, "_get_model_config_dict", return_value={}), \
         patch.object(mod, "fetch_api_models", fake):
        mod._custom_catalog("custom", False)
    assert seen == [("sk-or-SECRET", "https://openrouter.ai/api/v1")]


def test_profile_api_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-env-key")
    seen, fake = _capture_fetch_api_models()
    with patch.object(mod, "_get_custom_base_url", return_value="https://third-party.example/v1"), \
         patch.object(mod, "_get_model_config_dict", return_value={"api_key": "profile-key"}), \
         patch.object(mod, "fetch_api_models", fake):
        mod._custom_catalog("custom", False)
    assert seen == [("profile-key", "https://third-party.example/v1")]


def test_custom_api_key_used_for_third_party_host(monkeypatch):
    monkeypatch.setenv("CUSTOM_API_KEY", "custom-env-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    seen, fake = _capture_fetch_api_models()
    with patch.object(mod, "_get_custom_base_url", return_value="https://third-party.example/v1"), \
         patch.object(mod, "_get_model_config_dict", return_value={}), \
         patch.object(mod, "fetch_api_models", fake):
        mod._custom_catalog("custom", False)
    assert seen == [("custom-env-key", "https://third-party.example/v1")]
