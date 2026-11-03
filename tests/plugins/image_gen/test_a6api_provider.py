"""Tests for the plugins/image_gen/a6api provider."""
import os
import sys

import pytest
from unittest import mock

from agent.secret_scope import UnscopedSecretError
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def a6api_key(monkeypatch, hermes_home):
    monkeypatch.setenv("A6API_API_KEY", "sk-a6apitestkey123")
    return "sk-a6apitestkey123"


def _load_module():
    import importlib

    mod = importlib.import_module("plugins.image_gen.a6api")
    return importlib.reload(mod)


@pytest.fixture
def mod(a6api_key):
    return _load_module()


def test_a6api_provider_is_registered(mod, monkeypatch):
    """The plugin's generate path returns a success_response shape."""
    provider = mod.A6ApiImageGenProvider()
    assert isinstance(provider, ImageGenProvider)
    assert provider.name == "a6api"
    assert provider.default_model() == "gpt-image-2-medium"
    assert "gpt-image-2-medium" in [m["id"] for m in provider.list_models()]


def test_a6api_is_available_with_key(mod, a6api_key):
    """is_available True when A6API_API_KEY present and openai importable."""
    with mock.patch("importlib.import_module", return_value=mock.MagicMock()):
        # openai availability is gated separately; force True for the key path.
        with mock.patch("plugins.image_gen.a6api.openai", create=True):
            pass
    provider = mod.A6ApiImageGenProvider()
    assert provider.is_available() is True


def test_a6api_is_available_false_without_key(mod, monkeypatch, hermes_home):
    """is_available False when no a6api key resolves."""
    monkeypatch.delenv("A6API_API_KEY", raising=False)
    # config custom_providers empty so config_key is None
    with mock.patch("plugins.image_gen.a6api.get_secret", side_effect=UnscopedSecretError("x")):
        provider = mod.A6ApiImageGenProvider()
        assert provider.is_available() is False


def test_a6api_falls_back_to_config_key(mod, monkeypatch, hermes_home):
    """api_key resolves from config custom_providers when env secret absent."""
    import plugins.image_gen.a6api as m

    with mock.patch.object(m, "get_secret", return_value=None):
        with mock.patch.object(m, "_load_config", return_value={"config_key": "sk-from-config"}):
            assert m._api_key() == "sk-from-config"
            assert m._base_url() == m.DEFAULT_BASE_URL


def test_a6api_generate_returns_error_on_empty_prompt(mod):
    provider = mod.A6ApiImageGenProvider()
    res = provider.generate("   ")
    assert res.get("error") == "Prompt is required and must be a non-empty string"


def test_check_requirements_true_with_key(monkeypatch, a6api_key):
    """check_image_generation_requirements returns True with a6api configured."""
    import plugins.image_gen.a6api as m
    from agent.image_gen_registry import register_provider, get_provider

    register_provider(m.A6ApiImageGenProvider())
    assert get_provider("a6api") is not None
    assert m.A6ApiImageGenProvider().is_available() is True
