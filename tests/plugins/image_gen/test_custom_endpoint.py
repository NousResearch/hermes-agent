"""Tests for the Custom Endpoint image generation provider plugin.

Covers:
- Config parsing (providers with capabilities.image_gen)
- Provider registration from config
- Image generation via the OpenAI-compatible /images/generations endpoint
- Per-request model override
- Error paths (missing key, invalid model, API failure)
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.custom_endpoint.plugin as ce_plugin
from agent.image_gen_provider import ImageGenProvider
from agent.image_gen_registry import _reset_for_tests, get_provider, list_providers


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_VALID_CONFIG = {
    "providers": {
        "my-gateway": {
            "name": "My Gateway",
            "base_url": "https://gateway.example.com/v1",
            "key_env": "HERMES_CUSTOM_MY_GATEWAY_API_KEY",
            "capabilities": {
                "image_gen": {
                    "models": ["gpt-image-2", "grok-imagine-image"],
                    "default_model": "grok-imagine-image",
                }
            },
        },
        "no-caps-provider": {
            "name": "No Caps",
            "base_url": "https://nocaps.example.com/v1",
            "key_env": "NO_CAPS_KEY",
        },
        "open-endpoint": {
            "name": "Open Endpoint",
            "base_url": "https://open.example.com/v1",
            # no key_env — should be available without credentials
            "capabilities": {
                "image_gen": {
                    "models": ["free-model"],
                }
            },
        },
    }
}


@pytest.fixture
def reset_registry():
    """Ensure the image_gen registry is clean before and after each test."""
    _reset_for_tests()
    yield
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

def test_load_provider_configs_finds_image_gen_capabilities():
    """Only providers with capabilities.image_gen are returned."""
    with patch("hermes_cli.config.load_config_readonly", return_value=_VALID_CONFIG):
        result = ce_plugin._load_provider_configs()
    assert "my-gateway" in result
    assert "open-endpoint" in result
    assert "no-caps-provider" not in result


def test_load_provider_configs_empty_when_no_providers():
    """Returns empty dict when no providers section exists."""
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        result = ce_plugin._load_provider_configs()
    assert result == {}


def test_load_provider_configs_skips_entries_without_models():
    """Entries with capabilities.image_gen but no models list are skipped."""
    config = {
        "providers": {
            "bad": {
                "name": "Bad",
                "base_url": "https://bad.example.com/v1",
                "key_env": "BAD_KEY",
                "capabilities": {
                    "image_gen": {"default_model": "foo"}
                },
            }
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        result = ce_plugin._load_provider_configs()
    assert result == {}


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------

def test_provider_name_uses_custom_prefix():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.name == "custom:my-gateway"


def test_provider_display_name_from_config():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.display_name == "My Gateway"


def test_provider_list_models_from_config():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    models = p.list_models()
    ids = [m["id"] for m in models]
    assert "gpt-image-2" in ids
    assert "grok-imagine-image" in ids


def test_provider_default_model_from_config():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.default_model() == "grok-imagine-image"


def test_provider_default_model_falls_back_to_first():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "open-endpoint", _VALID_CONFIG["providers"]["open-endpoint"]
    )
    assert p.default_model() == "free-model"


def test_provider_is_available_with_key_env(monkeypatch):
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    assert p.is_available() is True


def test_provider_is_available_without_key_env(monkeypatch):
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    monkeypatch.delenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", raising=False)
    assert p.is_available() is False


def test_provider_is_available_when_no_key_env_required():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "open-endpoint", _VALID_CONFIG["providers"]["open-endpoint"]
    )
    assert p.is_available() is True


def test_provider_inherits_image_gen_provider():
    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert isinstance(p, ImageGenProvider)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

def test_register_creates_providers_from_config(reset_registry, monkeypatch):
    """The register() function reads config and registers providers."""
    monkeypatch.setattr(
        ce_plugin, "_load_provider_configs",
        lambda: _VALID_CONFIG["providers"]
    )
    ctx = MagicMock()
    ce_plugin.register(ctx)
    # Should have called register_image_gen_provider for each valid entry
    assert ctx.register_image_gen_provider.call_count == 2


def test_register_no_op_when_no_caps(reset_registry, monkeypatch):
    monkeypatch.setattr(ce_plugin, "_load_provider_configs", lambda: {})
    ctx = MagicMock()
    ce_plugin.register(ctx)
    ctx.register_image_gen_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------

def _mock_b64_response():
    """Build a mock OpenAI-compatible /images/generations response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    # Small valid 1x1 PNG
    import base64
    tiny_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x00\x03\x00\x01\x82\x84\x8a\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("ascii")
    mock_resp.json.return_value = {"data": [{"b64_json": tiny_png}]}
    return mock_resp


def test_generate_success_b64(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = _mock_b64_response()
    with patch("requests.post", return_value=mock_resp):
        result = p.generate("a cat on the moon", aspect_ratio="landscape")

    assert result["success"] is True
    assert result["model"] == "grok-imagine-image"
    assert result["provider"] == "custom:my-gateway"
    assert result["aspect_ratio"] == "landscape"
    assert result["image"]  # should be a file path


def test_generate_with_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = _mock_b64_response()
    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = p.generate("a cat", aspect_ratio="square", model="gpt-image-2")

    assert result["success"] is True
    assert result["model"] == "gpt-image-2"

    # Verify the model was sent in the request
    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["model"] == "gpt-image-2"


def test_generate_invalid_model_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    result = p.generate("a cat", model="nonexistent-model")

    assert result["success"] is False
    assert "not in the configured model list" in result["error"]


def test_generate_api_failure_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = p.generate("a cat", aspect_ratio="landscape")

    assert result["success"] is False
    assert "Connection refused" in result["error"]


def test_generate_malformed_response_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"unexpected": "shape"}

    with patch("requests.post", return_value=mock_resp):
        result = p.generate("a cat")

    assert result["success"] is False
    assert "missing 'data'" in result["error"]


def test_generate_no_default_model_returns_error():
    """When no default_model is configured and no model= is passed."""
    entry = {
        "name": "NoDefault",
        "base_url": "https://nodefault.example.com/v1",
        "key_env": "NO_DEFAULT_KEY",
        "capabilities": {
            "image_gen": {
                "models": ["model-a"],
                # no default_model — but first model becomes default
            }
        },
    }
    p = ce_plugin.CustomEndpointImageGenProvider("no-default", entry)
    # Falls back to first model
    assert p.default_model() == "model-a"


def test_generate_url_fallback(monkeypatch, tmp_path):
    """When API returns a URL instead of b64_json, download and save it."""
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_plugin.CustomEndpointImageGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"data": [{"url": "https://cdn.example.com/img.png"}]}

    # Mock the download too
    mock_download = MagicMock()
    mock_download.return_value = tmp_path / "downloaded.png"

    with patch("requests.post", return_value=mock_resp):
        with patch.object(ce_plugin, "save_url_image", mock_download):
            result = p.generate("a cat", aspect_ratio="portrait")

    assert result["success"] is True
    assert str(tmp_path / "downloaded.png") in result["image"]