"""Tests for the Custom Endpoint video generation provider plugin.

Covers:
- Config parsing (providers with capabilities.video_gen)
- Provider registration from config
- Video generation via the OpenAI-compatible /videos endpoint (sync + async)
- Per-request model override
- Error paths (missing key, invalid model, API failure, poll timeout)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import plugins.video_gen.custom_endpoint.plugin as ce_video_plugin
from agent.video_gen_provider import VideoGenProvider
from agent.video_gen_registry import _reset_for_tests


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
                "video_gen": {
                    "models": ["grok-imagine-video", "grok-imagine-video-1.5"],
                    "default_model": "grok-imagine-video",
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
            "capabilities": {
                "video_gen": {
                    "models": ["free-video-model"],
                }
            },
        },
    }
}


@pytest.fixture
def reset_registry():
    """Ensure the video_gen registry is clean before and after each test."""
    _reset_for_tests()
    yield
    _reset_for_tests()


# ---------------------------------------------------------------------------
# Config parsing tests
# ---------------------------------------------------------------------------

def test_load_provider_configs_finds_video_gen_capabilities():
    with patch("hermes_cli.config.load_config_readonly", return_value=_VALID_CONFIG):
        result = ce_video_plugin._load_provider_configs()
    assert "my-gateway" in result
    assert "open-endpoint" in result
    assert "no-caps-provider" not in result


def test_load_provider_configs_empty_when_no_providers():
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        result = ce_video_plugin._load_provider_configs()
    assert result == {}


def test_load_provider_configs_skips_entries_without_models():
    config = {
        "providers": {
            "bad": {
                "name": "Bad",
                "base_url": "https://bad.example.com/v1",
                "key_env": "BAD_KEY",
                "capabilities": {
                    "video_gen": {"default_model": "foo"}
                },
            }
        }
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        result = ce_video_plugin._load_provider_configs()
    assert result == {}


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------

def test_provider_name_uses_custom_prefix():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.name == "custom:my-gateway"


def test_provider_display_name_from_config():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.display_name == "My Gateway"


def test_provider_list_models_from_config():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    models = p.list_models()
    ids = [m["id"] for m in models]
    assert "grok-imagine-video" in ids
    assert "grok-imagine-video-1.5" in ids


def test_provider_default_model_from_config():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert p.default_model() == "grok-imagine-video"


def test_provider_default_model_falls_back_to_first():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "open-endpoint", _VALID_CONFIG["providers"]["open-endpoint"]
    )
    assert p.default_model() == "free-video-model"


def test_provider_is_available_with_key_env(monkeypatch):
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    assert p.is_available() is True


def test_provider_is_available_without_key_env(monkeypatch):
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    monkeypatch.delenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", raising=False)
    assert p.is_available() is False


def test_provider_is_available_when_no_key_env_required():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "open-endpoint", _VALID_CONFIG["providers"]["open-endpoint"]
    )
    assert p.is_available() is True


def test_provider_inherits_video_gen_provider():
    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )
    assert isinstance(p, VideoGenProvider)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------

def test_register_creates_providers_from_config(reset_registry, monkeypatch):
    monkeypatch.setattr(
        ce_video_plugin, "_load_provider_configs",
        lambda: _VALID_CONFIG["providers"]
    )
    ctx = MagicMock()
    ce_video_plugin.register(ctx)
    assert ctx.register_video_gen_provider.call_count == 2


def test_register_no_op_when_no_caps(reset_registry, monkeypatch):
    monkeypatch.setattr(ce_video_plugin, "_load_provider_configs", lambda: {})
    ctx = MagicMock()
    ce_video_plugin.register(ctx)
    ctx.register_video_gen_provider.assert_not_called()


# ---------------------------------------------------------------------------
# Generate tests — sync path (direct URL in response)
# ---------------------------------------------------------------------------

def test_generate_sync_success_url(monkeypatch, tmp_path):
    """Sync endpoint returns video URL directly in the create response."""
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"url": "https://cdn.example.com/video.mp4"}]
    }

    mock_download = MagicMock(return_value=tmp_path / "downloaded.mp4")

    with patch("requests.post", return_value=mock_resp):
        with patch.object(ce_video_plugin, "save_url_video", mock_download):
            result = p.generate("a cat running", aspect_ratio="16:9")

    assert result["success"] is True
    assert result["model"] == "grok-imagine-video"
    assert result["provider"] == "custom:my-gateway"
    assert result["modality"] == "text"


def test_generate_with_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"url": "https://cdn.example.com/video.mp4"}]
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        with patch.object(ce_video_plugin, "save_url_video", return_value=tmp_path / "v.mp4"):
            result = p.generate("a cat", model="grok-imagine-video-1.5")

    assert result["success"] is True
    assert result["model"] == "grok-imagine-video-1.5"
    sent = mock_post.call_args[1]["json"]
    assert sent["model"] == "grok-imagine-video-1.5"


def test_generate_image_to_video_sets_modality(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"url": "https://cdn.example.com/video.mp4"}]
    }

    with patch("requests.post", return_value=mock_resp):
        with patch.object(ce_video_plugin, "save_url_video", return_value=tmp_path / "v.mp4"):
            result = p.generate("animate this", image_url="https://example.com/img.png")

    assert result["success"] is True
    assert result["modality"] == "image"


def test_generate_invalid_model_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    result = p.generate("a cat", model="nonexistent-model")
    assert result["success"] is False
    assert "not in the configured model list" in result["error"]


def test_generate_api_failure_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    with patch("requests.post", side_effect=Exception("Connection refused")):
        result = p.generate("a cat")

    assert result["success"] is False
    assert "Connection refused" in result["error"]


def test_generate_malformed_response_returns_error(monkeypatch):
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"unexpected": "shape"}

    with patch("requests.post", return_value=mock_resp):
        result = p.generate("a cat")

    assert result["success"] is False
    assert "neither a video URL nor a job ID" in result["error"]


# ---------------------------------------------------------------------------
# Generate tests — async path (poll for completion)
# ---------------------------------------------------------------------------

def test_generate_async_poll_success(monkeypatch, tmp_path):
    """Async endpoint returns a job ID, then poll until completed with URL."""
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    # Create response: job pending
    create_resp = MagicMock()
    create_resp.raise_for_status = MagicMock()
    create_resp.json.return_value = {"id": "vid-123", "status": "pending"}

    # First poll: still processing
    poll_resp_1 = MagicMock()
    poll_resp_1.raise_for_status = MagicMock()
    poll_resp_1.json.return_value = {"id": "vid-123", "status": "processing"}

    # Second poll: completed with URL
    poll_resp_2 = MagicMock()
    poll_resp_2.raise_for_status = MagicMock()
    poll_resp_2.json.return_value = {
        "id": "vid-123",
        "status": "completed",
        "data": [{"url": "https://cdn.example.com/result.mp4"}],
    }

    mock_download = MagicMock(return_value=tmp_path / "result.mp4")

    # Patch time.sleep to skip waiting
    with patch("requests.post", return_value=create_resp):
        with patch("requests.get", side_effect=[poll_resp_1, poll_resp_2]):
            with patch("time.sleep"):
                with patch.object(ce_video_plugin, "save_url_video", mock_download):
                    result = p.generate("a cat running")

    assert result["success"] is True
    assert result["model"] == "grok-imagine-video"


def test_generate_async_poll_failure(monkeypatch):
    """Async endpoint job fails — should return error response."""
    monkeypatch.setenv("HERMES_CUSTOM_MY_GATEWAY_API_KEY", "sk-test-123")

    p = ce_video_plugin.CustomEndpointVideoGenProvider(
        "my-gateway", _VALID_CONFIG["providers"]["my-gateway"]
    )

    create_resp = MagicMock()
    create_resp.raise_for_status = MagicMock()
    create_resp.json.return_value = {"id": "vid-456", "status": "pending"}

    poll_resp = MagicMock()
    poll_resp.raise_for_status = MagicMock()
    poll_resp.json.return_value = {
        "id": "vid-456",
        "status": "failed",
        "error": {"message": "content policy violation"},
    }

    with patch("requests.post", return_value=create_resp):
        with patch("requests.get", return_value=poll_resp):
            with patch("time.sleep"):
                result = p.generate("inappropriate content")

    assert result["success"] is False
    assert "content policy violation" in result["error"]


# ---------------------------------------------------------------------------
# URL extraction helper tests
# ---------------------------------------------------------------------------

def test_extract_video_url_openai_shape():
    body = {"data": [{"url": "https://cdn.example.com/v.mp4"}]}
    assert ce_video_plugin._extract_video_url(body) == "https://cdn.example.com/v.mp4"


def test_extract_video_url_direct():
    body = {"url": "https://cdn.example.com/v.mp4"}
    assert ce_video_plugin._extract_video_url(body) == "https://cdn.example.com/v.mp4"


def test_extract_video_url_output():
    body = {"output": "https://cdn.example.com/v.mp4"}
    assert ce_video_plugin._extract_video_url(body) == "https://cdn.example.com/v.mp4"


def test_extract_video_url_nested():
    body = {"video": {"url": "https://cdn.example.com/v.mp4"}}
    assert ce_video_plugin._extract_video_url(body) == "https://cdn.example.com/v.mp4"


def test_extract_video_url_none():
    body = {"status": "pending"}
    assert ce_video_plugin._extract_video_url(body) is None


def test_generate_no_default_model_falls_back():
    entry = {
        "name": "NoDefault",
        "base_url": "https://nodefault.example.com/v1",
        "key_env": "NO_DEFAULT_KEY",
        "capabilities": {
            "video_gen": {
                "models": ["model-a"],
            }
        },
    }
    p = ce_video_plugin.CustomEndpointVideoGenProvider("no-default", entry)
    assert p.default_model() == "model-a"