#!/usr/bin/env python3
"""Tests for the MiniMax image generation provider plugin.

Mirrors ``test_xai_provider.py`` so the MiniMax backend has the same
behavioural coverage as the other built-in image-gen plugins:

* name / display_name / is_available
* list_models / default_model / capabilities / get_setup_schema
* generate() happy path, missing key, business error, network error,
  image-to-image with subject_reference, URL caching fallback
* auth header carries the right key
* register() wires the provider into the registry
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Make the provider think it has a key for every test, unless the
    individual test overrides ``MINIMAX_API_KEY`` explicitly.

    We also set ``MINIMAX_SUBSCRIPTION_KEY`` and ``MINIMAX_SUBS_KEY`` to
    empty strings — otherwise a leftover dev env var (from a real key)
    could make tests see "available" when they shouldn't.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-1234567890")
    monkeypatch.setenv("MINIMAX_SUBSCRIPTION_KEY", "")
    monkeypatch.setenv("MINIMAX_SUBS_KEY", "")


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestMiniMaxImageGenProvider:
    def test_name(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.name == "minimax"

    def test_display_name(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.display_name == "MiniMax"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-cp-...-key")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_SUBSCRIPTION_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_SUBS_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.is_available() is False

    def test_is_available_falls_back_to_subscription_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_SUBSCRIPTION_KEY", "sk-cp-...-key")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.is_available() is True

    def test_api_key_resolution_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "  test-key-1234567890  ")
        from plugins.image_gen.minimax import _resolve_api_key

        assert _resolve_api_key() == "test-key-1234567890"

    def test_api_key_resolution_ignores_blank_primary(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "   ")
        monkeypatch.setenv("MINIMAX_SUBSCRIPTION_KEY", "fallback-key")
        from plugins.image_gen.minimax import _resolve_api_key

        assert _resolve_api_key() == "fallback-key"

    def test_list_models(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        models = provider.list_models()
        assert len(models) >= 1
        assert models[0]["id"] == "image-01"

    def test_default_model(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        assert provider.default_model() == "image-01"

    def test_get_setup_schema(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "MiniMax"
        assert schema["badge"] == "subscription"
        env_keys = {v["key"] for v in schema["env_vars"]}
        assert "MINIMAX_API_KEY" in env_keys

    def test_capabilities_advertises_image_to_image(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        caps = MiniMaxImageGenProvider().capabilities()
        assert "text" in caps["modalities"]
        assert "image" in caps["modalities"]
        assert caps["max_reference_images"] == 1


# ---------------------------------------------------------------------------
# Config / resolution helpers
# ---------------------------------------------------------------------------


class TestConfigHelpers:
    def test_resolve_model_default(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_IMAGE_MODEL", raising=False)
        from plugins.image_gen.minimax import _resolve_model

        with patch("hermes_cli.config.load_config", return_value={}):
            assert _resolve_model() == "image-01"

    def test_resolve_model_env_override(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_IMAGE_MODEL", "image-02-future")
        from plugins.image_gen.minimax import _resolve_model

        assert _resolve_model() == "image-02-future"

    def test_resolve_model_scoped_config_override(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_IMAGE_MODEL", raising=False)
        from plugins.image_gen.minimax import _resolve_model

        with patch(
            "hermes_cli.config.load_config",
            return_value={"image_gen": {"minimax": {"model": "image-02-future"}}},
        ):
            assert _resolve_model() == "image-02-future"

    def test_resolve_model_ignores_unrelated_top_level_model(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_IMAGE_MODEL", raising=False)
        from plugins.image_gen.minimax import _resolve_model

        with patch(
            "hermes_cli.config.load_config",
            return_value={"image_gen": {"model": "gpt-image-2"}},
        ):
            assert _resolve_model() == "image-01"

    def test_resolve_base_url_default(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
        from plugins.image_gen.minimax import _resolve_base_url

        assert _resolve_base_url() == "https://api.minimax.io/v1"

    def test_resolve_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.example.com/v1/")
        from plugins.image_gen.minimax import _resolve_base_url

        # trailing slash stripped
        assert _resolve_base_url() == "https://api.example.com/v1"

    def test_aspect_to_ratio_known(self):
        from plugins.image_gen.minimax import _aspect_to_ratio

        assert _aspect_to_ratio("landscape") == "16:9"
        assert _aspect_to_ratio("square") == "1:1"
        assert _aspect_to_ratio("portrait") == "9:16"

    def test_aspect_to_ratio_passthrough(self):
        from plugins.image_gen.minimax import _aspect_to_ratio

        # If the caller already passes a valid MiniMax ratio string,
        # _aspect_to_ratio should not mangle it.
        assert _aspect_to_ratio("4:3") == "4:3"
        assert _aspect_to_ratio("21:9") == "21:9"

    def test_aspect_to_ratio_unknown_falls_back_to_landscape(self):
        from plugins.image_gen.minimax import _aspect_to_ratio

        assert _aspect_to_ratio("banana") == "16:9"


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


def _mock_ok_response(image_url: str = "https://hailuo-image.example/test.jpeg") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "id": "abc123",
        "data": {"image_urls": [image_url]},
        "metadata": {"failed_count": "0", "success_count": "1"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    return resp


class TestGenerate:
    def test_missing_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_SUBSCRIPTION_KEY", raising=False)
        monkeypatch.delenv("MINIMAX_SUBS_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate(prompt="a cat")
        assert result["success"] is False
        assert result["provider"] == "minimax"
        assert result["error_type"] == "missing_credentials"
        assert "MINIMAX_API_KEY" in result["error"]
        assert "D:\\Hermes\\home" not in result["error"]

    def test_blank_prompt_returns_invalid_argument(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch("plugins.image_gen.minimax.requests.post") as mock_post:
            result = MiniMaxImageGenProvider().generate(prompt="   ")

        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        mock_post.assert_not_called()

    def test_successful_url_response_is_cached_locally(self):
        """MiniMax returns signed Aliyun OSS URLs that expire. The plugin
        must download them to a local path so the gateway doesn't 404."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://hailuo-image.example/x.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/minimax_image-01_20260625_000000_deadbeef.jpg"),
        ) as mock_save:
            result = MiniMaxImageGenProvider().generate(prompt="a poodle on a rocket")

        assert result["success"] is True
        assert Path(result["image"]) == Path("/tmp/minimax_image-01_20260625_000000_deadbeef.jpg")
        assert result["provider"] == "minimax"
        assert result["model"] == "image-01"
        assert result["modality"] == "text"
        assert result["cache_status"] == "cached"
        assert "remote_url" not in result
        # Download was triggered, with the right URL and prefix.
        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        assert args[0] == "https://hailuo-image.example/x.jpeg"
        assert kwargs.get("prefix", "").startswith("minimax")

    def test_url_response_falls_back_to_bare_url_on_cache_failure(self):
        """If the CDN download fails (expired URL, network blip), we
        still return success with the raw URL so the gateway has a chance
        to retry — never turn it into a hard tool error."""
        import requests as req_lib
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://hailuo-image.example/expired.jpeg"),
        ), patch(
            "plugins.image_gen.minimax.save_url_image",
            side_effect=req_lib.HTTPError("404"),
        ):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is True
        assert result["image"] == "https://hailuo-image.example/expired.jpeg"
        assert result["modality"] == "text"

    def test_business_error_returns_provider_error(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": None,
            "base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
        }

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert result["error_type"] == "provider_error"
        assert "1008" in result["error"]
        assert "insufficient balance" in result["error"]

    def test_auth_error_returns_provider_error(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": None,
            "base_resp": {"status_code": 1004, "status_msg": "login fail"},
        }

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert "1004" in result["error"]

    def test_http_error_returns_api_error_with_status_and_body(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        assert "unauthorized" in result["error"]

    def test_network_error_returns_network_error(self):
        import requests as req_lib
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            side_effect=req_lib.ConnectionError("DNS failure"),
        ):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert result["error_type"] == "network_error"

    def test_invalid_json_returns_invalid_response(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert result["error_type"] == "invalid_response"

    def test_empty_image_urls_returns_empty_result(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"image_urls": []},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider().generate(prompt="a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_result"

    def test_image_to_image_sends_subject_reference(self):
        """When image_url is supplied, the plugin must add
        subject_reference to the payload and report modality=image."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/ref.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/ref.jpg"),
        ):
            result = MiniMaxImageGenProvider().generate(
                prompt="same cat, different background",
                image_url="https://example.com/original-cat.jpg",
            )

        assert result["success"] is True
        assert result["modality"] == "image"
        payload = mock_post.call_args.kwargs["json"]
        assert "subject_reference" in payload
        assert payload["subject_reference"][0]["image_file"] == "https://example.com/original-cat.jpg"
        assert payload["subject_reference"][0]["type"] == "character"

    def test_reference_image_urls_falls_back_to_first_when_no_image_url(self):
        """When only reference_image_urls is given (no image_url), the
        first entry becomes the subject_reference image_file."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/ref.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/ref.jpg"),
        ):
            MiniMaxImageGenProvider().generate(
                prompt="same character in new context",
                reference_image_urls=[
                    "https://example.com/character1.jpg",
                    "https://example.com/character2.jpg",
                ],
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["subject_reference"][0]["image_file"] == "https://example.com/character1.jpg"

    def test_auth_header_contains_bearer_token(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/x.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/x.jpg"),
        ):
            MiniMaxImageGenProvider().generate(prompt="a cat")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test-key-1234567890"

    def test_endpoint_url_uses_base_url(self, monkeypatch):
        """The plugin must hit the resolved base_url, not hardcoded."""
        monkeypatch.setenv("MINIMAX_BASE_URL", "https://staging.example.com/v9")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/x.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/x.jpg"),
        ):
            MiniMaxImageGenProvider().generate(prompt="a cat")

        assert mock_post.call_args.args[0] == "https://staging.example.com/v9/image_generation"

    def test_payload_carries_correct_aspect_ratio(self):
        """landscape must hit the wire as 16:9, not the tool-level name."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/x.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/x.jpg"),
        ):
            MiniMaxImageGenProvider().generate(prompt="a cat", aspect_ratio="portrait")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["aspect_ratio"] == "9:16"

    def test_seed_and_num_images_forwarded(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/x.jpeg"),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/x.jpg"),
        ):
            MiniMaxImageGenProvider().generate(
                prompt="a cat", num_images=2, seed=42,
            )

        payload = mock_post.call_args.kwargs["json"]
        assert payload["num_images"] == 2
        assert payload["seed"] == 42

    def test_unknown_kwargs_are_ignored(self):
        """ABC contract: implementations MUST NOT TypeError on unknown
        kwargs (forward-compat for future schema additions)."""
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_mock_ok_response("https://example.com/x.jpeg"),
        ), patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value=Path("/tmp/x.jpg"),
        ):
            # Future-version kwargs that don't exist today.
            result = MiniMaxImageGenProvider().generate(
                prompt="a cat",
                future_param=True,
                another_one={"nested": "value"},
            )

        assert result["success"] is True


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, MiniMaxImageGenProvider)
        assert provider.name == "minimax"
