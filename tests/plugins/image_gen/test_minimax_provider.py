#!/usr/bin/env python3
"""Tests for the MiniMax image generation provider (global + China)."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure MINIMAX_API_KEY is set for all tests."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-12345")


def _ok_response(payload):
    """Build a MagicMock HTTP response returning ``payload`` as JSON."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload
    return mock_resp


_SUCCESS_B64_RESPONSE = {
    "id": "trace-abc",
    "data": {"image_base64": ["dGVzdC1pbWFnZS1kYXRh"]},
    "metadata": {"success_count": "1", "failed_count": "0"},
    "base_resp": {"status_code": 0, "status_msg": "success"},
}

_SUCCESS_URL_RESPONSE = {
    "id": "trace-abc",
    "data": {"image_urls": ["https://example.com/img.png"]},
    "metadata": {"success_count": "1", "failed_count": "0"},
    "base_resp": {"status_code": 0, "status_msg": "success"},
}


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestMiniMaxImageGenProvider:
    def test_name(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").name == "minimax"
        assert MiniMaxImageGenProvider("minimax-cn").name == "minimax-cn"

    def test_unknown_region_rejected(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with pytest.raises(ValueError):
            MiniMaxImageGenProvider("minimax-eu")

    def test_display_name(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").display_name == "MiniMax"
        assert MiniMaxImageGenProvider("minimax-cn").display_name == "MiniMax (China)"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-xxx")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").is_available() is False

    def test_cn_uses_own_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "cn-key")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").is_available() is False
        assert MiniMaxImageGenProvider("minimax-cn").is_available() is True

    def test_list_models(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        models = MiniMaxImageGenProvider("minimax").list_models()
        ids = [m["id"] for m in models]
        assert "image-01" in ids
        assert "image-01-live" in ids

    def test_default_model(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider("minimax").default_model() == "image-01"

    def test_capabilities_include_image_to_image(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        caps = MiniMaxImageGenProvider("minimax").capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 2

    def test_get_setup_schema(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        schema = MiniMaxImageGenProvider("minimax").get_setup_schema()
        assert schema["name"] == "MiniMax"
        assert schema["badge"] == "paid"
        assert schema["env_vars"][0]["key"] == "MINIMAX_API_KEY"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_model_resolution_default(self):
        from plugins.image_gen.minimax import _resolve_model

        assert _resolve_model(None) == "image-01"
        assert _resolve_model("not-a-model") == "image-01"

    def test_model_resolution_valid(self):
        from plugins.image_gen.minimax import _resolve_model

        assert _resolve_model("image-01-live") == "image-01-live"

    def test_split_base64_image_strips_data_uri(self):
        from plugins.image_gen.minimax import _split_base64_image

        raw, ext = _split_base64_image("data:image/jpeg;base64,aGVsbG8=")
        assert raw == "aGVsbG8="
        assert ext == "jpg"


# ---------------------------------------------------------------------------
# Generate tests — text-to-image
# ---------------------------------------------------------------------------


class TestGenerateTextToImage:
    def test_empty_prompt(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider("minimax").generate(prompt="")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_oversized_prompt(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider("minimax").generate(prompt="x" * 2000)
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "1500" in result["error"]

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider("minimax").generate(prompt="test")
        assert result["success"] is False
        assert "MINIMAX_API_KEY" in result["error"]
        assert result["error_type"] == "auth_required"

    def test_invalid_response_format(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider("minimax").generate(
            prompt="test", response_format="gif"
        )
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_successful_base64_generation(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_B64_RESPONSE),
        ), patch(
            "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
        ) as mock_save:
            provider = MiniMaxImageGenProvider("minimax")
            result = provider.generate(
                prompt="A cat playing piano", response_format="base64"
            )

        assert result["success"] is True
        assert result["image"] == "/tmp/test.png"
        assert result["provider"] == "minimax"
        assert result["model"] == "image-01"
        assert result["modality"] == "text"
        mock_save.assert_called_once()
        assert mock_save.call_args.kwargs.get("prefix", "").startswith("minimax_")

    def test_successful_url_generation_is_cached(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ), patch(
            "plugins.image_gen.minimax.save_url_image",
            return_value="/tmp/minimax_cached.png",
        ) as mock_save:
            provider = MiniMaxImageGenProvider("minimax")
            result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/minimax_cached.png"
        call_args, _ = mock_save.call_args
        assert call_args[0] == "https://example.com/img.png"
        assert mock_save.call_args.kwargs.get("prefix", "").startswith("minimax_")

    def test_url_falls_back_when_cache_fails(self):
        import requests as req_lib

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ), patch(
            "plugins.image_gen.minimax.save_url_image",
            side_effect=req_lib.HTTPError("404"),
        ):
            provider = MiniMaxImageGenProvider("minimax")
            result = provider.generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"

    def test_base64_request_uses_base64_field(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_B64_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            provider.generate(prompt="test", response_format="base64")

        payload = mock_post.call_args.kwargs.get("json") or {}
        assert payload["response_format"] == "base64"
        assert payload["model"] == "image-01"
        assert payload["n"] == 1

    def test_auth_header_and_global_endpoint(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_B64_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            provider.generate(prompt="test")

        assert mock_post.call_args.args[0] == "https://api.minimax.io/v1/image_generation"
        headers = mock_post.call_args.kwargs.get("headers") or {}
        assert headers["Authorization"] == "Bearer test-key-12345"
        assert headers["Content-Type"] == "application/json"

    def test_cn_endpoint_used(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "cn-key")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_B64_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
        ):
            provider = MiniMaxImageGenProvider("minimax-cn")
            provider.generate(prompt="test")

        assert mock_post.call_args.args[0] == "https://api.minimaxi.com/v1/image_generation"

    def test_aspect_ratio_mapping(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        for hermes_ar, expected_native in [
            ("landscape", "16:9"),
            ("square", "1:1"),
            ("portrait", "9:16"),
        ]:
            with patch(
                "plugins.image_gen.minimax.requests.post",
                return_value=_ok_response(_SUCCESS_B64_RESPONSE),
            ) as mock_post, patch(
                "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
            ):
                provider = MiniMaxImageGenProvider("minimax")
                provider.generate(prompt="test", aspect_ratio=hermes_ar)
            payload = mock_post.call_args.kwargs.get("json") or {}
            assert payload["aspect_ratio"] == expected_native

    def test_seed_and_prompt_optimizer_passthrough(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_B64_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_b64_image", return_value="/tmp/test.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            provider.generate(
                prompt="test", seed=12345, prompt_optimizer=True, width=1024, height=1024
            )

        payload = mock_post.call_args.kwargs.get("json") or {}
        assert payload["seed"] == 12345
        assert payload["prompt_optimizer"] is True
        assert payload["width"] == 1024
        assert payload["height"] == 1024


# ---------------------------------------------------------------------------
# Generate tests — image-to-image (subject_reference)
# ---------------------------------------------------------------------------


class TestGenerateImageToImage:
    def test_routes_to_subject_reference_with_live_model(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image", return_value="/tmp/edit.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            result = provider.generate(
                prompt="make her smile",
                image_url="https://example.com/portrait.jpg",
            )

        assert result["success"] is True
        assert result["modality"] == "image"
        assert result["model"] == "image-01-live"
        payload = mock_post.call_args.kwargs.get("json") or {}
        assert payload["model"] == "image-01-live"
        assert payload["subject_reference"] == [
            {
                "type": "character",
                "image_file": "https://example.com/portrait.jpg",
            }
        ]

    def test_reference_images_are_appended(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image", return_value="/tmp/edit.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            provider.generate(
                prompt="keep both faces",
                image_url="https://example.com/a.jpg",
                reference_image_urls=["https://example.com/b.jpg"],
            )

        payload = mock_post.call_args.kwargs.get("json") or {}
        assert payload["model"] == "image-01-live"
        assert payload["subject_reference"] == [
            {"type": "character", "image_file": "https://example.com/a.jpg"},
            {"type": "character", "image_file": "https://example.com/b.jpg"},
        ]

    def test_local_path_becomes_data_url(self, tmp_path):
        image_file = tmp_path / "portrait.png"
        image_file.write_bytes(b"\x89PNG\r\n\x1a\nfakepngdata")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ) as mock_post, patch(
            "plugins.image_gen.minimax.save_url_image", return_value="/tmp/edit.png"
        ):
            provider = MiniMaxImageGenProvider("minimax")
            provider.generate(
                prompt="make her smile", image_url=str(image_file)
            )

        payload = mock_post.call_args.kwargs.get("json") or {}
        reference = payload["subject_reference"][0]
        assert reference["type"] == "character"
        assert reference["image_file"].startswith("data:image/png;base64,")
        decoded = base64.b64decode(reference["image_file"].split(",", 1)[1])
        assert decoded == b"\x89PNG\r\n\x1a\nfakepngdata"

    def test_too_many_references(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider("minimax")
        result = provider.generate(
            prompt="test",
            image_url="https://example.com/a.jpg",
            reference_image_urls=[
                "https://example.com/b.jpg",
                "https://example.com/c.jpg",
                "https://example.com/d.jpg",
            ],
        )
        assert result["success"] is False
        assert result["error_type"] == "too_many_references"

    def test_missing_local_file_returns_io_error(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider("minimax")
        result = provider.generate(
            prompt="test", image_url="/nonexistent/path.png"
        )
        assert result["success"] is False
        assert result["error_type"] == "io_error"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_base_resp_failure_surfaces_msg(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = _ok_response(
            {
                "data": {},
                "base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
            }
        )
        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "provider_error"
        assert "insufficient balance" in result["error"]

    def test_http_error(self):
        import requests as req_lib

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_resp.json.return_value = {"base_resp": {"status_msg": "Invalid API key"}}
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        assert "Invalid API key" in result["error"]

    def test_timeout(self):
        import requests as req_lib

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post", side_effect=req_lib.Timeout()
        ):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_connection_error(self):
        import requests as req_lib

        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            side_effect=req_lib.ConnectionError("nope"),
        ):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "connection_error"

    def test_invalid_json_response(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "<html>error</html>"

        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "invalid_response"

    def test_empty_data(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        mock_resp = _ok_response(
            {"data": {"image_urls": []}, "base_resp": {"status_code": 0}}
        )
        with patch("plugins.image_gen.minimax.requests.post", return_value=mock_resp):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_metadata_surfaces_success_count(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        with patch(
            "plugins.image_gen.minimax.requests.post",
            return_value=_ok_response(_SUCCESS_URL_RESPONSE),
        ), patch(
            "plugins.image_gen.minimax.save_url_image", return_value="/tmp/edit.png"
        ):
            result = MiniMaxImageGenProvider("minimax").generate(prompt="test")

        assert result["success_count"] == "1"
        assert result["failed_count"] == "0"


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_both_regional_providers(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        registered = [
            call.args[0] for call in mock_ctx.register_image_gen_provider.call_args_list
        ]
        names = {provider.name for provider in registered}
        assert names == {"minimax", "minimax-cn"}
        assert all(isinstance(p, MiniMaxImageGenProvider) for p in registered)
