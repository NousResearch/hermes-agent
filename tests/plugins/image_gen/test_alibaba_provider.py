#!/usr/bin/env python3
"""Tests for Alibaba DashScope image generation provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch, tmp_path):
    """Ensure DASHSCOPE_API_KEY is set for all tests."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Provider class tests
# ---------------------------------------------------------------------------


class TestAlibabaImageGenProvider:
    def test_name(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        assert provider.name == "alibaba"

    def test_display_name(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        assert provider.display_name == "Alibaba (Wan 2.7)"

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-xxx")
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        assert provider.is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        assert provider.is_available() is False

    def test_list_models(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        models = provider.list_models()
        assert len(models) >= 2
        ids = {m["id"] for m in models}
        assert "wan2.7-image-pro" in ids
        assert "wan2.7-image" in ids

    def test_default_model(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        assert provider.default_model() == "wan2.7-image-pro"

    def test_get_setup_schema(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        schema = provider.get_setup_schema()
        assert schema["name"] == "Alibaba Wan 2.7 (image)"
        assert schema["badge"] == "paid"
        assert len(schema["env_vars"]) == 1
        assert schema["env_vars"][0]["key"] == "DASHSCOPE_API_KEY"

    def test_capabilities(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        caps = AlibabaImageGenProvider().capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 9


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_model(self):
        from plugins.image_gen.alibaba import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "wan2.7-image-pro"

    def test_env_override_model(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_IMAGE_MODEL", "wan2.7-image")
        from plugins.image_gen.alibaba import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "wan2.7-image"

    def test_env_override_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_IMAGE_MODEL", "nonexistent-model")
        from plugins.image_gen.alibaba import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "wan2.7-image-pro"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
        from plugins.image_gen.alibaba import _resolve_base_url

        assert _resolve_base_url() == "https://dashscope-intl.aliyuncs.com/api/v1"

    def test_env_base_url(self, monkeypatch):
        monkeypatch.setenv(
            "DASHSCOPE_BASE_URL",
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
        )
        from plugins.image_gen.alibaba import _resolve_base_url

        assert (
            _resolve_base_url()
            == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )


# ---------------------------------------------------------------------------
# Endpoint derivation tests
# ---------------------------------------------------------------------------


class TestEndpointDerivation:
    def test_standard_api_v1(self):
        from plugins.image_gen.alibaba import _image_generation_endpoint

        url = _image_generation_endpoint("https://dashscope-intl.aliyuncs.com/api/v1")
        assert url == (
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/"
            "image-generation/generation"
        )

    def test_compatible_mode_v1(self):
        from plugins.image_gen.alibaba import _image_generation_endpoint

        url = _image_generation_endpoint(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        assert url == (
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/"
            "image-generation/generation"
        )

    def test_token_plan_compatible_mode(self):
        from plugins.image_gen.alibaba import _image_generation_endpoint

        url = _image_generation_endpoint(
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )
        assert url == (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/api/v1/services/aigc/"
            "image-generation/generation"
        )

    def test_token_plan_native_api(self):
        from plugins.image_gen.alibaba import _image_generation_endpoint

        url = _image_generation_endpoint(
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/api/v1"
        )
        assert url == (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/api/v1/services/aigc/"
            "image-generation/generation"
        )

    def test_trailing_slash_stripped(self):
        from plugins.image_gen.alibaba import _image_generation_endpoint

        url = _image_generation_endpoint(
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"
        )
        assert url == (
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/"
            "image-generation/generation"
        )


# ---------------------------------------------------------------------------
# Image content item tests
# ---------------------------------------------------------------------------


class TestImageToContentItem:
    def test_https_url_passthrough(self):
        from plugins.image_gen.alibaba import _image_to_content_item

        item = _image_to_content_item("https://example.com/image.png")
        assert item == {"image": "https://example.com/image.png"}

    def test_data_uri_passthrough(self):
        from plugins.image_gen.alibaba import _image_to_content_item

        uri = "data:image/png;base64,abc123"
        item = _image_to_content_item(uri)
        assert item == {"image": uri}

    def test_local_file_to_base64(self, tmp_path):
        from plugins.image_gen.alibaba import _image_to_content_item

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        item = _image_to_content_item(str(img))
        assert item["image"].startswith("data:image/png;base64,")

    def test_local_jpg_extension_mapped(self, tmp_path):
        from plugins.image_gen.alibaba import _image_to_content_item

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        item = _image_to_content_item(str(img))
        assert item["image"].startswith("data:image/jpeg;base64,")

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        from plugins.image_gen.alibaba import _image_to_content_item

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")
        item = _image_to_content_item("~/pic.png")
        assert item["image"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


def _mock_dashscope_response(image_url="https://oss.example.com/image.png"):
    """Build a mock response matching the DashScope image-generation API."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "request_id": "test-req-123",
        "output": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "image", "image": image_url}
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "debug_info": [
                {
                    "actual_seed": 42,
                    "output_W": 1024,
                    "output_H": 1024,
                    "inference_cost": 5.5,
                }
            ],
        },
    }
    return mock_resp


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        result = provider.generate(prompt="test")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"
        assert "DASHSCOPE_API_KEY" in result["error"]

    def test_successful_text_to_image(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/dashscope_test.png"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/dashscope_test.png"
        assert result["provider"] == "alibaba"
        assert result["model"] == "wan2.7-image-pro"
        assert result["modality"] == "text"

    def test_successful_image_edit(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/dashscope_edit.png"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(
                    prompt="Make the sky purple",
                    image_url="https://example.com/photo.png",
                )

        assert result["success"] is True
        assert result["modality"] == "image"
        # Verify the image was included in the content array
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        content = payload["input"]["messages"][0]["content"]
        assert content[0] == {"image": "https://example.com/photo.png"}
        assert content[1] == {"text": "Make the sky purple"}

    def test_multiple_reference_images(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/dashscope_multi.png"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(
                    prompt="Combine these styles",
                    image_url="https://example.com/ref1.png",
                    reference_image_urls=[
                        "https://example.com/ref2.png",
                        "https://example.com/ref3.png",
                    ],
                )

        assert result["success"] is True
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        content = payload["input"]["messages"][0]["content"]
        # 3 images + 1 text = 4 items
        assert len(content) == 4

    def test_too_many_references(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        provider = AlibabaImageGenProvider()
        result = provider.generate(
            prompt="too many",
            reference_image_urls=[f"https://example.com/img{i}.png" for i in range(10)],
        )
        assert result["success"] is False
        assert result["error_type"] == "too_many_references"

    def test_api_error(self):
        import requests as req_lib
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Access denied"
        mock_resp.json.return_value = {
            "code": "AccessDenied.Unpurchased",
            "message": "Access to model denied.",
        }
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(response=mock_resp)

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            provider = AlibabaImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "403" in result["error"]
        assert "Access to model denied" in result["error"]

    def test_timeout(self):
        import requests as req_lib
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        with patch(
            "plugins.image_gen.alibaba.requests.post",
            side_effect=req_lib.Timeout(),
        ):
            provider = AlibabaImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_connection_error(self):
        import requests as req_lib
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        with patch(
            "plugins.image_gen.alibaba.requests.post",
            side_effect=req_lib.ConnectionError("DNS failure"),
        ):
            provider = AlibabaImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "connection_error"

    def test_empty_choices(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"output": {"choices": []}}

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            provider = AlibabaImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_no_image_in_content(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "I cannot generate that."}],
                        }
                    }
                ]
            }
        }

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            provider = AlibabaImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_cached_locally(self):
        """Ephemeral OSS URLs must be materialised to local cache."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        oss_url = "https://dashscope-463f.oss-accelerate.aliyuncs.com/test.png?Expires=123"
        mock_resp = _mock_dashscope_response(image_url=oss_url)

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/dashscope_cached.png"),
            ) as mock_save:
                provider = AlibabaImageGenProvider()
                result = provider.generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == "/tmp/dashscope_cached.png"
        assert "oss-accelerate" not in result["image"]
        mock_save.assert_called_once()
        call_args = mock_save.call_args
        assert call_args[0][0] == oss_url

    def test_url_fallback_on_cache_failure(self):
        """If caching fails, return the bare URL rather than erroring."""
        import requests as req_lib
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        oss_url = "https://dashscope-463f.oss-accelerate.aliyuncs.com/test.png"
        mock_resp = _mock_dashscope_response(image_url=oss_url)

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                side_effect=req_lib.HTTPError("404"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == oss_url

    def test_auth_header(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/test.png"),
            ):
                provider = AlibabaImageGenProvider()
                provider.generate(prompt="test")

        call_args = mock_post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Bearer sk-test-key-12345" in headers["Authorization"]
        assert headers["X-DashScope-Async"] == "enable"

    def test_payload_structure(self):
        """Verify the wire payload matches the DashScope API contract."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/test.png"),
            ):
                provider = AlibabaImageGenProvider()
                provider.generate(prompt="A blue circle")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["model"] == "wan2.7-image-pro"
        assert "input" in payload
        assert "messages" in payload["input"]
        msg = payload["input"]["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        # Last content item should be the text prompt
        assert msg["content"][-1] == {"text": "A blue circle"}
        # Parameters should include size and n
        assert "parameters" in payload
        assert payload["parameters"]["n"] == 1
        assert "size" in payload["parameters"]

    def test_thinking_mode_enabled_for_pro_t2i(self):
        """thinking_mode defaults to True for wan2.7-image-pro text-to-image."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/test.png"),
            ):
                provider = AlibabaImageGenProvider()
                provider.generate(prompt="test")

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert payload["parameters"]["thinking_mode"] is True

    def test_thinking_mode_absent_for_edit(self):
        """thinking_mode must not be sent for image editing."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp) as mock_post:
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/test.png"),
            ):
                provider = AlibabaImageGenProvider()
                provider.generate(
                    prompt="edit this",
                    image_url="https://example.com/photo.png",
                )

        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "thinking_mode" not in payload["parameters"]

    def test_debug_info_in_extra(self):
        """Debug info from the API response is surfaced in the result."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = _mock_dashscope_response()

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/test.png"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(prompt="test")

        assert result["actual_seed"] == 42
        assert result["output_W"] == 1024
        assert result["output_H"] == 1024

    def test_response_without_type_field(self):
        """Handle responses where content items use {"image": url} without "type"."""
        from plugins.image_gen.alibaba import AlibabaImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"image": "https://oss.example.com/no-type.png"}
                            ],
                        }
                    }
                ]
            }
        }

        with patch("plugins.image_gen.alibaba.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.alibaba.save_url_image",
                return_value=Path("/tmp/no-type.png"),
            ):
                provider = AlibabaImageGenProvider()
                result = provider.generate(prompt="test")

        assert result["success"] is True
        assert result["image"] == "/tmp/no-type.png"


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.alibaba import AlibabaImageGenProvider, register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_image_gen_provider.assert_called_once()
        provider = mock_ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, AlibabaImageGenProvider)
        assert provider.name == "alibaba"
