"""Tests for the Cloudflare Workers AI image generation provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _fake_cf_creds(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")


class TestCloudflareImageGenProvider:
    def test_name_and_display(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        provider = CloudflareImageGenProvider()
        assert provider.name == "cloudflare"
        assert provider.display_name == "Cloudflare Workers AI"

    def test_is_available_with_creds(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        assert CloudflareImageGenProvider().is_available() is True

    def test_is_available_without_token(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        assert CloudflareImageGenProvider().is_available() is False

    def test_is_available_without_account(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        assert CloudflareImageGenProvider().is_available() is False

    def test_list_models_includes_current_ids(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        provider = CloudflareImageGenProvider()
        ids = {m["id"] for m in provider.list_models()}
        # The wrong id `flux-2-klein` (no size suffix) must never appear —
        # Cloudflare only hosts `flux-2-klein-9b` and `flux-2-klein-4b`.
        assert "@cf/black-forest-labs/flux-2-klein" not in ids
        assert "@cf/black-forest-labs/flux-1-schnell" in ids
        assert "@cf/black-forest-labs/flux-2-klein-9b" in ids
        assert "@cf/black-forest-labs/flux-2-klein-4b" in ids

    def test_default_model_is_schnell(self):
        from plugins.image_gen.cloudflare import (
            CloudflareImageGenProvider,
            DEFAULT_MODEL,
        )

        assert (
            CloudflareImageGenProvider().default_model()
            == "@cf/black-forest-labs/flux-1-schnell"
        )
        assert DEFAULT_MODEL == "@cf/black-forest-labs/flux-1-schnell"

    def test_get_setup_schema(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        schema = CloudflareImageGenProvider().get_setup_schema()
        assert schema["name"] == "Cloudflare Workers AI"
        keys = {v["key"] for v in schema["env_vars"]}
        assert keys == {"CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"}


class TestModelResolution:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv(
            "CLOUDFLARE_IMAGE_MODEL",
            "@cf/black-forest-labs/flux-2-klein-9b",
        )
        from plugins.image_gen.cloudflare import _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == "@cf/black-forest-labs/flux-2-klein-9b"

    def test_unknown_env_value_falls_through_to_default(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_IMAGE_MODEL", "@cf/does-not-exist")
        from plugins.image_gen.cloudflare import DEFAULT_MODEL, _resolve_model

        model_id, _ = _resolve_model()
        assert model_id == DEFAULT_MODEL


class TestGenerate:
    def test_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        result = CloudflareImageGenProvider().generate(prompt="test")
        assert result["success"] is False
        assert result["error_type"] == "missing_api_key"

    def test_json_image_response(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"result": {"image": "dGVzdC1pbWFnZS1kYXRh"}}

        with patch("plugins.image_gen.cloudflare.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.cloudflare.save_b64_image",
                return_value="/tmp/test.png",
            ):
                provider = CloudflareImageGenProvider()
                result = provider.generate(prompt="A cat playing piano")

        assert result["success"] is True
        assert result["image"] == "/tmp/test.png"
        assert result["provider"] == "cloudflare"
        assert result["resolution"] == "1280x720"

    def test_raw_png_response(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "image/png"}
        mock_resp.content = b"\x89PNG\r\n\x1a\nfake-binary"
        mock_resp.raise_for_status = MagicMock()

        with patch("plugins.image_gen.cloudflare.requests.post", return_value=mock_resp):
            with patch(
                "plugins.image_gen.cloudflare.save_b64_image",
                return_value="/tmp/test.png",
            ) as save:
                provider = CloudflareImageGenProvider()
                result = provider.generate(prompt="test")

        assert result["success"] is True
        # The raw bytes path should base64-encode the PNG before saving.
        saved_b64 = save.call_args.args[0]
        assert isinstance(saved_b64, str) and saved_b64

    def test_empty_response(self):
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"result": {}}
        mock_resp.content = b""

        with patch("plugins.image_gen.cloudflare.requests.post", return_value=mock_resp):
            provider = CloudflareImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_timeout(self):
        import requests as req_lib

        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        with patch(
            "plugins.image_gen.cloudflare.requests.post",
            side_effect=req_lib.Timeout(),
        ):
            provider = CloudflareImageGenProvider()
            result = provider.generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_http_error_surfaces_status(self):
        import requests as req_lib

        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        response = req_lib.Response()
        response.status_code = 401
        response._content = b'{"errors":[{"message":"invalid token"}]}'
        response.headers["Content-Type"] = "application/json"
        response.raise_for_status = MagicMock(
            side_effect=req_lib.HTTPError(response=response)
        )

        with patch(
            "plugins.image_gen.cloudflare.requests.post", return_value=response
        ):
            result = CloudflareImageGenProvider().generate(prompt="test")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]

    def test_schnell_default_steps(self):
        """FLUX schnell should auto-set num_steps=4 (it's a 4-step model)."""
        from plugins.image_gen.cloudflare import CloudflareImageGenProvider

        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            mock = MagicMock()
            mock.status_code = 200
            mock.headers = {"Content-Type": "application/json"}
            mock.raise_for_status = MagicMock()
            mock.json.return_value = {"result": {"image": "Zg=="}}
            return mock

        with patch("plugins.image_gen.cloudflare.requests.post", side_effect=fake_post):
            with patch(
                "plugins.image_gen.cloudflare.save_b64_image",
                return_value="/tmp/x.png",
            ):
                CloudflareImageGenProvider().generate(prompt="test")

        assert captured["json"]["num_steps"] == 4
