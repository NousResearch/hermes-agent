"""Tests for the bundled Google AI Studio (Gemini / Nano Banana) image_gen plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.gemini as gemini_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_http_response(payload: Dict[str, Any], *, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    if status_code >= 400:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError(f"{status_code} Error", response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _gemini_payload(*, b64: str | None = None, mime: str = "image/png", text: str = "Generated image",
                    usage: Dict[str, Any] | None = None) -> Dict[str, Any]:
    parts = [{"text": text}]
    if b64 is not None:
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    body: Dict[str, Any] = {"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}]}
    if usage is not None:
        body["usageMetadata"] = usage
    return body


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for var in (
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_IMAGE_MODEL", "GEMINI_BASE_URL",
        "GEMINI_IMAGE_ASPECT_RATIO", "GEMINI_IMAGE_SIZE", "GEMINI_IMAGE_GOOGLE_SEARCH",
    ):
        monkeypatch.delenv(var, raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-key")
    return gemini_plugin.GeminiImageGenProvider()


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "gemini"

    def test_default_model(self, provider):
        assert provider.default_model() == "gemini-3.1-flash-image"

    def test_picker_matches_resolvable_catalog(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert set(ids) == set(provider.models)
        assert set(ids) == {"gemini-3.1-flash-image", "gemini-3.1-flash-lite-image", "gemini-3-pro-image"}
        assert provider.default_model() in ids

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert "Nano Banana" in entry["display"]
            assert entry["speed"]
            assert entry["strengths"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert gemini_plugin.GeminiImageGenProvider().is_available() is False

    @pytest.mark.parametrize("env_var", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
    def test_api_key_set_available(self, monkeypatch, env_var):
        monkeypatch.setenv(env_var, "AIza-test")
        assert gemini_plugin.GeminiImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:
    def test_env_var_override_and_google_prefix_stripping(self, monkeypatch):
        monkeypatch.setenv("GEMINI_IMAGE_MODEL", "google/gemini-3-pro-image")
        model_id, meta = gemini_plugin._resolve_model()
        assert model_id == "gemini-3-pro-image"
        assert meta["api_model"] == "gemini-3-pro-image"

    def test_config_gemini_model_and_foreign_top_level_ignored(self, tmp_path):
        import yaml

        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-medium", "gemini": {"model": "gemini-3.1-flash-lite-image"}}})
        )
        model_id, _ = gemini_plugin._resolve_model()
        assert model_id == "gemini-3.1-flash-lite-image"

        # When only a foreign top-level model is set, fallback to DEFAULT_MODEL
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"model": "gpt-image-2-medium"}}))
        model_id, _ = gemini_plugin._resolve_model()
        assert model_id == "gemini-3.1-flash-image"


# ── Endpoint / credential routing ───────────────────────────────────────────


class TestEndpointConfig:
    def test_config_base_url_and_key_env_reach_availability_and_endpoint(self, monkeypatch, tmp_path):
        import yaml

        monkeypatch.setenv("CUSTOM_GEMINI_TOKEN", "custom-secret")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"image_gen": {"gemini": {
            "base_url": "https://proxy.example.com/v1beta/", "key_env": "CUSTOM_GEMINI_TOKEN"}}}))
        assert gemini_plugin.GeminiImageGenProvider().is_available() is True
        assert gemini_plugin._resolve_endpoint() == ("https://proxy.example.com/v1beta", "custom-secret")


# ── Generate ────────────────────────────────────────────────────────────────


class TestSourceImageLoading:
    def test_load_image_bytes_blocks_credential_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            gemini_plugin._load_image_bytes(str(auth_json))

    def test_load_image_bytes_allows_legit_local_image(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(bytes.fromhex(_PNG_HEX))

        data, mime = gemini_plugin._load_image_bytes(str(img))
        assert data == bytes.fromhex(_PNG_HEX)
        assert mime == "image/png"


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("   ", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = gemini_plugin.GeminiImageGenProvider().generate("a yellow banana")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache_and_sends_x_goog_api_key_header(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_resp = _fake_http_response(_gemini_payload(b64=_b64_png(), mime="image/jpeg"))

        with patch("requests.post", return_value=fake_resp) as mock_post:
            result = provider.generate("a cute nano banana", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gemini-3.1-flash-image"
        assert result["aspect_ratio"] == "landscape"
        assert result["exact_aspect_ratio"] == "16:9"
        assert result["provider"] == "gemini"
        assert result["modality"] == "text"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.suffix == ".jpg"
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        called_url = mock_post.call_args.args[0]
        assert called_url == "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent"
        assert "key=" not in called_url
        assert mock_post.call_args.kwargs["headers"]["x-goog-api-key"] == "AIza-test-key"
        assert mock_post.call_args.kwargs["json"]["generationConfig"]["imageConfig"] == {"aspectRatio": "16:9"}

    @pytest.mark.parametrize("has_image", [True, False])
    def test_token_usage_reaches_session_accounting(self, provider, has_image):
        """Gemini bills per token: ``usageMetadata`` lands as an ``image_generation`` row
        even when a billed HTTP 200 carries only a text refusal and no image."""
        from agent import aux_accounting

        recorded = []

        class _DB:
            def record_auxiliary_usage(self, *args, **kwargs):
                recorded.append((args, kwargs))

        payload = _gemini_payload(
            b64=_b64_png() if has_image else None,
            text="Here is your image" if has_image else "I cannot generate that image",
            usage={"promptTokenCount": 42, "candidatesTokenCount": 1290, "totalTokenCount": 1332},
        )
        token = aux_accounting.set_accounting_context(_DB(), "sess-gemini-1")
        try:
            with patch("requests.post", return_value=_fake_http_response(payload)):
                result = provider.generate("a yellow banana", aspect_ratio="landscape")
        finally:
            aux_accounting.reset_accounting_context(token)

        assert result["success"] is has_image
        if not has_image:
            assert result["error_type"] == "empty_response"
            assert "I cannot generate that image" in result["error"]
        ((session_id, task), kwargs), = recorded
        assert (session_id, task) == ("sess-gemini-1", "image_generation")
        assert (kwargs["model"], kwargs["billing_provider"]) == ("gemini-3.1-flash-image", "gemini")
        assert (kwargs["input_tokens"], kwargs["output_tokens"]) == (42, 1290)

    def test_env_and_config_exact_aspect_ratio_override_default_semantic(self, provider, monkeypatch, tmp_path):
        import yaml

        monkeypatch.setenv("GEMINI_IMAGE_ASPECT_RATIO", "21:9")
        monkeypatch.setenv("GEMINI_IMAGE_SIZE", "2K")
        monkeypatch.setenv("GEMINI_IMAGE_GOOGLE_SEARCH", "true")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"gemini": {"aspect_ratio": "4:5"}}})
        )

        with patch("requests.post", return_value=_fake_http_response(_gemini_payload(b64=_b64_png()))) as mock_post:
            result = provider.generate("ultrawide banner", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["exact_aspect_ratio"] == "21:9"
        assert result["image_size"] == "2K"
        assert result["google_search"] is True
        sent = mock_post.call_args.kwargs["json"]
        assert sent["generationConfig"]["imageConfig"] == {"aspectRatio": "21:9", "imageSize": "2K"}
        assert sent["tools"] == [{"googleSearch": {}}]

    def test_reference_images_sent_before_text_prompt(self, provider, tmp_path):
        ref_path = tmp_path / "ref.png"
        ref_path.write_bytes(bytes.fromhex(_PNG_HEX))

        with patch("requests.post", return_value=_fake_http_response(_gemini_payload(b64=_b64_png()))) as mock_post:
            result = provider.generate("make it cyberpunk", image_url=str(ref_path))

        assert result["success"] is True
        assert result["modality"] == "image"
        parts = mock_post.call_args.kwargs["json"]["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0]["inlineData"]["mimeType"] == "image/png"
        assert parts[0]["inlineData"]["data"] == _b64_png()
        assert parts[1] == {"text": "make it cyberpunk"}

    def test_http_error_surfaces_api_message(self, provider):
        err_resp = _fake_http_response({"error": {"message": "API key not valid"}}, status_code=403)
        with patch("requests.post", return_value=err_resp):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "API key not valid" in result["error"]


# ── Plugin registration ─────────────────────────────────────────────────────


class TestPluginRegistration:
    def test_plugin_registers_in_image_gen_registry(self):
        from agent.image_gen_registry import get_provider
        from hermes_cli.plugins import get_plugin_manager

        get_plugin_manager().discover_and_load()
        registered = get_provider("gemini")
        assert registered is not None
        assert registered.name == "gemini"
