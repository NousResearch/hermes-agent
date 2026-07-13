"""Tests for the Cloudflare Workers AI Aura TTS plugin.

Covers the ``TTSProvider`` ABC surface, credential validation, the
``synthesize()`` HTTP path, config resolution, and end-to-end dispatch
via ``text_to_speech_tool`` through the plugin registry.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Provider surface
# ---------------------------------------------------------------------------


class TestCloudflareProviderSurface:
    def test_name(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().name == "cloudflare"

    def test_display_name(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().display_name == "Cloudflare Workers AI"

    def test_default_model(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().default_model() == "@cf/deepgram/aura-2-en"

    def test_default_voice(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().default_voice() == "asteria"

    def test_list_voices_non_empty(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        voices = CloudflareTTSProvider().list_voices()
        assert len(voices) >= 5
        ids = {v["id"] for v in voices}
        assert "asteria" in ids
        for entry in voices:
            assert "id" in entry
            assert "language" in entry

    def test_list_models(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        models = CloudflareTTSProvider().list_models()
        assert len(models) == 1
        assert models[0]["id"] == "@cf/deepgram/aura-2-en"
        assert models[0]["max_text_length"] == 5000

    def test_setup_schema_advertises_credentials(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        schema = CloudflareTTSProvider().get_setup_schema()
        assert schema["name"] == "Cloudflare Workers AI"
        assert schema["badge"] == "free"
        env_keys = {entry["key"] for entry in schema.get("env_vars", [])}
        assert "CLOUDFLARE_API_TOKEN" in env_keys
        assert "CLOUDFLARE_ACCOUNT_ID" in env_keys

    def test_voice_compatible_is_true(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().voice_compatible is True

    def test_name_not_in_builtin_set(self):
        """The plugin name must not collide with a built-in provider."""
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS

        assert "cloudflare" not in BUILTIN_TTS_PROVIDERS


class TestCloudflareProviderAvailability:
    def test_available_with_credentials(self, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        assert CloudflareTTSProvider().is_available() is True

    def test_unavailable_without_token(self, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        assert CloudflareTTSProvider().is_available() is False

    def test_unavailable_without_account_id(self, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        assert CloudflareTTSProvider().is_available() is False

    def test_unavailable_without_anything(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        assert CloudflareTTSProvider().is_available() is False


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------


class TestCloudflareSynthesize:
    def test_missing_credentials_raises_value_error(self, tmp_path):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        with pytest.raises(
            ValueError, match="CLOUDFLARE_API_TOKEN.*CLOUDFLARE_ACCOUNT_ID"
        ):
            CloudflareTTSProvider().synthesize("Hello", str(tmp_path / "out.mp3"))

    def test_posts_to_aura_endpoint_and_writes_audio(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.content = b"mp3-bytes"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        output_path = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=response) as mock_post:
            result = CloudflareTTSProvider().synthesize("Hello", output_path)

        assert result == output_path
        assert (tmp_path / "out.mp3").read_bytes() == b"mp3-bytes"
        endpoint = mock_post.call_args[0][0]
        assert endpoint == (
            "https://api.cloudflare.com/client/v4/accounts/"
            "acct-123/ai/run/@cf/deepgram/aura-2-en"
        )
        kwargs = mock_post.call_args[1]
        assert kwargs["headers"]["Authorization"] == "Bearer cf-token"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["json"] == {
            "text": "Hello",
            "speaker": "asteria",
            "encoding": "mp3",
        }

    def test_dispatcher_kwargs_override_defaults(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.content = b"audio"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=response) as mock_post:
            CloudflareTTSProvider().synthesize(
                "Hi",
                str(tmp_path / "out.mp3"),
                voice="zeus",
                model="@cf/deepgram/aura-2-en",
                format="mp3",
            )

        assert mock_post.call_args[1]["json"]["speaker"] == "zeus"

    def test_config_section_overrides_defaults(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.content = b"audio"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        fake_config = {
            "tts": {
                "cloudflare": {
                    "model": "@cf/deepgram/aura-2-en",
                    "voice": "luna",
                    "encoding": "linear16",
                    "container": "wav",
                    "sample_rate": 24000,
                    "bit_rate": 128000,
                    "base_url": "https://cf.example/client/v4/accounts",
                }
            }
        }
        with (
            patch(
                "plugins.tts.cloudflare._load_cf_config",
                return_value=fake_config["tts"]["cloudflare"],
            ),
            patch("requests.post", return_value=response) as mock_post,
        ):
            CloudflareTTSProvider().synthesize("Hi", str(tmp_path / "out.wav"))

        assert mock_post.call_args[0][0] == (
            "https://cf.example/client/v4/accounts/"
            "acct-123/ai/run/@cf/deepgram/aura-2-en"
        )
        assert mock_post.call_args[1]["json"] == {
            "text": "Hi",
            "speaker": "luna",
            "encoding": "linear16",
            "container": "wav",
            "sample_rate": 24000,
            "bit_rate": 128000,
        }

    def test_dispatcher_kwargs_override_config_section(self, tmp_path, monkeypatch):
        """Dispatcher voice/model kwargs win over the tts.cloudflare section."""
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.content = b"audio"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        cf_config = {
            "voice": "luna",
            "model": "@cf/deepgram/aura-2-en",
            "encoding": "mp3",
        }
        with (
            patch("plugins.tts.cloudflare._load_cf_config", return_value=cf_config),
            patch("requests.post", return_value=response) as mock_post,
        ):
            CloudflareTTSProvider().synthesize(
                "Hi",
                str(tmp_path / "out.mp3"),
                voice="zeus",
            )

        assert mock_post.call_args[1]["json"]["speaker"] == "zeus"

    def test_api_error_raises_runtime_error(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.status_code = 401
        response.json.return_value = {"errors": [{"message": "Invalid API token"}]}

        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="HTTP 401.*Invalid API token"):
                CloudflareTTSProvider().synthesize("Hello", str(tmp_path / "out.mp3"))

    def test_empty_audio_raises_runtime_error(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.status_code = 200
        response.content = b""
        response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="empty audio data"):
                CloudflareTTSProvider().synthesize("Hello", str(tmp_path / "out.mp3"))

    def test_format_wav_maps_to_linear16(self, tmp_path, monkeypatch):
        from plugins.tts.cloudflare import CloudflareTTSProvider

        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
        response = MagicMock()
        response.content = b"audio"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        with patch("requests.post", return_value=response) as mock_post:
            CloudflareTTSProvider().synthesize(
                "Hi",
                str(tmp_path / "out.wav"),
                format="wav",
            )

        assert mock_post.call_args[1]["json"]["encoding"] == "linear16"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class TestCloudflareRegistration:
    def test_register_calls_register_tts_provider(self):
        from plugins.tts.cloudflare import CloudflareTTSProvider, register

        ctx = MagicMock()
        register(ctx)

        ctx.register_tts_provider.assert_called_once()
        provider = ctx.register_tts_provider.call_args[0][0]
        assert isinstance(provider, CloudflareTTSProvider)
        assert provider.name == "cloudflare"

    def test_registry_accepts_cloudflare_name(self):
        """The registry must accept 'cloudflare' (not a built-in name)."""
        from agent.tts_registry import (
            _BUILTIN_NAMES,
            register_provider,
            _reset_for_tests,
        )
        from plugins.tts.cloudflare import CloudflareTTSProvider

        _reset_for_tests()
        assert "cloudflare" not in _BUILTIN_NAMES
        register_provider(CloudflareTTSProvider())
        from agent.tts_registry import get_provider

        assert get_provider("cloudflare") is not None
        _reset_for_tests()


# ---------------------------------------------------------------------------
# End-to-end dispatch via text_to_speech_tool
# ---------------------------------------------------------------------------


class TestCloudflareDispatch:
    def test_text_to_speech_routes_to_plugin(self, tmp_path, monkeypatch):
        """text_to_speech_tool dispatches to the plugin when provider=cloudflare."""
        from agent import tts_registry
        from plugins.tts.cloudflare import CloudflareTTSProvider

        tts_registry._reset_for_tests()
        tts_registry.register_provider(CloudflareTTSProvider())
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")

        response = MagicMock()
        response.content = b"audio-bytes"
        response.status_code = 200
        response.raise_for_status = MagicMock()

        from tools import tts_tool

        monkeypatch.setattr(
            tts_tool, "_load_tts_config", lambda: {"provider": "cloudflare"}
        )
        try:
            with patch("requests.post", return_value=response) as mock_post:
                result = json.loads(
                    tts_tool.text_to_speech_tool("Hello", str(tmp_path / "out.mp3"))
                )
            assert result["success"] is True
            assert result["provider"] == "cloudflare"
            assert mock_post.call_args[1]["json"]["text"] == "Hello"
        finally:
            tts_registry._reset_for_tests()

    def test_check_tts_requirements_satisfied_by_plugin(self, monkeypatch):
        """check_tts_requirements returns True when the plugin is available."""
        from agent import tts_registry
        from plugins.tts.cloudflare import CloudflareTTSProvider

        tts_registry._reset_for_tests()
        tts_registry.register_provider(CloudflareTTSProvider())
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")

        from tools import tts_tool

        monkeypatch.setattr(
            tts_tool, "_has_any_command_tts_provider", lambda *a, **k: False
        )
        try:
            with (
                patch("tools.tts_tool._import_edge_tts", side_effect=ImportError),
                patch("tools.tts_tool._import_elevenlabs", side_effect=ImportError),
                patch("tools.tts_tool._import_openai_client", side_effect=ImportError),
                patch("tools.tts_tool._import_mistral_client", side_effect=ImportError),
                patch("tools.tts_tool._check_neutts_available", return_value=False),
                patch("tools.tts_tool._check_kittentts_available", return_value=False),
                patch("tools.tts_tool._check_piper_available", return_value=False),
            ):
                assert tts_tool.check_tts_requirements() is True
        finally:
            tts_registry._reset_for_tests()
