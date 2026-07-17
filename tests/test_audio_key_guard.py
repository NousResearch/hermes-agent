"""Shared audio credential guard: a real cloud key must never travel to a
private/self-hosted base_url. Covers the helper directly plus each provider
site that was previously unguarded (xai/elevenlabs/minimax/gemini/deepinfra)."""

from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.audio_key_guard import (
    base_url_is_private,
    resolve_provider_key,
    PLACEHOLDER_KEY,
)


class TestBaseUrlIsPrivate:
    @pytest.mark.parametrize("url,expected", [
        ("http://192.168.1.50:8000/v1", True),
        ("http://example.com/v1", True),           # any cleartext http
        ("https://10.0.0.5/v1", True),
        ("https://[::1]:8443/v1", True),
        ("https://localhost:8443/v1", True),
        ("https://169.254.169.254/latest", True),  # link-local metadata
        ("https://api.openai.com/v1", False),
        ("https://api.x.ai/v1", False),
        ("https://generativelanguage.googleapis.com", False),
        ("", False),
    ])
    def test_classification(self, url, expected):
        assert base_url_is_private(url) is expected


class TestResolveProviderKey:
    def test_private_target_drops_env_cloud_key(self):
        assert resolve_provider_key("", "sk-real-cloud", "http://192.168.1.5/v1") == PLACEHOLDER_KEY

    def test_private_target_honours_config_key(self):
        assert resolve_provider_key("local-token", "sk-real-cloud", "http://192.168.1.5/v1") == "local-token"

    def test_public_target_keeps_env_key(self):
        assert resolve_provider_key("", "sk-real-cloud", "https://api.x.ai/v1") == "sk-real-cloud"

    def test_public_target_config_key_wins(self):
        assert resolve_provider_key("cfg", "sk-env", "https://api.x.ai/v1") == "cfg"


class TestProviderSitesGuarded:
    """Each provider that computes a config base_url must route its key
    through the guard — a private base_url override yields the placeholder."""

    def test_xai_stt_guarded(self, tmp_path):
        import tools.transcription_tools as tt
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF0000WAVE")
        captured = {}

        def _fake_post(url, headers=None, **kw):
            captured["auth"] = (headers or {}).get("Authorization", "")
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"text": "hi"}
            return resp

        import tools.xai_http as xai_http
        with patch.object(xai_http, "resolve_xai_http_credentials",
                          return_value={"api_key": "xai-real-cloud"}), \
             patch.object(tt, "_load_stt_config",
                          return_value={"provider": "xai",
                                        "xai": {"base_url": "http://192.168.1.9:9000/v1"}}), \
             patch("requests.post", _fake_post):
            tt._transcribe_xai(str(wav), "grok")
        assert captured["auth"] == f"Bearer {PLACEHOLDER_KEY}"

    def test_elevenlabs_stt_guarded(self, tmp_path):
        import tools.transcription_tools as tt
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF0000WAVE")
        captured = {}

        def _fake_post(url, headers=None, **kw):
            captured["key"] = (headers or {}).get("xi-api-key", "")
            resp = MagicMock(status_code=200)
            resp.json.return_value = {"text": "hi"}
            return resp

        with patch.object(tt, "get_env_value",
                          side_effect=lambda k, *a: "el-real-cloud" if k == "ELEVENLABS_API_KEY" else ""), \
             patch.object(tt, "_load_stt_config",
                          return_value={"provider": "elevenlabs",
                                        "elevenlabs": {"base_url": "http://192.168.1.9:9000"}}), \
             patch("requests.post", _fake_post):
            tt._transcribe_elevenlabs(str(wav), "scribe_v1")
        assert captured["key"] == PLACEHOLDER_KEY

    def test_deepinfra_stt_guarded(self, tmp_path):
        import tools.transcription_tools as tt
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFF0000WAVE")
        captured = {}

        def _fake_transcribe_openai(fp, model, **kw):
            captured["api_key"] = kw.get("api_key")
            return {"success": True, "transcript": "hi", "provider": "deepinfra"}

        with patch.object(tt, "get_env_value",
                          side_effect=lambda k, *a: "di-real-cloud" if k == "DEEPINFRA_API_KEY" else ""), \
             patch.object(tt, "_load_stt_config",
                          return_value={"provider": "deepinfra",
                                        "deepinfra": {"base_url": "http://192.168.1.9:9000/v1"}}), \
             patch.object(tt, "_transcribe_openai", _fake_transcribe_openai):
            tt._transcribe_deepinfra(str(wav), "some-model")
        assert captured["api_key"] == PLACEHOLDER_KEY
