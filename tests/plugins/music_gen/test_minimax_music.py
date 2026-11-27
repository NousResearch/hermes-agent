"""Tests for the MiniMax music generation plugin (plugins/music_gen/minimax).

Covers the tool-availability ladder (BYOK key / managed Nous gateway /
neither), the /v1/music_generation payload contract (lyrics verbatim,
lyrics_optimizer without an empty lyrics field, instrumental mode),
base_resp application-error mapping, and url-vs-hex audio materialization
into HERMES_HOME. No live network — every seam is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate credentials and HERMES_HOME for every test.

    The plugin materializes audio under ``get_hermes_home()/cache/music`` —
    point HERMES_HOME at a per-test tmpdir so nothing can land in the real
    Hermes home, and start from a no-credential baseline.
    """
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_GATEWAY_URL", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _fake_gateway(origin: str = "https://minimax-gateway.nousresearch.com"):
    gw = MagicMock()
    gw.gateway_origin = origin
    gw.nous_user_token = "nous-portal-token"
    return gw


def _ok_response(audio: str, status: int = 2, extra: dict | None = None) -> dict:
    return {
        "data": {"audio": audio, "status": status},
        "extra_info": extra or {"music_duration": 185000, "music_sample_rate": 44100},
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "trace_id": "trace-123",
    }


def _err_response(code: int, msg: str) -> dict:
    return {
        "data": None,
        "base_resp": {"status_code": code, "status_msg": msg},
        "trace_id": "trace-err",
    }


def _urlopen_returning(payload: bytes):
    """Context-manager mock matching ``with urllib.request.urlopen(...) as r``."""
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = payload
    cm.__exit__.return_value = False
    return MagicMock(return_value=cm)


# ---------------------------------------------------------------------------
# Availability ladder (check_fn)
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_api_key(self, monkeypatch):
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_API_KEY", "mk-test-123")
        with patch.object(minimax, "_resolve_gateway", return_value=None):
            assert minimax._check_music_tools() is True

    def test_available_with_managed_gateway_only(self):
        from plugins.music_gen import minimax

        with patch(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            return_value=_fake_gateway(),
        ):
            assert minimax._check_music_tools() is True

    def test_hidden_without_any_credential(self):
        from plugins.music_gen import minimax

        with patch(
            "tools.managed_tool_gateway.resolve_managed_tool_gateway",
            return_value=None,
        ):
            assert minimax._check_music_tools() is False

    def test_register_wires_tool_into_music_gen_toolset(self):
        from plugins.music_gen import minimax

        ctx = MagicMock()
        minimax.register(ctx)

        ctx.register_tool.assert_called_once()
        kwargs = ctx.register_tool.call_args.kwargs
        assert kwargs["name"] == "generate_music"
        assert kwargs["toolset"] == "music_gen"
        assert kwargs["check_fn"] is minimax._check_music_tools
        assert kwargs["requires_env"] == ["MINIMAX_API_KEY"]
        assert kwargs["schema"]["name"] == "generate_music"


class TestAuthTokenLadder:
    def test_gateway_preferred_when_route_reachable(self, monkeypatch):
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_API_KEY", "mk-byok")
        with patch.object(minimax, "_resolve_gateway", return_value=_fake_gateway()), \
             patch.object(minimax, "_gateway_host_reachable", return_value=True):
            auth = minimax._auth_token()
        assert auth["managed"] is True
        assert auth["token"] == "nous-portal-token"
        assert auth["base"] == "https://minimax-gateway.nousresearch.com"

    def test_dead_gateway_route_falls_back_to_byok_key(self, monkeypatch):
        """Until tool-gateway onboards MiniMax the vendor host is NXDOMAIN —
        a signed-in Nous user with a working key must not be steered at it."""
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_API_KEY", "mk-byok")
        with patch.object(minimax, "_resolve_gateway", return_value=_fake_gateway()), \
             patch.object(minimax, "_gateway_host_reachable", return_value=False):
            auth = minimax._auth_token()
        assert auth["managed"] is False
        assert auth["token"] == "mk-byok"
        assert auth["base"] == minimax.API_BASE_DIRECT

    def test_no_credential_yields_empty_auth(self):
        from plugins.music_gen import minimax

        with patch.object(minimax, "_resolve_gateway", return_value=None):
            auth = minimax._auth_token()
        assert auth["token"] is None and auth["base"] is None

    def test_gateway_url_override_uses_nous_token(self, monkeypatch):
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_GATEWAY_URL", "https://staging.example/")
        with patch.object(minimax, "_resolve_gateway", return_value=_fake_gateway()):
            auth = minimax._auth_token()
        assert auth["managed"] is True
        assert auth["base"] == "https://staging.example"

    def test_post_composes_versionless_base_with_v1_path(self, monkeypatch):
        """Full URL = {base}{path}: direct base has no /v1 suffix, the request
        path carries it — the shape the managed gateway allowlists."""
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_API_KEY", "mk-byok")
        ok_body = json.dumps(_ok_response("cafe")).encode("utf-8")
        with patch.object(minimax, "_resolve_gateway", return_value=None), \
             patch(
                 "plugins.music_gen.minimax.urllib.request.urlopen",
                 _urlopen_returning(ok_body),
             ) as fake_urlopen:
            minimax._post("/v1/music_generation", {"model": "music-3.0"})

        req = fake_urlopen.call_args.args[0]
        assert req.full_url == "https://api.minimax.io/v1/music_generation"
        assert req.get_header("Authorization") == "Bearer mk-byok"


# ---------------------------------------------------------------------------
# Payload contract
# ---------------------------------------------------------------------------


class TestPayloadShape:
    def _generate_capturing_payload(self, minimax, **kwargs):
        captured = {}

        def fake_post(path, payload):
            captured["path"] = path
            captured["payload"] = payload
            return _ok_response("cafe")  # valid hex so _save_audio succeeds

        with patch.object(minimax, "_post", side_effect=fake_post):
            result = minimax.generate_music(**kwargs)
        return captured, result

    def test_explicit_lyrics_sent_verbatim(self):
        from plugins.music_gen import minimax

        lyrics = "[Verse]\nneon rain on the interstate\n[Chorus]\nrun the lights"
        captured, _ = self._generate_capturing_payload(
            minimax, prompt="synthwave, 100 BPM", lyrics=lyrics
        )
        payload = captured["payload"]
        assert payload["lyrics"] == lyrics
        assert "lyrics_optimizer" not in payload
        assert payload["model"] == "music-3.0"
        # /v1 lives in the request path, not the base — the managed gateway
        # origin has no /v1 prefix and allowlists POST /v1/music_generation.
        assert captured["path"] == "/v1/music_generation"

    def test_absent_lyrics_uses_optimizer_without_empty_field(self):
        """MiniMax 2013s when lyrics_optimizer:true is sent alongside an
        explicit empty lyrics string — the field must be ABSENT."""
        from plugins.music_gen import minimax

        captured, _ = self._generate_capturing_payload(minimax, prompt="lofi hip hop")
        payload = captured["payload"]
        assert payload["lyrics_optimizer"] is True
        assert "lyrics" not in payload

    def test_whitespace_lyrics_treated_as_absent(self):
        from plugins.music_gen import minimax

        captured, _ = self._generate_capturing_payload(
            minimax, prompt="lofi", lyrics="   \n  "
        )
        payload = captured["payload"]
        assert payload["lyrics_optimizer"] is True
        assert "lyrics" not in payload

    def test_instrumental_mode_sets_flag_and_no_lyrics(self):
        from plugins.music_gen import minimax

        captured, _ = self._generate_capturing_payload(
            minimax, prompt="cinematic strings", lyrics="ignored", mode="instrumental"
        )
        payload = captured["payload"]
        assert payload["is_instrumental"] is True
        assert "lyrics" not in payload
        assert "lyrics_optimizer" not in payload

    def test_cover_mode_requires_reference_audio(self):
        from plugins.music_gen import minimax

        with patch.object(minimax, "_post") as post:
            result = minimax.generate_music(prompt="bossa nova", mode="cover")
        post.assert_not_called()
        assert "reference_audio_url" in result["error"]

    def test_cover_mode_switches_model_and_carries_audio_url(self):
        from plugins.music_gen import minimax

        captured, _ = self._generate_capturing_payload(
            minimax,
            prompt="bossa nova",
            mode="cover",
            reference_audio_url="https://example.com/ref.mp3",
        )
        payload = captured["payload"]
        assert payload["model"] == "music-cover"
        assert payload["audio_url"] == "https://example.com/ref.mp3"

    def test_empty_prompt_rejected_before_network(self):
        from plugins.music_gen import minimax

        with patch.object(minimax, "_post") as post:
            result = minimax.generate_music(prompt="   ")
        post.assert_not_called()
        assert "prompt is required" in result["error"]


# ---------------------------------------------------------------------------
# base_resp application-error mapping
# ---------------------------------------------------------------------------


class TestBaseRespErrors:
    def test_insufficient_balance_1008_maps_hint(self):
        from plugins.music_gen import minimax

        with patch.object(
            minimax, "_post", return_value=_err_response(1008, "insufficient balance")
        ):
            result = minimax.generate_music(prompt="jazz")
        assert "MiniMax error 1008" in result["error"]
        assert "top up" in result["error"]
        assert result["trace_id"] == "trace-err"

    def test_moderation_1026_maps_hint(self):
        from plugins.music_gen import minimax

        with patch.object(
            minimax, "_post", return_value=_err_response(1026, "content violation")
        ):
            result = minimax.generate_music(prompt="jazz")
        assert "MiniMax error 1026" in result["error"]
        assert "moderation" in result["error"]

    def test_unknown_code_still_surfaces_msg(self):
        from plugins.music_gen import minimax

        with patch.object(
            minimax, "_post", return_value=_err_response(9999, "mystery")
        ):
            result = minimax.generate_music(prompt="jazz")
        assert "MiniMax error 9999: mystery" in result["error"]

    def test_status_zero_is_success(self):
        from plugins.music_gen import minimax

        assert minimax._check_base_resp({"base_resp": {"status_code": 0}}) is None
        assert minimax._check_base_resp({}) is None


# ---------------------------------------------------------------------------
# Audio materialization (url + hex) under HERMES_HOME
# ---------------------------------------------------------------------------


class TestAudioMaterialization:
    def test_url_audio_downloaded_into_hermes_home(self, tmp_path):
        from plugins.music_gen import minimax

        fake_mp3 = b"ID3\x04fake-mp3-bytes"
        with patch(
            "plugins.music_gen.minimax.urllib.request.urlopen",
            _urlopen_returning(fake_mp3),
        ):
            saved = minimax._save_audio(
                _ok_response("https://cdn.minimax.io/song.mp3"), "test song", "mp3"
            )

        out = Path(saved["file"])
        assert out.read_bytes() == fake_mp3
        # Profile safety: everything lands under the mocked HERMES_HOME.
        assert out.is_relative_to(tmp_path)
        assert out.parent == tmp_path / "cache" / "music"
        assert saved["audio_url"] == "https://cdn.minimax.io/song.mp3"
        assert saved["duration_s"] == 185.0

    def test_hex_audio_decoded_into_hermes_home(self, tmp_path):
        from plugins.music_gen import minimax

        hex_audio = b"RIFFfake-wav".hex()
        saved = minimax._save_audio(_ok_response(hex_audio), "hex song", "wav")

        out = Path(saved["file"])
        assert out.read_bytes() == b"RIFFfake-wav"
        assert out.is_relative_to(tmp_path)
        assert out.suffix == ".wav"

    def test_missing_audio_payload_is_error(self):
        from plugins.music_gen import minimax

        saved = minimax._save_audio(
            {"data": {"status": 2}, "extra_info": {}}, "empty", "mp3"
        )
        assert "no audio payload" in saved["error"]

    def test_handler_marks_media_for_gateway_delivery(self, tmp_path, monkeypatch):
        from plugins.music_gen import minimax

        monkeypatch.setenv("MINIMAX_API_KEY", "mk-test")
        hex_audio = b"mp3-bytes".hex()
        with patch.object(minimax, "_resolve_gateway", return_value=None), \
             patch.object(minimax, "_post", return_value=_ok_response(hex_audio)):
            out = minimax._generate_music_handler({"prompt": "chiptune"})

        result = json.loads(out)
        assert result["MEDIA"] == result["file"]
        assert Path(result["file"]).is_relative_to(tmp_path)
        assert result["mode"] == "song"

    def test_handler_error_has_no_media_key(self):
        from plugins.music_gen import minimax

        with patch.object(minimax, "_resolve_gateway", return_value=None):
            out = minimax._generate_music_handler({"prompt": "chiptune"})
        result = json.loads(out)
        assert "MEDIA" not in result
        assert "No MiniMax credential" in result["error"]


# ---------------------------------------------------------------------------
# Manifest sanity
# ---------------------------------------------------------------------------


class TestManifest:
    def test_plugin_yaml_parses_with_backend_kind(self):
        import yaml

        manifest_path = (
            Path(__file__).resolve().parents[3]
            / "plugins" / "music_gen" / "minimax" / "plugin.yaml"
        )
        data = yaml.safe_load(manifest_path.read_text())
        assert data["name"] == "minimax-music"
        assert data["kind"] == "backend"
        assert data["author"] == "NousResearch"
        assert data["requires_env"] == ["MINIMAX_API_KEY"]
