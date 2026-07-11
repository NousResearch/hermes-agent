import json
from pathlib import Path

import pytest
import toml

from app.config import config
from app.services import minimax


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", status_code=200):
        self._payload = payload or {}
        self.content = content
        self.status_code = status_code
        self.text = json.dumps(self._payload, ensure_ascii=False)
        self.headers = {"Content-Type": "application/json"}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _configure_minimax(monkeypatch, tmp_path):
    minimax_config = {
        "api_key": "test-minimax-key",
        "base_url": "https://api.example.test",
        "t2a_model": "speech-2.8-hd",
        "voice_clone_model": "speech-2.8-hd",
        "music_model": "music-2.6-free",
    }
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml.dumps({"app": {}, "minimax": minimax_config}), encoding="utf-8")
    monkeypatch.setattr(config, "config_file", str(config_file), raising=False)
    monkeypatch.setattr(config, "minimax", minimax_config, raising=False)
    monkeypatch.setattr(minimax.utils, "song_dir", lambda sub_dir="": str((tmp_path / "songs" / sub_dir).resolve()))
    monkeypatch.setattr(minimax.utils, "storage_dir", lambda sub_dir="", create=False: str((tmp_path / "storage" / sub_dir).resolve()))


def test_validate_voice_id_accepts_minimax_rules():
    assert minimax.validate_voice_id("MiniMaxDemo_001") == "MiniMaxDemo_001"

    with pytest.raises(ValueError, match="8-256"):
        minimax.validate_voice_id("short")
    with pytest.raises(ValueError, match="首字符"):
        minimax.validate_voice_id("1InvalidVoice")
    with pytest.raises(ValueError, match="末位"):
        minimax.validate_voice_id("InvalidVoice_")


def test_t2a_sync_decodes_hex_audio(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(
            {
                "data": {"audio": b"fake-mp3".hex(), "status": 2},
                "base_resp": {"status_code": 0, "status_msg": "success"},
                "trace_id": "trace-1",
            }
        )

    monkeypatch.setattr(minimax.requests, "post", fake_post)
    output = tmp_path / "voice.mp3"

    result = minimax.t2a_sync("你好，Hermes", "MiniMaxDemo001", str(output), speed=1.2, vol=0.8)

    assert output.read_bytes() == b"fake-mp3"
    assert result["file"] == str(output)
    assert result["voice_id"] == "MiniMaxDemo001"
    assert calls[0]["url"] == "https://api.example.test/v1/t2a_v2"
    assert calls[0]["json"]["voice_setting"]["voice_id"] == "MiniMaxDemo001"
    assert calls[0]["json"]["voice_setting"]["speed"] == 1.2
    assert calls[0]["headers"]["Authorization"] == "Bearer test-minimax-key"


def test_list_voices_returns_categorized_provider_records(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    monkeypatch.setattr(
        minimax.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(
            {
                "system_voice": [
                    {"voice_id": "Korean_GentleBoss", "voice_name": "Gentle Boss"}
                ],
                "voice_cloning": [{"voice_id": "MyClone001"}],
                "voice_generation": [],
                "base_resp": {"status_code": 0},
            }
        ),
    )

    result = minimax.list_voices("all")

    assert result["voices"][0] == {
        "category": "system",
        "id": "Korean_GentleBoss",
        "name": "Gentle Boss",
        "providerConfirmed": True,
    }
    assert result["voices"][1]["category"] == "voice_cloning"


def test_audio_url_download_does_not_forward_api_key(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers, "timeout": timeout})
        return FakeResponse(content=b"remote-audio")

    monkeypatch.setattr(minimax.requests, "get", fake_get)
    output = tmp_path / "remote.mp3"

    minimax._write_audio_payload(
        {"data": {"audio_url": "https://cdn.example.test/audio.mp3"}},
        str(output),
    )

    assert output.read_bytes() == b"remote-audio"
    assert calls == [
        {
            "url": "https://cdn.example.test/audio.mp3",
            "headers": None,
            "timeout": 60,
        }
    ]


def test_clone_voice_uses_inline_preview_without_tts_activation(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    clone_audio = tmp_path / "clone.wav"
    prompt_audio = tmp_path / "prompt.mp3"
    clone_audio.write_bytes(b"clone-audio")
    prompt_audio.write_bytes(b"prompt-audio")
    purposes = []

    def fake_post(url, data=None, files=None, json=None, headers=None, timeout=None):
        if url.endswith("/v1/files/upload"):
            purpose = data["purpose"]
            purposes.append(purpose)
            return FakeResponse({"file": {"file_id": 111 if purpose == "voice_clone" else 222}, "base_resp": {"status_code": 0}})
        if url.endswith("/v1/voice_clone"):
            assert json["file_id"] == 111
            assert json["voice_id"] == "MiniMaxDemo001"
            assert json["clone_prompt"] == {"prompt_audio": 222, "prompt_text": "提示音频文本"}
            assert json["text"] == "这是低成本克隆试听。"
            return FakeResponse(
                {
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                    "demo_audio": "https://cdn.example.test/clone-preview.mp3",
                    "trace_id": "clone-trace",
                }
            )
        if url.endswith("/v1/t2a_v2"):
            raise AssertionError("clone preview must not activate the voice through ordinary TTS")
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(minimax.requests, "post", fake_post)
    monkeypatch.setattr(
        minimax.requests,
        "get",
        lambda url, timeout=None: FakeResponse(content=b"clone-preview"),
    )
    output_dir = tmp_path / "voices" / "MiniMaxDemo001"

    result = minimax.clone_voice(
        voice_id="MiniMaxDemo001",
        clone_audio_file=str(clone_audio),
        prompt_audio_file=str(prompt_audio),
        prompt_text="提示音频文本",
        trial_text="这是低成本克隆试听。",
        output_dir=str(output_dir),
    )

    assert purposes == ["voice_clone", "prompt_audio"]
    assert result["voice_id"] == "MiniMaxDemo001"
    assert result["voiceNameForVideo"] == "minimax:MiniMaxDemo001"
    assert result["activated"] is False
    assert Path(result["trialAudioFile"]).read_bytes() == b"clone-preview"
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["voice_id"] == "MiniMaxDemo001"
    assert "test-minimax-key" not in json.dumps(metadata)


def test_clone_voice_reports_partial_success_when_inline_preview_download_fails(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    clone_audio = tmp_path / "clone.wav"
    clone_audio.write_bytes(b"clone-audio")

    def fake_upload(path, purpose):
        return 111

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse(
            {
                "base_resp": {"status_code": 0},
                "demo_audio": "https://cdn.example.test/unavailable.mp3",
                "trace_id": "clone-trace",
            }
        )

    def fake_download(*args, **kwargs):
        raise RuntimeError("preview unavailable")

    monkeypatch.setattr(minimax, "upload_file", fake_upload)
    monkeypatch.setattr(minimax.requests, "post", fake_post)
    monkeypatch.setattr(minimax, "_download_public_audio", fake_download, raising=False)
    output_dir = tmp_path / "voices" / "MiniMaxDemo001"

    result = minimax.clone_voice(
        voice_id="MiniMaxDemo001",
        clone_audio_file=str(clone_audio),
        trial_text="试听激活文本",
        output_dir=str(output_dir),
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert result == metadata
    assert metadata["activated"] is False
    assert metadata["previewError"] == "preview unavailable"


def test_clone_voice_requires_prompt_audio_and_text_together(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    clone_audio = tmp_path / "clone.wav"
    prompt_audio = tmp_path / "prompt.wav"
    clone_audio.write_bytes(b"clone-audio")
    prompt_audio.write_bytes(b"prompt-audio")
    monkeypatch.setattr(minimax, "upload_file", lambda path, purpose: 111)

    with pytest.raises(ValueError, match="prompt audio requires prompt_text"):
        minimax.clone_voice(
            voice_id="MiniMaxDemo001",
            clone_audio_file=str(clone_audio),
            prompt_audio_file=str(prompt_audio),
            output_dir=str(tmp_path / "voices"),
        )


def test_configured_models_are_used_when_request_omits_model(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)
    Path(config.config_file).write_text(
        toml.dumps({"minimax": {**config.minimax, "t2a_model": "speech-custom-config"}}),
        encoding="utf-8",
    )
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return FakeResponse({"data": {"audio": b"voice".hex()}, "base_resp": {"status_code": 0}})

    monkeypatch.setattr(minimax.requests, "post", fake_post)

    minimax.t2a_sync("Configured model", "MiniMaxDemo001", str(tmp_path / "configured.mp3"))

    assert calls[0]["model"] == "speech-custom-config"


def test_minimax_config_reloads_credentials_without_sidecar_restart(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    first = minimax.get_minimax_config()
    Path(config.config_file).write_text(
        toml.dumps(
            {
                "minimax": {
                    **config.minimax,
                    "api_key": "updated-key",
                    "base_url": "https://updated.example.test",
                }
            }
        ),
        encoding="utf-8",
    )
    second = minimax.get_minimax_config()

    assert first["api_key"] == "test-minimax-key"
    assert second["api_key"] == "updated-key"
    assert second["base_url"] == "https://updated.example.test"


def test_generate_music_saves_hex_audio_as_bgm(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    def fake_post(url, json=None, headers=None, timeout=None):
        assert url == "https://api.example.test/v1/music_generation"
        assert json["model"] == "music-2.6-free"
        assert json["is_instrumental"] is True
        return FakeResponse({"data": {"audio": b"music-bytes".hex(), "status": 2}, "base_resp": {"status_code": 0}})

    monkeypatch.setattr(minimax.requests, "post", fake_post)

    result = minimax.generate_music(prompt="科技感短视频开场", is_instrumental=True, save_as_bgm=True, filename_slug="tech opener")

    assert result["bgm"]["file"].startswith("minimax-music-tech-opener-")
    assert Path(result["file"]).read_bytes() == b"music-bytes"
    assert Path(result["file"]).parent == tmp_path / "songs"


def test_generate_music_returns_custom_audio_metadata_when_not_bgm(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse({"data": {"audio": b"music-bytes".hex(), "status": 2}, "base_resp": {"status_code": 0}})

    monkeypatch.setattr(minimax.requests, "post", fake_post)

    result = minimax.generate_music(
        prompt="科技感短视频开场",
        is_instrumental=True,
        save_as_bgm=False,
    )

    assert result["audio"]["file"].startswith("storage/custom_audio/")
    assert result["audio"]["name"].endswith(".mp3")


def test_generate_music_uses_unique_output_names_for_concurrent_requests(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    def fake_post(url, json=None, headers=None, timeout=None):
        return FakeResponse({"data": {"audio": b"music-bytes".hex(), "status": 2}, "base_resp": {"status_code": 0}})

    monkeypatch.setattr(minimax.requests, "post", fake_post)
    ids = iter(["firstrequest0001", "secondrequest002"])
    monkeypatch.setattr(minimax.utils, "get_uuid", lambda remove_hyphen=True: next(ids))

    first = minimax.generate_music(prompt="same prompt", is_instrumental=True, save_as_bgm=True)
    second = minimax.generate_music(prompt="same prompt", is_instrumental=True, save_as_bgm=True)

    assert first["file"] != second["file"]


def test_generate_music_requires_lyrics_or_optimizer_for_vocals(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="lyrics or lyrics_optimizer"):
        minimax.generate_music(
            prompt="melancholic pop",
            is_instrumental=False,
            lyrics="",
            lyrics_optimizer=False,
        )


def test_generate_lyrics_preserves_top_level_response_fields(monkeypatch, tmp_path):
    _configure_minimax(monkeypatch, tmp_path)

    def fake_post(url, json=None, headers=None, timeout=None):
        assert json["mode"] == "write_full_song"
        return FakeResponse(
            {
                "song_title": "Hermes Summer",
                "style_tags": "Pop, Upbeat",
                "lyrics": "[Verse]\nHello Hermes",
                "trace_id": "lyrics-trace",
                "base_resp": {"status_code": 0},
            }
        )

    monkeypatch.setattr(minimax.requests, "post", fake_post)

    result = minimax.generate_lyrics(prompt="A summer song")

    assert result == {
        "lyrics": "[Verse]\nHello Hermes",
        "song_title": "Hermes Summer",
        "style_tags": "Pop, Upbeat",
        "trace_id": "lyrics-trace",
    }

    with pytest.raises(ValueError, match="mode"):
        minimax.generate_lyrics(mode="unknown", prompt="A summer song")


def test_minimax_routes_are_registered():
    from app.asgi import app

    paths = {route.path for route in app.routes}

    assert {
        "/api/v1/minimax/lyrics",
        "/api/v1/minimax/music",
        "/api/v1/minimax/tts",
        "/api/v1/minimax/voices",
        "/api/v1/minimax/voices/clone",
    }.issubset(paths)
