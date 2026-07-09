import json
from pathlib import Path

import pytest

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
    monkeypatch.setattr(
        config,
        "minimax",
        {
            "api_key": "test-minimax-key",
            "base_url": "https://api.example.test",
            "t2a_model": "speech-2.8-hd",
            "voice_clone_model": "speech-2.8-hd",
            "music_model": "music-2.6-free",
        },
        raising=False,
    )
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


def test_clone_voice_uploads_files_and_writes_trial(monkeypatch, tmp_path):
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
            return FakeResponse({"base_resp": {"status_code": 0, "status_msg": "success"}, "trace_id": "clone-trace"})
        if url.endswith("/v1/t2a_v2"):
            return FakeResponse({"data": {"audio": b"trial".hex(), "status": 2}, "base_resp": {"status_code": 0}})
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(minimax.requests, "post", fake_post)
    output_dir = tmp_path / "voices" / "MiniMaxDemo001"

    result = minimax.clone_voice(
        voice_id="MiniMaxDemo001",
        clone_audio_file=str(clone_audio),
        prompt_audio_file=str(prompt_audio),
        prompt_text="提示音频文本",
        trial_text="试听激活文本",
        output_dir=str(output_dir),
    )

    assert purposes == ["voice_clone", "prompt_audio"]
    assert result["voice_id"] == "MiniMaxDemo001"
    assert result["voiceNameForVideo"] == "minimax:MiniMaxDemo001"
    assert Path(result["trialAudioFile"]).read_bytes() == b"trial"
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["voice_id"] == "MiniMaxDemo001"
    assert "test-minimax-key" not in json.dumps(metadata)


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
