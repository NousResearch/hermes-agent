"""Tests for the bundled Inworld backend plugin."""

from __future__ import annotations

import base64
from pathlib import Path


class _Response:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


class _Requests:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append({"url": url, **kwargs})
        if url.endswith("/tts/v1/voice"):
            return _Response(data={"audioContent": base64.b64encode(b"opus").decode()})
        return _Response(data={"transcription": {"transcript": "hello world"}})

    def get(self, url, **kwargs):
        self.gets.append({"url": url, **kwargs})
        return _Response(
            data={
                "voices": [
                    {
                        "voiceId": "Dennis",
                        "displayName": "Dennis",
                        "langCode": "en",
                    }
                ]
            }
        )


def test_inworld_tts_synthesizes_native_ogg(monkeypatch, tmp_path):
    import plugins.inworld as inworld

    requests = _Requests()
    monkeypatch.setenv("INWORLD_API_KEY", "basic-token")
    monkeypatch.setattr(inworld, "_require_requests", lambda: requests)
    monkeypatch.setattr(
        inworld,
        "_load_config_section",
        lambda section: {
            "inworld": {
                "model": "inworld-tts-2",
                "voice": "Dennis",
                "sample_rate": 48000,
                "delivery_mode": "BALANCED",
            }
        },
    )

    provider = inworld.InworldTTSProvider()
    out = provider.synthesize("Hello", str(tmp_path / "speech.mp3"))

    assert Path(out).suffix == ".ogg"
    assert Path(out).read_bytes() == b"opus"
    req = requests.posts[0]
    assert req["url"] == "https://api.inworld.ai/tts/v1/voice"
    assert req["headers"]["Authorization"] == "Basic basic-token"
    assert req["json"]["voiceId"] == "Dennis"
    assert req["json"]["audioConfig"]["audioEncoding"] == "OGG_OPUS"
    assert req["json"]["audioConfig"]["sampleRateHertz"] == 48000


def test_inworld_stt_posts_audio_data(monkeypatch, tmp_path):
    import plugins.inworld as inworld

    requests = _Requests()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"pcm-data")
    monkeypatch.setenv("INWORLD_API_KEY", "basic-token")
    monkeypatch.setattr(inworld, "_require_requests", lambda: requests)
    monkeypatch.setattr(
        inworld,
        "_load_config_section",
        lambda section: {
            "inworld": {
                "model": "inworld/inworld-stt-1",
                "audio_encoding": "AUTO_DETECT",
                "language": "en",
            }
        },
    )

    provider = inworld.InworldTranscriptionProvider()
    result = provider.transcribe(str(audio))

    assert result["success"] is True
    assert result["transcript"] == "hello world"
    req = requests.posts[0]
    assert req["url"] == "https://api.inworld.ai/stt/v1/transcribe"
    assert req["headers"]["Authorization"] == "Basic basic-token"
    assert req["json"]["transcribeConfig"]["modelId"] == "inworld/inworld-stt-1"
    assert req["json"]["transcribeConfig"]["audioEncoding"] == "AUTO_DETECT"
    assert req["json"]["audioData"]["content"] == base64.b64encode(b"pcm-data").decode()


def test_inworld_lists_voices(monkeypatch):
    import plugins.inworld as inworld

    requests = _Requests()
    monkeypatch.setenv("INWORLD_API_KEY", "basic-token")
    monkeypatch.setattr(inworld, "_require_requests", lambda: requests)
    monkeypatch.setattr(inworld, "_load_config_section", lambda section: {"inworld": {}})

    voices = inworld.InworldTTSProvider().list_voices()

    assert voices == [{"id": "Dennis", "display": "Dennis", "language": "en", "gender": None}]
    assert requests.gets[0]["headers"]["Authorization"] == "Basic basic-token"
