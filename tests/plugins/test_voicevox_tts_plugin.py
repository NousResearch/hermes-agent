from __future__ import annotations

import importlib
from pathlib import Path


class FakePluginContext:
    def __init__(self) -> None:
        self.tts_providers = {}

    def register_tts_provider(self, provider) -> None:
        self.tts_providers[provider.name] = provider


class FakeResponse:
    def __init__(self, payload=None, content: bytes = b"", text: str = "", status_code: int = 200):
        self._payload = payload
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_voicevox_plugin_registers_provider() -> None:
    plugin = importlib.import_module("plugins.voicevox_tts")
    ctx = FakePluginContext()

    plugin.register(ctx)

    assert "voicevox" in ctx.tts_providers
    assert ctx.tts_providers["voicevox"].display_name == "VOICEVOX"


def test_synthesize_writes_voicevox_wav(monkeypatch, tmp_path: Path) -> None:
    core = importlib.import_module("plugins.voicevox_tts.core")
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/audio_query"):
            return FakeResponse(payload={"speedScale": 1.0})
        if url.endswith("/synthesis"):
            return FakeResponse(content=b"RIFFvoicevox")
        raise AssertionError(url)

    monkeypatch.setenv("VOICEVOX_URL", "http://127.0.0.1:50021")
    monkeypatch.setattr(core.requests, "post", fake_post)

    output = tmp_path / "sample.mp3"
    result = core.synthesize_text("こんにちは", output_path=output, voice="8", speed=1.25)

    assert result["ok"] is True
    assert result["provider"] == "voicevox"
    assert result["file_path"] == str(tmp_path / "sample.wav")
    assert (tmp_path / "sample.wav").read_bytes() == b"RIFFvoicevox"
    assert calls[0][1]["params"] == {"speaker": 8, "text": "こんにちは"}
    assert calls[1][1]["json"]["speedScale"] == 1.25


def test_list_speakers_maps_voicevox_styles(monkeypatch) -> None:
    core = importlib.import_module("plugins.voicevox_tts.core")

    def fake_get(url, **kwargs):
        assert url.endswith("/speakers")
        return FakeResponse(
            payload=[
                {
                    "name": "四国めたん",
                    "styles": [{"id": 2, "name": "ノーマル"}],
                }
            ]
        )

    monkeypatch.setattr(core.requests, "get", fake_get)

    assert core.list_speakers() == [
        {"id": "2", "display": "四国めたん - ノーマル", "language": "ja"}
    ]
