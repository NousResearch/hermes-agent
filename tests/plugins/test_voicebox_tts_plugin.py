from __future__ import annotations

import importlib
import json
from pathlib import Path


class FakePluginContext:
    def __init__(self) -> None:
        self.tts_providers = {}
        self.tools = {}
        self.cli_commands = {}

    def register_tts_provider(self, provider) -> None:
        self.tts_providers[provider.name] = provider

    def register_tool(self, name, handler, **kwargs) -> None:
        self.tools[name] = {"handler": handler, "kwargs": kwargs}

    def register_cli_command(self, name, help, setup_fn, handler_fn=None, description="") -> None:
        self.cli_commands[name] = {
            "help": help,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "description": description,
        }


class FakeResponse:
    def __init__(
        self,
        payload=None,
        content: bytes = b"",
        text: str = "",
        status_code: int = 200,
    ):
        self._payload = payload
        if payload is not None and not content:
            content = json.dumps(payload).encode("utf-8")
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        del chunk_size
        if self.content:
            yield self.content


def test_voicebox_plugin_registers_provider_tools_and_cli() -> None:
    plugin = importlib.import_module("plugins.voicebox")
    ctx = FakePluginContext()

    plugin.register(ctx)

    assert "voicebox" in ctx.tts_providers
    assert ctx.tts_providers["voicebox"].display_name == "Voicebox"
    assert "voicebox_status" in ctx.tools
    assert "voicebox_synthesize" in ctx.tools
    assert "voicebox_transcribe" in ctx.tools
    assert "voicebox" in ctx.cli_commands


def test_synthesize_polls_and_downloads_audio(monkeypatch, tmp_path: Path) -> None:
    core = importlib.import_module("plugins.voicebox.core")
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "POST" and url.endswith("/speak"):
            return FakeResponse(payload={"id": "gen-1", "status": "generating"})
        if method == "GET" and url.endswith("/history/gen-1"):
            return FakeResponse(payload={"id": "gen-1", "status": "completed", "engine": "qwen"})
        raise AssertionError((method, url))

    def fake_get(url, **kwargs):
        calls.append(("GET", url, kwargs))
        if url.endswith("/audio/gen-1"):
            return FakeResponse(content=b"RIFFvoicebox")
        raise AssertionError(url)

    monkeypatch.setattr(core, "settings", lambda tts_config=None: core.VoiceboxSettings(
        base_url="http://127.0.0.1:17493",
        client_id="hermes",
        profile="hakua",
        language="ja",
        engine="",
        timeout=30.0,
        personality=False,
        poll_interval=0.25,
        irodori_ref_audio=str(tmp_path / "missing.ogg"),
        auto_import_profile=False,
        reference_text="ref",
    ))
    monkeypatch.setattr(core, "list_profiles", lambda tts_config=None: [{"id": "p1", "name": "hakua"}])
    monkeypatch.setattr(core.requests, "request", fake_request)
    monkeypatch.setattr(core.requests, "get", fake_get)

    output = tmp_path / "sample.wav"
    result = core.synthesize_text("こんにちは", output_path=output, voice="hakua")

    assert result["ok"] is True
    assert result["provider"] == "voicebox"
    assert result["file_path"] == str(output)
    assert output.read_bytes() == b"RIFFvoicebox"
    assert calls[0][0] == "POST"
    assert calls[0][1].endswith("/speak")
    assert calls[0][2]["json"]["text"] == "こんにちは"
    assert calls[0][2]["json"]["profile"] == "hakua"


def test_ensure_hakua_profile_creates_profile_from_ref_audio(monkeypatch, tmp_path: Path) -> None:
    core = importlib.import_module("plugins.voicebox.core")
    ref_audio = tmp_path / "hakua.ogg"
    ref_audio.write_bytes(b"OggS")
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if method == "GET" and url.endswith("/profiles"):
            return FakeResponse(payload=[])
        if method == "POST" and url.endswith("/profiles"):
            return FakeResponse(payload={"id": "profile-1", "name": "hakua"})
        raise AssertionError((method, url))

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        assert "/profiles/profile-1/samples" in url
        return FakeResponse(payload={"ok": True})

    cfg = core.VoiceboxSettings(
        base_url="http://127.0.0.1:17493",
        client_id="hermes",
        profile="hakua",
        language="ja",
        engine="",
        timeout=30.0,
        personality=False,
        poll_interval=0.25,
        irodori_ref_audio=str(ref_audio),
        auto_import_profile=True,
        reference_text="はくあ",
    )
    monkeypatch.setattr(core, "settings", lambda tts_config=None: cfg)
    monkeypatch.setattr(core.requests, "request", fake_request)
    monkeypatch.setattr(core.requests, "post", fake_post)

    result = core.ensure_hakua_profile()

    assert result["ok"] is True
    assert result["created"] is True
    assert result["profile_id"] == "profile-1"
    assert any("/profiles/profile-1/samples" in call[1] for call in calls)


def test_status_handler_returns_json(monkeypatch) -> None:
    plugin = importlib.import_module("plugins.voicebox")

    payload = {
        "ok": True,
        "provider": "voicebox",
        "available": True,
        "server": {"reachable": True},
        "profiles": {"count": 1},
        "defaults": {},
        "paths": {},
    }
    monkeypatch.setattr(plugin, "status_payload", lambda tts_config=None: payload)

    raw = plugin._status_handler()
    parsed = json.loads(raw)
    assert parsed["provider"] == "voicebox"
    assert parsed["available"] is True
