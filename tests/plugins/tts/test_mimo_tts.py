"""Tests for the bundled Xiaomi MiMo TTS plugin (issue #46257).

Covers the plugin-specific contracts the generic dispatcher tests
(``tests/tools/test_tts_plugin_dispatch.py``) cannot reach:

1. MiMo's chat/completions request shape — assistant text message,
   optional user style message, ``audio={"format": "wav", "voice": ...}``
2. base64 WAV extraction + format handling (wav written directly;
   other formats converted via ffmpeg; missing ffmpeg raises)
3. Error surfacing (missing API key, no choices, no audio.data)
4. One retry on transient (429/connection) failures
5. Bundled discovery end-to-end: plugin.yaml + register(ctx) wire the
   provider into ``agent.tts_registry``

No real network calls — the openai SDK client is faked.
"""

from __future__ import annotations

import base64
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import plugins.tts.mimo as mimo_plugin
from plugins.tts.mimo import MiMoTTSError, MiMoTTSProvider

_REPO_MIMO_PLUGIN_DIR = Path(__file__).resolve().parents[3] / "plugins" / "tts" / "mimo"

_WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt fake-mimo-audio"


def _b64_wav() -> str:
    return base64.b64encode(_WAV_BYTES).decode()


def _fake_response(audio_data: Optional[str] = "sentinel") -> SimpleNamespace:
    """Shape of an openai chat-completions response with audio output."""
    if audio_data == "sentinel":
        audio_data = _b64_wav()
    audio: Any = {"data": audio_data} if audio_data is not None else {}
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(audio=audio))]
    )


class _FakeOpenAI:
    """Stands in for ``openai.OpenAI`` inside the plugin."""

    captured: Dict[str, Any] = {}
    response: Any = None
    exc: Optional[BaseException] = None
    call_count: int = 0

    def __init__(self, api_key=None, base_url=None, timeout=None):
        _FakeOpenAI.captured["api_key"] = api_key
        _FakeOpenAI.captured["base_url"] = base_url
        _FakeOpenAI.captured["timeout"] = timeout
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        _FakeOpenAI.call_count += 1
        _FakeOpenAI.captured["kwargs"] = kwargs
        if _FakeOpenAI.exc is not None and _FakeOpenAI.call_count == 1:
            raise _FakeOpenAI.exc
        return _FakeOpenAI.response

    def close(self):
        _FakeOpenAI.captured["closed"] = True


@pytest.fixture(autouse=True)
def _isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "XIAOMI_API_KEY", "MIMO_API_KEY",
        "XIAOMI_BASE_URL", "MIMO_BASE_URL",
        "MIMO_TTS_STYLE", "MIMO_TTS_TIMEOUT",
    ):
        monkeypatch.delenv(key, raising=False)
    _FakeOpenAI.captured = {}
    _FakeOpenAI.response = _fake_response()
    _FakeOpenAI.exc = None
    _FakeOpenAI.call_count = 0
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    yield


@pytest.fixture
def _key(monkeypatch):
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_writes_wav_directly_with_defaults(self, _key, tmp_path):
        out = tmp_path / "speech.wav"
        result = MiMoTTSProvider().synthesize("hello", str(out), format="wav")

        assert result == str(out)
        assert out.read_bytes() == _WAV_BYTES
        kwargs = _FakeOpenAI.captured["kwargs"]
        assert kwargs["model"] == "mimo-v2.5-tts"
        assert kwargs["audio"] == {"format": "wav", "voice": "mimo_default"}
        # No style configured -> assistant message only (docs: user-role
        # content is never spoken, so omit it entirely when unused).
        assert kwargs["messages"] == [{"role": "assistant", "content": "hello"}]

    def test_voice_model_and_style_passthrough(self, _key, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMO_TTS_STYLE", "Warm and upbeat")
        out = tmp_path / "speech.wav"
        MiMoTTSProvider().synthesize(
            "hi there", str(out), voice="Chloe", model="mimo-v2.5-tts", format="wav"
        )
        kwargs = _FakeOpenAI.captured["kwargs"]
        assert kwargs["audio"]["voice"] == "Chloe"
        assert kwargs["messages"][0] == {"role": "user", "content": "Warm and upbeat"}
        assert kwargs["messages"][1] == {"role": "assistant", "content": "hi there"}

    def test_base_url_and_key_resolution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIMO_API_KEY", "fallback-key")
        monkeypatch.setenv("XIAOMI_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1/")
        out = tmp_path / "speech.wav"
        MiMoTTSProvider().synthesize("hi", str(out), format="wav")
        assert _FakeOpenAI.captured["api_key"] == "fallback-key"
        # Trailing slash stripped.
        assert _FakeOpenAI.captured["base_url"] == (
            "https://token-plan-sgp.xiaomimimo.com/v1"
        )


# ---------------------------------------------------------------------------
# Output format handling
# ---------------------------------------------------------------------------


class TestOutputFormats:
    def test_mp3_triggers_ffmpeg_conversion(self, _key, tmp_path, monkeypatch):
        calls: List[List[str]] = []

        monkeypatch.setattr(mimo_plugin.shutil, "which", lambda name: "/usr/bin/ffmpeg")

        def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            # Simulate ffmpeg writing the target file.
            Path(cmd[-1]).write_bytes(b"ID3fake-mp3")
            return SimpleNamespace(returncode=0, stderr="")

        monkeypatch.setattr(mimo_plugin.subprocess, "run", _fake_run)

        out = tmp_path / "speech.mp3"
        result = MiMoTTSProvider().synthesize("hello", str(out), format="mp3")
        assert result == str(out)
        assert out.read_bytes() == b"ID3fake-mp3"
        assert len(calls) == 1
        assert calls[0][0] == "/usr/bin/ffmpeg"
        assert calls[0][-1] == str(out)
        # Input is a temp WAV that must have been cleaned up.
        wav_input = calls[0][calls[0].index("-i") + 1]
        assert wav_input.endswith(".wav")
        assert not Path(wav_input).exists()

    def test_mp3_without_ffmpeg_raises(self, _key, tmp_path, monkeypatch):
        monkeypatch.setattr(mimo_plugin.shutil, "which", lambda name: None)
        with pytest.raises(MiMoTTSError, match="ffmpeg"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.mp3"), format="mp3")

    def test_ffmpeg_failure_raises_with_stderr(self, _key, tmp_path, monkeypatch):
        monkeypatch.setattr(mimo_plugin.shutil, "which", lambda name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(
            mimo_plugin.subprocess, "run",
            lambda cmd, **kw: SimpleNamespace(returncode=1, stderr="codec exploded"),
        )
        with pytest.raises(MiMoTTSError, match="codec exploded"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.ogg"), format="ogg")


# ---------------------------------------------------------------------------
# Error surfacing
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_api_key_raises_with_guidance(self, tmp_path):
        assert MiMoTTSProvider().is_available() is False
        with pytest.raises(MiMoTTSError, match="XIAOMI_API_KEY"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.wav"), format="wav")

    def test_empty_text_raises(self, _key, tmp_path):
        with pytest.raises(MiMoTTSError, match="empty"):
            MiMoTTSProvider().synthesize("   ", str(tmp_path / "o.wav"), format="wav")

    def test_no_choices_raises(self, _key, tmp_path):
        _FakeOpenAI.response = SimpleNamespace(choices=[])
        with pytest.raises(MiMoTTSError, match="no choices"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.wav"), format="wav")

    def test_missing_audio_data_raises(self, _key, tmp_path):
        _FakeOpenAI.response = _fake_response(audio_data=None)
        with pytest.raises(MiMoTTSError, match="audio.data"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.wav"), format="wav")

    def test_undecodable_audio_raises(self, _key, tmp_path):
        _FakeOpenAI.response = _fake_response(audio_data="!!!not-base64!!!")
        with pytest.raises(MiMoTTSError, match="undecodable"):
            MiMoTTSProvider().synthesize("hello", str(tmp_path / "o.wav"), format="wav")


# ---------------------------------------------------------------------------
# Transient retry
# ---------------------------------------------------------------------------


class TestTransientRetry:
    def test_429_retried_once_then_succeeds(self, _key, tmp_path, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        _FakeOpenAI.exc = type("RateLimit", (Exception,), {})(
            "429 rate limited"
        )
        _FakeOpenAI.exc.status_code = 429  # type: ignore[attr-defined]

        out = tmp_path / "speech.wav"
        MiMoTTSProvider().synthesize("hello", str(out), format="wav")
        assert _FakeOpenAI.call_count == 2
        assert out.read_bytes() == _WAV_BYTES

    def test_non_transient_error_not_retried(self, _key, tmp_path, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda seconds: None)
        _FakeOpenAI.exc = type("Auth", (Exception,), {})("401 invalid key")
        _FakeOpenAI.exc.status_code = 401  # type: ignore[attr-defined]

        with pytest.raises(MiMoTTSError, match="failed"):
            MiMoTTSProvider().synthesize(
                "hello", str(tmp_path / "o.wav"), format="wav"
            )
        assert _FakeOpenAI.call_count == 1


# ---------------------------------------------------------------------------
# Catalog surface
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_list_voices_covers_documented_presets(self):
        ids = {voice["id"] for voice in MiMoTTSProvider().list_voices()}
        assert {"mimo_default", "冰糖", "茉莉", "苏打", "白桦",
                "Mia", "Chloe", "Milo", "Dean"} <= ids

    def test_setup_schema_prompts_for_key(self):
        schema = MiMoTTSProvider().get_setup_schema()
        assert schema["env_vars"][0]["key"] == "XIAOMI_API_KEY"

    def test_voice_compatible_for_voice_bubbles(self):
        assert MiMoTTSProvider().voice_compatible is True


# ---------------------------------------------------------------------------
# Bundled discovery end-to-end
# ---------------------------------------------------------------------------


class TestBundledDiscovery:
    def test_plugin_registers_via_plugin_manager(self, tmp_path, monkeypatch):
        """Copy the shipped plugin into an isolated bundled dir, run real
        discovery, assert the provider lands in the TTS registry."""
        from agent import tts_registry
        from hermes_cli.plugins import PluginManager

        tts_registry._reset_for_tests()
        try:
            bundled = tmp_path / "bundled"
            (bundled / "tts").mkdir(parents=True)
            shutil.copytree(_REPO_MIMO_PLUGIN_DIR, bundled / "tts" / "mimo")
            monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))

            mgr = PluginManager()
            mgr.discover_and_load()

            loaded = mgr._plugins.get("tts/mimo")
            assert loaded is not None, "bundled mimo TTS plugin not discovered"
            assert loaded.enabled, f"plugin failed to load: {loaded.error}"
            provider = tts_registry.get_provider("mimo")
            assert provider is not None
            # The loader imports plugin modules under its own namespace
            # (hermes_plugins.*), so assert the behavioural contract
            # instead of class identity.
            assert type(provider).__name__ == "MiMoTTSProvider"
            assert provider.name == "mimo"
            assert provider.voice_compatible is True
        finally:
            tts_registry._reset_for_tests()
