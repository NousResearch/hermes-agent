"""Tests for hermes doctor helpers."""

import importlib.util
import sys
import types
from pathlib import Path

from hermes_cli.doctor import _get_stt_runtime_status, _has_provider_env_config

MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "transcription_tools.py"
SPEC = importlib.util.spec_from_file_location("test_transcription_tools_module", MODULE_PATH)
tt = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tt)


def _install_fake_tools_package(monkeypatch):
    tools_pkg = types.ModuleType("tools")
    tools_pkg.transcription_tools = tt
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.transcription_tools", tt)


class TestProviderEnvDetection:
    def test_detects_openai_api_key(self):
        content = "OPENAI_BASE_URL=http://localhost:1234/v1\nOPENAI_API_KEY=sk-test-key\n"
        assert _has_provider_env_config(content)

    def test_detects_custom_endpoint_without_openrouter_key(self):
        content = "OPENAI_BASE_URL=http://localhost:8080/v1\n"
        assert _has_provider_env_config(content)

    def test_returns_false_when_no_provider_settings(self):
        content = "TERMINAL_ENV=local\n"
        assert not _has_provider_env_config(content)


class TestSttRuntimeStatus:
    def test_reports_disabled_state(self, monkeypatch):
        _install_fake_tools_package(monkeypatch)
        monkeypatch.setattr(tt, "resolve_stt_config", lambda: {"enabled": False, "provider": "whispercpp", "whispercpp": {}})
        status = _get_stt_runtime_status()
        assert status["enabled"] is False

    def test_reports_whispercpp_prerequisites(self, tmp_path, monkeypatch):
        _install_fake_tools_package(monkeypatch)
        model_path = tmp_path / "model.bin"
        model_path.write_bytes(b"model")
        monkeypatch.setattr(
            tt,
            "resolve_stt_config",
            lambda: {
                "enabled": True,
                "provider": "whispercpp",
                "whispercpp": {
                    "binary_path": "/opt/whisper-cli",
                    "model_path": str(model_path),
                    "ffmpeg_path": "/usr/bin/ffmpeg",
                },
            },
        )
        monkeypatch.setattr(tt, "resolve_whispercpp_binary", lambda config=None: "/opt/whisper-cli")
        monkeypatch.setattr(tt, "resolve_ffmpeg_binary", lambda config=None: "/usr/bin/ffmpeg")

        status = _get_stt_runtime_status()

        assert status["enabled"] is True
        assert status["provider"] == "whispercpp"
        assert status["binary_found"] is True
        assert status["model_found"] is True
        assert status["ffmpeg_found"] is True
