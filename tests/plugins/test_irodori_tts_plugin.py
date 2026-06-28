from __future__ import annotations

import importlib
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


def test_irodori_plugin_registers_provider_tools_and_cli() -> None:
    plugin = importlib.import_module("plugins.irodori_tts")
    ctx = FakePluginContext()

    plugin.register(ctx)

    assert "irodori" in ctx.tts_providers
    assert ctx.tts_providers["irodori"].get_setup_schema()["name"] == "Irodori TTS"
    assert "irodori_tts_status" in ctx.tools
    assert "irodori_tts_synthesize" in ctx.tools
    assert "irodori-tts" in ctx.cli_commands


def test_synthesize_uses_windows_script(monkeypatch, tmp_path: Path) -> None:
    core = importlib.import_module("plugins.irodori_tts.core")
    invoke_script = tmp_path / "invoke.ps1"
    invoke_script.write_text("# test", encoding="utf-8")
    repo_dir = tmp_path / "irodori"
    repo_dir.mkdir()
    output_path = tmp_path / "sample.wav"
    captured = {}

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output_path.write_bytes(b"RIFFtest")
        return Completed()

    monkeypatch.setenv("IRODORI_TTS_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("IRODORI_TTS_INVOKE_SCRIPT", str(invoke_script))
    monkeypatch.setenv("IRODORI_TTS_BASE_URL", "http://127.0.0.1:8088")
    monkeypatch.setattr(core, "_load_tts_section", lambda: {})
    monkeypatch.setattr(core, "powershell_path", lambda: "powershell")
    monkeypatch.setattr(core.subprocess, "run", fake_run)

    result = core.synthesize_text("hello", output_path=output_path, output_format="wav")

    assert result["ok"] is True
    assert result["file_path"] == str(output_path)
    assert "-File" in captured["command"]
    assert str(invoke_script) in captured["command"]
    assert "-InputPath" in captured["command"]
    assert "-OutputPath" in captured["command"]
    assert captured["kwargs"]["cwd"] == str(repo_dir)
