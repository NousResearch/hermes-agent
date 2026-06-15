from __future__ import annotations

import importlib
from pathlib import Path

from plugins.unsloth_studio import core, register


class _FakeContext:
    def __init__(self) -> None:
        self.tools = {}
        self.commands = {}
        self.cli_commands = {}

    def register_tool(self, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def test_registers_tools_slash_and_cli_command() -> None:
    ctx = _FakeContext()
    register(ctx)

    assert "unsloth_studio_status" in ctx.tools
    assert "unsloth_studio_start" in ctx.tools
    assert "unsloth_studio_stop" in ctx.tools
    assert "unsloth_studio_install_info" in ctx.tools
    assert "unsloth-studio" in ctx.commands
    assert "unsloth-studio" in ctx.cli_commands


def test_status_reports_missing_cli(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(core, "_unsloth_exe", lambda: None)
    monkeypatch.setattr(core, "state_file", lambda: tmp_path / "state.json")

    payload = core.status_payload({"probe_url": False})

    assert payload["ok"] is False
    assert payload["available"] is False
    assert "not found" in payload["notes"][0]


def test_start_rejects_public_host_without_confirmation(monkeypatch) -> None:
    monkeypatch.setattr(core, "_unsloth_exe", lambda: "unsloth")

    result = core.start_studio({"host": "0.0.0.0", "port": 8888})

    assert result["ok"] is False
    assert result["confirmation_required"] is True


def test_start_builds_detached_command(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeProc:
        pid = 4242

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProc()

    monkeypatch.setattr(core, "_unsloth_exe", lambda: "unsloth")
    monkeypatch.setattr(core, "state_file", lambda: tmp_path / "state.json")
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(core, "_url_ready", lambda _url, _wait: True)
    monkeypatch.setattr(core.subprocess, "Popen", fake_popen)

    result = core.start_studio(
        {
            "host": "127.0.0.1",
            "port": 8899,
            "wait_seconds": 1,
            "extra_args": ["--example"],
        }
    )

    assert result["ok"] is True
    assert result["pid"] == 4242
    command = calls[0][0]
    assert command == ["unsloth", "studio", "-H", "127.0.0.1", "-p", "8899", "--example"]
    state = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "4242" in state


def test_stop_uses_recorded_pid(monkeypatch, tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    state.write_text('{"pid": 3131, "url": "http://127.0.0.1:8888"}', encoding="utf-8")
    seen = {}

    def fake_terminate(pid):
        seen["pid"] = pid
        return {"ok": True}

    monkeypatch.setattr(core, "state_file", lambda: state)
    monkeypatch.setattr(core, "_terminate_pid", fake_terminate)

    result = core.stop_studio({})

    assert result["ok"] is True
    assert seen["pid"] == 3131
    assert not state.exists()


def test_install_info_has_platform_command() -> None:
    payload = core.install_info({"local_only": True})

    assert payload["ok"] is True
    assert "unsloth" in payload["install"].lower()
    assert "colab_notebook" not in payload


def test_module_importable() -> None:
    assert importlib.import_module("plugins.unsloth_studio.core") is core
