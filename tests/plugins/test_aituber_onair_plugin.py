"""Tests for the AITuber OnAir Hermes plugin bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from plugins.aituber_onair import core, register


class _FakeContext:
    def __init__(self) -> None:
        self.tools = []
        self.commands = {}
        self.cli_commands = {}

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "aituber-onair"
    (repo / "packages" / "chat" / "examples" / "codex-character-chat").mkdir(
        parents=True
    )
    (repo / "packages" / "chat" / "dist" / "cjs").mkdir(parents=True)
    (repo / "packages" / "core" / "examples" / "react-fbx-app").mkdir(
        parents=True
    )
    (repo / "package.json").write_text('{"name":"aituber-onair"}', encoding="utf-8")
    (repo / "packages" / "chat" / "examples" / "codex-character-chat" / "index.js").write_text(
        "console.log('ok')\n", encoding="utf-8"
    )
    (repo / "packages" / "chat" / "dist" / "cjs" / "agent.js").write_text(
        "exports.createAgentChatService = () => ({})\n", encoding="utf-8"
    )
    (repo / "packages" / "core" / "examples" / "react-fbx-app" / "package.json").write_text(
        '{"name":"react-fbx-app"}', encoding="utf-8"
    )
    return repo


def test_registers_tools_slash_and_cli_command():
    ctx = _FakeContext()
    register(ctx)

    names = {tool["name"] for tool in ctx.tools}
    assert "aituber_onair_status" in names
    assert "aituber_onair_tts_status" in names
    assert "aituber_onair_speak" in names
    assert "aituber_onair_say" in names
    assert all(tool["toolset"] == "aituber-onair" for tool in ctx.tools)
    assert "aituber" in ctx.commands
    assert "aituber-onair" in ctx.cli_commands


def test_resolve_repo_root_accepts_aituber_checkout(tmp_path):
    repo = _fake_repo(tmp_path)

    assert core.resolve_repo_root(str(repo)) == repo


def test_tts_status_prefers_installed_irodori(monkeypatch):
    monkeypatch.setattr(
        core,
        "_irodori_status",
        lambda: {
            "provider": "irodori",
            "available": True,
            "usable": False,
            "server": {"ok": False},
        },
    )
    monkeypatch.setattr(
        core,
        "_voicevox_engine_status",
        lambda: {
            "provider": "voicevox",
            "installed": True,
            "reachable": False,
        },
    )
    monkeypatch.setattr(core, "_read_json_file", lambda _path: {})

    result = core.tts_status()

    assert result["ok"] is True
    assert result["selected_provider"] == "irodori"
    assert result["ready"] is False


def test_start_tts_falls_back_to_voicevox(monkeypatch):
    seen = {}

    monkeypatch.setattr(core, "_irodori_status", lambda: {"available": False})
    monkeypatch.setattr(
        core,
        "_voicevox_engine_status",
        lambda: {"installed": True, "reachable": False},
    )

    def fake_start(values):
        seen.update(values)
        return {"ok": True, "provider": "voicevox", "ready": True}

    monkeypatch.setattr(core, "_start_voicevox_tts", fake_start)

    result = core.start_tts({"provider": "auto", "timeout_seconds": 12})

    assert result["ok"] is True
    assert result["provider"] == "voicevox"
    assert seen["timeout_seconds"] == 12


def test_speak_dispatches_selected_tts_provider(monkeypatch):
    seen = {}

    monkeypatch.setattr(core, "_select_tts_provider", lambda _explicit=None: "irodori")

    def fake_synthesize(values):
        seen.update(values)
        return {"ok": True, "provider": "irodori", "file_path": "voice.wav"}

    monkeypatch.setattr(core, "_synthesize_irodori", fake_synthesize)

    result = core.synthesize_speech({"text": "hello", "provider": "auto"})

    assert result["ok"] is True
    assert result["provider"] == "irodori"
    assert seen["text"] == "hello"


def test_irodori_speech_uses_configured_hakua_voice(monkeypatch, tmp_path):
    from plugins.irodori_tts import core as irodori_core

    seen = {}

    monkeypatch.setattr(core, "_plugin_config", lambda: {"tts_voice": "hakua", "tts_speed": 1.06})
    monkeypatch.setattr(
        irodori_core,
        "synthesize_text",
        lambda **kwargs: seen.update(kwargs)
        or {"ok": True, "provider": "irodori", "file_path": str(kwargs["output_path"])},
    )

    result = core._synthesize_irodori({"text": "hello", "output_path": str(tmp_path / "voice.wav")})

    assert result["ok"] is True
    assert seen["voice"] == "hakua"
    assert seen["speed"] == 1.06


def test_run_hakua_once_dispatches_codex_character_cli(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    calls = []

    monkeypatch.setattr(core, "_node_exe", lambda: "node")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": True, "installed": True},
    )
    monkeypatch.setattr(
        core,
        "_codex_cli_auth_status",
        lambda: {"has_access_token": True},
    )
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")
    monkeypatch.setattr(core, "_plugin_working_directory", lambda _repo: str(repo))

    def fake_run(cmd, cwd, env, timeout_seconds):
        calls.append((cmd, cwd, env, timeout_seconds))
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "=== Codex Character Chat ===\nはくあ> [happy] こんにちは\n",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.run_hakua_once({"repo_root": str(repo), "prompt": "挨拶して"})

    assert result["ok"] is True
    assert result["reply"] == "[happy] こんにちは"
    cmd, cwd, env, timeout_seconds = calls[0]
    assert cmd[0] == "node"
    assert "index.js" in cmd[1]
    assert "--once=挨拶して" in cmd
    assert env["CODEX_CHARACTER_NAME"] == "はくあ"
    assert env["CODEX_CHARACTER_SYSTEM_PROMPT"] == "Hakua prompt"
    assert cwd == repo
    assert timeout_seconds == core.DEFAULT_TIMEOUT_SECONDS


def test_run_hakua_once_can_synthesize_reply(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)

    monkeypatch.setattr(core, "_node_exe", lambda: "node")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": True, "installed": True},
    )
    monkeypatch.setattr(core, "_codex_cli_auth_status", lambda: {"has_access_token": True})
    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")
    monkeypatch.setattr(core, "_plugin_working_directory", lambda _repo: str(repo))
    monkeypatch.setattr(
        core,
        "_run_command",
        lambda *args, **kwargs: {
            "ok": True,
            "exit_code": 0,
            "command": args[0],
            "cwd": str(kwargs["cwd"]),
            "stdout": "Hakua> [happy] hello\n",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        core,
        "synthesize_speech",
        lambda values: {
            "ok": True,
            "provider": values.get("provider") or "auto",
            "file_path": "hakua.wav",
            "text": values["text"],
        },
    )

    result = core.run_hakua_once(
        {
            "repo_root": str(repo),
            "prompt": "say hello",
            "speak": True,
            "tts_provider": "voicevox",
        }
    )

    assert result["ok"] is True
    assert result["tts"]["ok"] is True
    assert result["tts"]["provider"] == "voicevox"
    assert result["tts"]["text"] == "[happy] hello"


def test_handle_smoke_uses_hakua_prompt(monkeypatch):
    seen = {}

    def fake_run(values):
        seen.update(values)
        return {"ok": True, "reply": "ok"}

    monkeypatch.setattr(core, "run_hakua_once", fake_run)

    payload = json.loads(core.handle_smoke({}))

    assert payload["ok"] is True
    assert "はくあ" in seen["prompt"]


def test_prepare_runs_install_then_chat_build(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    commands = []

    monkeypatch.setattr(core, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": False, "installed": False},
    )

    def fake_run(cmd, cwd, timeout_seconds, env=None):
        commands.append(cmd)
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.prepare({"repo_root": str(repo), "timeout_seconds": 10})

    assert result["ok"] is True
    assert commands[0][:3] == ["npm", "install", "--no-save"]
    assert commands[1] == ["npm", "-w", "@aituber-onair/chat", "run", "build"]


def test_prepare_falls_back_to_cjs_build_on_windows_shell_gap(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    commands = []

    monkeypatch.setattr(core, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": True, "installed": True},
    )
    monkeypatch.setattr(core, "_is_windows", lambda: True)

    def fake_run(cmd, cwd, timeout_seconds, env=None):
        commands.append(cmd)
        if cmd[-1] == "build":
            return {
                "ok": False,
                "exit_code": 1,
                "command": cmd,
                "cwd": str(cwd),
                "stdout": "",
                "stderr": "'rm' is not recognized as an internal or external command",
            }
        (repo / "packages" / "chat" / "dist" / "cjs").mkdir(parents=True, exist_ok=True)
        (repo / "packages" / "chat" / "dist" / "cjs" / "agent.js").write_text(
            "exports.createAgentChatService = () => ({})\n", encoding="utf-8"
        )
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.prepare({"repo_root": str(repo), "timeout_seconds": 10})

    assert result["ok"] is True
    assert commands[0] == ["npm", "-w", "@aituber-onair/chat", "run", "build"]
    assert commands[1] == ["npm", "-w", "@aituber-onair/chat", "run", "build:cjs"]
    assert result["steps"][1]["fallback"] == "build:cjs"


def test_start_uses_detached_vite_command(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    popen_calls = []

    monkeypatch.setattr(core, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(core, "_url_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path / "workspace")

    class FakeProc:
        pid = 12345

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = core.start_fbx_app({"repo_root": str(repo), "fbx_port": 5188})

    assert result["ok"] is True
    assert result["ready"] is True
    cmd, kwargs = popen_calls[0]
    assert cmd == ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5188"]
    assert kwargs["cwd"].endswith("react-fbx-app")
