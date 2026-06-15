"""Tests for the AITuber OnAir Hermes plugin bridge."""

from __future__ import annotations

import json
import subprocess
import sys
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
    (repo / "packages" / "core" / "examples" / "react-fbx-app").mkdir(parents=True)
    (repo / "packages" / "core" / "examples" / "react-vrm-app").mkdir(parents=True)
    (repo / "package.json").write_text('{"name":"aituber-onair"}', encoding="utf-8")
    (
        repo / "packages" / "chat" / "examples" / "codex-character-chat" / "index.js"
    ).write_text("console.log('ok')\n", encoding="utf-8")
    (repo / "packages" / "chat" / "dist" / "cjs" / "agent.js").write_text(
        "exports.createAgentChatService = () => ({})\n", encoding="utf-8"
    )
    (
        repo / "packages" / "core" / "examples" / "react-fbx-app" / "package.json"
    ).write_text('{"name":"react-fbx-app"}', encoding="utf-8")
    (
        repo / "packages" / "core" / "examples" / "react-vrm-app" / "package.json"
    ).write_text('{"name":"react-vrm-app"}', encoding="utf-8")
    return repo


def test_registers_tools_slash_and_cli_command():
    ctx = _FakeContext()
    register(ctx)

    names = {tool["name"] for tool in ctx.tools}
    assert "aituber_onair_status" in names
    assert "aituber_onair_tts_status" in names
    assert "aituber_onair_speak" in names
    assert "aituber_onair_say" in names
    assert "aituber_onair_youtube_ready" in names
    assert all(tool["toolset"] == "aituber-onair" for tool in ctx.tools)
    assert "aituber" in ctx.commands
    assert "aituber-onair" in ctx.cli_commands


def test_resolve_repo_root_accepts_aituber_checkout(tmp_path):
    repo = _fake_repo(tmp_path)

    assert core.resolve_repo_root(str(repo)) == repo


def test_child_process_env_omits_provider_secrets(monkeypatch):
    monkeypatch.setenv("PATH", "C:/tools")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-github-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-aws-secret")

    env = core._child_process_env({"AITUBER_ONAIR_HERMES_PLUGIN": "1"})

    assert env["PATH"] == "C:/tools"
    assert env["AITUBER_ONAIR_HERMES_PLUGIN"] == "1"
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


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

    monkeypatch.setattr(
        core, "_plugin_config", lambda: {"tts_voice": "hakua", "tts_speed": 1.06}
    )
    monkeypatch.setattr(
        irodori_core,
        "synthesize_text",
        lambda **kwargs: seen.update(kwargs)
        or {"ok": True, "provider": "irodori", "file_path": str(kwargs["output_path"])},
    )

    result = core._synthesize_irodori(
        {"text": "hello", "output_path": str(tmp_path / "voice.wav"), "voice": ""}
    )

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
    once_file_args = [arg for arg in cmd if arg.startswith("--onceFile=")]
    assert len(once_file_args) == 1
    once_file = Path(once_file_args[0].split("=", 1)[1])
    assert once_file.read_text(encoding="utf-8") == "挨拶して"
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
    monkeypatch.setattr(
        core, "_codex_cli_auth_status", lambda: {"has_access_token": True}
    )
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


def test_run_hakua_once_rejects_empty_provider_failure(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)

    monkeypatch.setattr(core, "_node_exe", lambda: "node")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": True, "installed": True},
    )
    monkeypatch.setattr(
        core, "_codex_cli_auth_status", lambda: {"has_access_token": True}
    )
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
            "stdout": "=== Codex Character Chat ===\nHakua> \n",
            "stderr": "[error] codex-sdk provider failed.\n\nOriginal error:\nUnable to locate Codex CLI binaries.",
        },
    )

    result = core.run_hakua_once({"repo_root": str(repo), "prompt": "say hello"})

    assert result["ok"] is False
    assert result["reply"] == ""
    assert result["error"] == "Codex SDK provider failed."


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
    assert commands[0][:2] == ["npm", "install"]
    assert "--include=optional" in commands[0]
    assert "@openai/codex-sdk" in commands[0]
    assert "@openai/codex" in commands[0]
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

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-github-token")
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
    assert "OPENAI_API_KEY" not in kwargs["env"]
    assert "GITHUB_TOKEN" not in kwargs["env"]


def test_start_vroid_uses_vrm_app_and_default_port(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    popen_calls = []

    monkeypatch.setattr(core, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(core, "_url_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path / "workspace")

    class FakeProc:
        pid = 23456

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = core.start_avatar_app({"repo_root": str(repo), "avatar_kind": "vroid"})

    assert result["ok"] is True
    assert result["ready"] is True
    assert result["avatar_kind"] == "vrm"
    assert result["url"] == "http://127.0.0.1:5175/"
    cmd, kwargs = popen_calls[0]
    assert cmd == ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5175"]
    assert kwargs["cwd"].endswith("react-vrm-app")
    assert kwargs["env"]["AITUBER_ONAIR_AVATAR_KIND"] == "vrm"


def test_run_command_uses_sanitized_default_env(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = core._run_command(["npm", "--version"], cwd=tmp_path, timeout_seconds=5)

    assert result["ok"] is True
    assert "OPENAI_API_KEY" not in captured["env"]


def test_obs_registry_candidates_skips_non_windows(monkeypatch):
    monkeypatch.setattr(core.os, "name", "posix", raising=False)

    assert (
        core._obs_candidates_from_uninstall_registry(
            "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
        )
        == []
    )


def test_obs_registry_candidates_reads_install_location(monkeypatch):
    monkeypatch.setattr(core.os, "name", "nt", raising=False)

    class FakeKey:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    class FakeWinreg:
        HKEY_LOCAL_MACHINE = "HKLM"
        HKEY_CURRENT_USER = "HKCU"

        @staticmethod
        def OpenKey(hive_or_key, subkey):
            if hive_or_key == "HKLM":
                return FakeKey("root")
            if isinstance(hive_or_key, FakeKey) and hive_or_key.name == "root":
                return FakeKey(subkey)
            raise OSError("missing key")

        @staticmethod
        def QueryInfoKey(key):
            assert key.name == "root"
            return (3, 0, 0)

        @staticmethod
        def EnumKey(key, index: int) -> str:
            assert key.name == "root"
            return ["obs", "other", "broken"][index]

        @staticmethod
        def QueryValueEx(key, value_name: str):
            values = {
                "obs": {
                    "DisplayName": "OBS Studio",
                    "InstallLocation": "D:/Apps/OBS Studio",
                },
                "other": {
                    "DisplayName": "Other App",
                    "InstallLocation": "D:/Apps/Other",
                },
                "broken": {"DisplayName": "OBS Plugin"},
            }
            try:
                return values[key.name][value_name], 1
            except KeyError as exc:
                raise OSError("missing value") from exc

    monkeypatch.setitem(sys.modules, "winreg", FakeWinreg)

    assert core._obs_candidates_from_uninstall_registry(
        "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
    ) == [Path("D:/Apps/OBS Studio/bin/64bit/obs64.exe")]


def test_youtube_ready_reports_missing_obs(monkeypatch):
    monkeypatch.setattr(
        core,
        "status",
        lambda: {
            "ok": True,
            "config": {"url": "http://127.0.0.1:5175/"},
            "active": {"alive": True, "url_ready": True},
            "tts": {"ready": True},
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        core,
        "_obs_status",
        lambda: {"installed": False, "candidates": [], "expected_paths": []},
    )

    result = core.youtube_ready({})

    assert result["ok"] is False
    assert result["readiness"]["avatar_app_running"] is True
    assert result["readiness"]["obs_available"] is False
    assert "OBS Studio was not found." in result["blockers"]


def test_youtube_ready_passes_with_running_avatar_and_obs(monkeypatch):
    monkeypatch.setattr(
        core,
        "status",
        lambda: {
            "ok": True,
            "config": {"url": "http://127.0.0.1:5175/"},
            "active": {"alive": True, "url_ready": True},
            "tts": {"ready": False},
            "recommended_actions": [],
        },
    )
    monkeypatch.setattr(
        core,
        "_obs_status",
        lambda: {
            "installed": True,
            "candidates": ["C:/Program Files/obs-studio/bin/64bit/obs64.exe"],
            "expected_paths": [],
        },
    )

    result = core.youtube_ready({})

    assert result["ok"] is True
    assert result["avatar_url"] == "http://127.0.0.1:5175/"
    assert result["obs_browser_source"]["recommended_source_type"] == "Browser Source"
    assert "stream_key" in result["youtube_encoder"]
