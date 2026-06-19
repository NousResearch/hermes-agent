"""Tests for the AITuber OnAir Hermes plugin bridge."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from plugins.aituber_onair import core, register
from plugins.aituber_onair import local_loops_worker


class _FakeContext:
    def __init__(self) -> None:
        self.tools = []
        self.commands = {}
        self.cli_commands = {}
        self.llm = object()

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


@pytest.fixture(autouse=True)
def _reset_aituber_llm_factory():
    core.bind_llm_factory(None)
    yield
    core.bind_llm_factory(None)


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
    assert "aituber_onair_context_status" in names
    assert "aituber_onair_stream_start_tweet" in names
    assert "aituber_onair_youtube_ready" in names
    assert "aituber_onair_youtube_comments_status" in names
    assert "aituber_onair_start_youtube_comments" in names
    assert "aituber_onair_stop_youtube_comments" in names
    assert "aituber_onair_loops_status" in names
    assert "aituber_onair_start_autonomous_talk" in names
    assert "aituber_onair_start_comment_reactions" in names
    assert "aituber_onair_enqueue_comment" in names
    assert "aituber_onair_stop_loops" in names
    assert all(tool["toolset"] == "aituber-onair" for tool in ctx.tools)
    assert "aituber" in ctx.commands
    assert "aituber-onair" in ctx.cli_commands


def test_register_binds_hermes_llm_factory(monkeypatch):
    ctx = _FakeContext()
    bound = {}

    def fake_bind(factory):
        bound["factory"] = factory

    monkeypatch.setattr(core, "bind_llm_factory", fake_bind)

    register(ctx)

    assert bound["factory"]() is ctx.llm


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


def test_context_status_reports_env_presence_without_secret_values(monkeypatch):
    monkeypatch.setenv("LM_TWITTERER_BOT_SCREEN_NAME", "hakua_public")
    monkeypatch.setenv("LM_TWITTERER_AUTH_TOKEN", "secret-auth-cookie")
    monkeypatch.setenv("LM_TWITTERER_CT0", "secret-csrf-cookie")
    monkeypatch.setenv("AITUBER_ONAIR_STREAM_URL", "https://example.com/live")

    result = core.context_status({"prompt": "email bob@example.com"})

    assert result["ok"] is True
    assert result["privacy"]["input_contains_sensitive_text"] is True
    assert result["privacy"]["speaks_secret_values"] is False
    assert result["readiness"]["lm_twitterer_ready"] is True
    env_by_name = {item["name"]: item for item in result["environment"]}
    assert env_by_name["LM_TWITTERER_AUTH_TOKEN"]["present"] is True
    assert "secret-auth-cookie" not in json.dumps(result, ensure_ascii=False)
    assert env_by_name["LM_TWITTERER_BOT_SCREEN_NAME"]["value"] == "hakua_public"
    assert result["stream"]["url"] == "https://example.com/live"


def test_run_hakua_once_can_include_safe_runtime_context(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    calls = []

    class Result:
        text = "[relaxed] safe hello"
        provider = "local"
        model = "main"
        usage = None
        audit = None

    class FakeLlm:
        def complete(self, messages, **kwargs):
            calls.append(messages)
            return Result()

    monkeypatch.setenv("LM_TWITTERER_AUTH_TOKEN", "secret-auth-cookie")
    monkeypatch.setenv("LM_TWITTERER_CT0", "secret-csrf-cookie")
    monkeypatch.setenv("LM_TWITTERER_BOT_SCREEN_NAME", "hakua_public")
    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")
    core.bind_llm_factory(lambda: FakeLlm())

    result = core.run_hakua_once(
        {
            "repo_root": str(repo),
            "prompt": "hello bob@example.com",
            "reply_backend": "hermes",
            "with_runtime_context": True,
        }
    )

    assert result["ok"] is True
    user_content = calls[0][1]["content"]
    assert "安全な実行文脈" in user_content
    assert "lm-twitterer=ready" in user_content
    assert "Do not repeat it literally" in user_content
    assert "secret-auth-cookie" not in user_content


def test_stream_start_tweet_dry_run_uses_lm_twitterer_with_url(monkeypatch):
    calls = []

    class FakeLmTwitterer:
        @staticmethod
        def post(topic, *, dry_run, provider, model, text):
            calls.append(
                {
                    "topic": topic,
                    "dry_run": dry_run,
                    "provider": provider,
                    "model": model,
                    "text": text,
                }
            )
            return {"ok": True, "dry_run": dry_run, "tweet_text": text}

    monkeypatch.setattr(
        core.importlib,
        "import_module",
        lambda name: FakeLmTwitterer if name == "plugins.lm-twitterer.core" else None,
    )

    result = core.stream_start_tweet(
        {
            "url": "https://example.com/live",
            "topic": "Galaxy S9 VRM test",
            "live": False,
        }
    )

    assert result["ok"] is True
    assert result["live"] is False
    assert calls[0]["dry_run"] is True
    assert "https://example.com/live" in calls[0]["text"]
    assert "#hermesagent" in calls[0]["text"]


def test_stream_start_tweet_refuses_private_live_url():
    result = core.stream_start_tweet(
        {
            "url": "http://127.0.0.1:5175/",
            "live": True,
        }
    )

    assert result["ok"] is False
    assert "local or unverified" in result["error"]


def test_local_loop_reply_enables_runtime_context(monkeypatch):
    seen = {}

    def fake_run(values):
        seen.update(values)
        return {"ok": True, "reply": "[relaxed] ok"}

    monkeypatch.setattr(core, "run_hakua_once", fake_run)

    result = local_loops_worker._reply("hello", play=False)

    assert result["ok"] is True
    assert seen["with_runtime_context"] is True


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
    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_working_directory", lambda _repo: str(repo))

    def fake_run(cmd, cwd, env, timeout_seconds):
        calls.append((cmd, cwd, env, timeout_seconds))
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "=== Codex Character Chat ===\nHakua> [happy] hello\n",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.run_hakua_once(
        {"repo_root": str(repo), "prompt": "greet", "reply_backend": "codex"}
    )

    assert result["ok"] is True
    assert result["reply"] == "[happy] hello"
    cmd, cwd, env, timeout_seconds = calls[0]
    assert cmd[0] == "node"
    assert "index.js" in cmd[1]
    once_file_args = [arg for arg in cmd if arg.startswith("--onceFile=")]
    assert len(once_file_args) == 1
    once_file = Path(once_file_args[0].split("=", 1)[1])
    assert once_file.read_text(encoding="utf-8") == "greet"
    assert env["CODEX_CHARACTER_NAME"] == "Hakua"
    assert env["CODEX_CHARACTER_SYSTEM_PROMPT"] == "Hakua prompt"
    assert cwd == repo
    assert timeout_seconds == core.DEFAULT_TIMEOUT_SECONDS


def test_run_hakua_once_prefers_bound_hermes_llm(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    calls = []

    class Result:
        text = "[happy] Hermes-side hello"
        provider = "local"
        model = "main"
        usage = None
        audit = {"purpose": "aituber-onair.hakua"}

    class FakeLlm:
        def complete(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return Result()

    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")
    monkeypatch.setattr(core, "_node_exe", lambda: "node")
    monkeypatch.setattr(
        core,
        "_codex_sdk_installed",
        lambda _repo: {"ok": True, "installed": True},
    )
    monkeypatch.setattr(
        core, "_codex_cli_auth_status", lambda: {"has_access_token": True}
    )
    monkeypatch.setattr(
        core,
        "_run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Codex SDK should not be called when Hermes LLM is bound")
        ),
    )
    core.bind_llm_factory(lambda: FakeLlm())
    try:
        result = core.run_hakua_once(
            {
                "repo_root": str(repo),
                "prompt": "say hello",
                "reply_backend": "auto",
                "hermes_provider": "local",
                "hermes_model": "main",
            }
        )
    finally:
        core.bind_llm_factory(None)

    assert result["ok"] is True
    assert result["provider"] == "hermes-agent"
    assert result["reply"] == "[happy] Hermes-side hello"
    assert calls[0][0] == [
        {"role": "system", "content": "Hakua prompt"},
        {"role": "user", "content": "say hello"},
    ]
    assert calls[0][1]["provider"] == "local"
    assert calls[0][1]["model"] == "main"
    assert calls[0][1]["purpose"] == "aituber-onair.hakua"


def test_run_hakua_once_can_use_hermes_cli_backend_without_bound_llm(
    monkeypatch, tmp_path
):
    repo = _fake_repo(tmp_path)
    calls = []

    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")

    def fake_run(cmd, cwd, env, timeout_seconds):
        calls.append((cmd, cwd, env, timeout_seconds))
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "[relaxed] Hermes CLI hello\n",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)
    core.bind_llm_factory(None)

    result = core.run_hakua_once(
        {"repo_root": str(repo), "prompt": "say hello", "reply_backend": "hermes"}
    )

    assert result["ok"] is True
    assert result["provider"] == "hermes-agent-cli"
    assert result["reply"] == "[relaxed] Hermes CLI hello"
    cmd, _cwd, env, _timeout = calls[0]
    assert "-m" in cmd
    assert "hermes_cli" in cmd
    assert "--oneshot" in cmd
    assert env["HERMES_YOLO_MODE"] == "1"


def test_run_hakua_once_falls_back_to_hermes_cli_when_bound_model_is_unsupported(
    monkeypatch, tmp_path
):
    repo = _fake_repo(tmp_path)

    class FakeLlm:
        def complete(self, messages, **kwargs):
            raise RuntimeError(
                "The 'gpt-5.5-low' model is not supported when using Codex."
            )

    monkeypatch.setattr(core, "_plugin_character_name", lambda: "Hakua")
    monkeypatch.setattr(core, "_plugin_system_prompt", lambda: "Hakua prompt")
    monkeypatch.setattr(
        core,
        "_run_command",
        lambda *args, **kwargs: {
            "ok": True,
            "exit_code": 0,
            "command": args[0],
            "cwd": str(kwargs["cwd"]),
            "stdout": "[relaxed] fallback hello\n",
            "stderr": "",
        },
    )
    core.bind_llm_factory(lambda: FakeLlm())

    result = core.run_hakua_once(
        {"repo_root": str(repo), "prompt": "say hello", "reply_backend": "hermes"}
    )

    assert result["ok"] is True
    assert result["provider"] == "hermes-agent-cli"
    assert result["reply"] == "[relaxed] fallback hello"
    assert "gpt-5.5-low" in result["hermes_facade_error"]


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


def test_run_hakua_once_retries_default_model_for_unsupported_model(
    monkeypatch, tmp_path
):
    repo = _fake_repo(tmp_path)
    calls = []

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
    monkeypatch.setattr(core, "_plugin_model", lambda _explicit=None: "deepseek-v4-pro")

    def fake_run(cmd, cwd, env, timeout_seconds):
        calls.append((cmd, env))
        if len(calls) == 1:
            return {
                "ok": True,
                "exit_code": 0,
                "command": cmd,
                "cwd": str(cwd),
                "stdout": "=== Codex Character Chat ===\nHakua> \n",
                "stderr": (
                    "[error] codex-sdk provider failed.\n\n"
                    "Original error:\n"
                    "Error\n"
                    '{"error":{"message":"The model is not supported when using Codex."}}'
                ),
            }
        return {
            "ok": True,
            "exit_code": 0,
            "command": cmd,
            "cwd": str(cwd),
            "stdout": "=== Codex Character Chat ===\nHakua> [happy] hello\n",
            "stderr": "",
        }

    monkeypatch.setattr(core, "_run_command", fake_run)

    result = core.run_hakua_once({"repo_root": str(repo), "prompt": "say hello"})

    assert result["ok"] is True
    assert result["reply"] == "[happy] hello"
    assert result["model"] == "Codex CLI default"
    assert result["fallback_from_model"] == "deepseek-v4-pro"
    assert len(calls) == 2
    assert "--model=deepseek-v4-pro" in calls[0][0]
    assert all(not arg.startswith("--model=") for arg in calls[1][0])
    assert "CODEX_SDK_MODEL" not in calls[1][1]


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


def test_start_vrm_can_bind_for_lan_display(monkeypatch, tmp_path):
    repo = _fake_repo(tmp_path)
    popen_calls = []

    monkeypatch.setattr(core, "_npm_exe", lambda: "npm")
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(core, "_url_ready", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path / "workspace")
    monkeypatch.setattr(core, "_detect_lan_ipv4", lambda: "192.168.1.23")

    class FakeProc:
        pid = 45678

        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = core.start_avatar_app(
        {
            "repo_root": str(repo),
            "avatar_kind": "vrm",
            "host": "0.0.0.0",
        }
    )

    assert result["ok"] is True
    assert result["url"] == "http://192.168.1.23:5175/"
    assert result["readiness_url"] == "http://127.0.0.1:5175/"
    assert result["host"] == "0.0.0.0"
    assert result["public_host"] == "192.168.1.23"
    cmd, _kwargs = popen_calls[0]
    assert cmd == ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5175"]


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
    monkeypatch.setattr(core, "_is_windows", lambda: False)

    assert (
        core._obs_candidates_from_uninstall_registry(
            "HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall"
        )
        == []
    )


def test_obs_registry_candidates_reads_install_location(monkeypatch):
    monkeypatch.setattr(core, "_is_windows", lambda: True)

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


def test_extract_youtube_live_id_from_watch_and_live_urls():
    assert (
        core._extract_youtube_live_id("https://www.youtube.com/watch?v=abc123XYZ")
        == "abc123XYZ"
    )
    assert core._extract_youtube_live_id("https://youtu.be/abc123XYZ") == "abc123XYZ"
    assert (
        core._extract_youtube_live_id("https://www.youtube.com/live/abc123XYZ")
        == "abc123XYZ"
    )


def test_start_youtube_comments_requires_secret_env(monkeypatch):
    monkeypatch.delenv("AITUBER_ONAIR_YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = core.start_youtube_comments({"live_id": "abc123XYZ"})

    assert result["ok"] is False
    assert "API key" in result["error"]
    assert result["api_key_env"] == "AITUBER_ONAIR_YOUTUBE_API_KEY"


def test_youtube_comments_status_reports_presence_without_key_value(monkeypatch):
    monkeypatch.setenv("AITUBER_ONAIR_YOUTUBE_API_KEY", "secret-youtube-key")
    monkeypatch.setattr(core, "_plugin_config", lambda: {"youtube_live_id": "abc123XYZ"})
    monkeypatch.setattr(
        core,
        "_youtube_comments_active_status",
        lambda: {"ok": False, "reason": "none"},
    )

    result = core.youtube_comments_status({})

    assert result["ok"] is True
    assert result["live_id"] == "abc123XYZ"
    assert result["api_key_present"] is True
    assert "secret-youtube-key" not in json.dumps(result)


def test_start_youtube_comments_spawns_worker_without_recording_api_key(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path)
    monkeypatch.setenv("AITUBER_ONAIR_YOUTUBE_API_KEY", "secret-youtube-key")
    popen_calls = []

    class FakeProc:
        pid = 34567

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)

    result = core.start_youtube_comments(
        {
            "live_id": "https://www.youtube.com/watch?v=live-123",
            "poll_seconds": 1,
            "skip_existing": True,
        }
    )

    assert result["ok"] is True
    assert result["active"]["pid"] == 34567
    assert result["active"]["poll_seconds"] == 2.0
    assert "secret-youtube-key" not in json.dumps(result)
    cmd, kwargs = popen_calls[0]
    assert "plugins.aituber_onair.youtube_comments_worker" in cmd
    assert "--skip-existing" in cmd
    assert kwargs["env"]["AITUBER_ONAIR_YOUTUBE_API_KEY"] == "secret-youtube-key"


def test_start_autonomous_talk_spawns_local_loop_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    popen_calls = []

    class FakeProc:
        pid = 45678

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = core.start_autonomous_talk_loop(
        {"interval_seconds": 30, "topic": "Galaxy display test", "play": False}
    )

    assert result["ok"] is True
    assert result["active"]["pid"] == 45678
    assert result["active"]["interval_seconds"] == 30.0
    cmd, kwargs = popen_calls[0]
    assert "plugins.aituber_onair.local_loops_worker" in cmd
    assert "--mode" in cmd
    assert "autonomous" in cmd
    assert "--topic" in cmd
    assert "Galaxy display test" in cmd
    assert "--play" not in cmd
    assert kwargs["env"]["PYTHONPATH"]


def test_enqueue_comment_writes_local_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path)

    result = core.enqueue_comment(
        {"author": "bob", "text": "反応して", "source": "local-test"}
    )

    assert result["ok"] is True
    queue_file = Path(result["queue_file"])
    rows = [json.loads(line) for line in queue_file.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["author"] == "bob"
    assert rows[-1]["text"] == "反応して"
    assert rows[-1]["source"] == "local-test"


def test_start_comment_reactions_spawns_local_loop_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "_workspace_root", lambda: tmp_path)
    monkeypatch.setattr(core, "_pid_alive", lambda _pid: False)
    popen_calls = []

    class FakeProc:
        pid = 56789

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = core.start_comment_reaction_loop({"poll_seconds": 2, "play": True})

    assert result["ok"] is True
    assert result["active"]["pid"] == 56789
    cmd, _kwargs = popen_calls[0]
    assert "plugins.aituber_onair.local_loops_worker" in cmd
    assert "--mode" in cmd
    assert "comments" in cmd
    assert "--queue-file" in cmd
    assert "--processed-file" in cmd
    assert "--play" in cmd


def test_comment_reaction_worker_skips_processed_comments(monkeypatch, tmp_path):
    queue_file = tmp_path / "queue.jsonl"
    processed_file = tmp_path / "processed.json"
    queue_file.write_text(
        "\n".join(
            [
                json.dumps({"id": "one", "author": "bob", "text": "first"}),
                json.dumps({"id": "two", "author": "bob", "text": "second"}),
            ]
        ),
        encoding="utf-8",
    )
    processed_file.write_text(json.dumps({"ids": ["one"]}), encoding="utf-8")

    replies = []
    monkeypatch.setattr(
        local_loops_worker,
        "_reply",
        lambda prompt, play: replies.append(prompt)
        or {"ok": True, "reply": "[happy] ok"},
    )
    monkeypatch.setattr(local_loops_worker, "_log", lambda _payload: None)

    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr(local_loops_worker.time, "sleep", fake_sleep)

    args = type(
        "Args",
        (),
        {
            "queue_file": str(queue_file),
            "processed_file": str(processed_file),
            "poll_seconds": 1.0,
            "play": False,
        },
    )()

    assert local_loops_worker.run_comments(args) == 0
    assert len(replies) == 1
    assert "second" in replies[0]
    processed = json.loads(processed_file.read_text(encoding="utf-8"))
    assert set(processed["ids"]) == {"one", "two"}
