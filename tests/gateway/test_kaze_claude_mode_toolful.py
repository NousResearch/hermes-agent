from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _reset_plugin_singleton(monkeypatch):
    import hermes_cli.plugins as _plugins_mod
    monkeypatch.setattr(_plugins_mod, "_plugin_manager", None)


def _enable_plugin(hermes_home: Path, name: str) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    cfg = {"plugins": {"enabled": [name]}}
    (hermes_home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


class _FakeGateway:
    def __init__(self, session_key: str):
        self._session_model_overrides: dict = {}
        self._evicted: list[str] = []
        self._session_key = session_key

    def _session_key_for_source(self, source) -> str:  # noqa: ARG002
        return self._session_key

    def _evict_cached_agent(self, session_key: str) -> None:
        self._evicted.append(session_key)


class _FakeSessionEntry:
    def __init__(self, session_id: str):
        self.session_id = session_id


class _FakeSessionStore:
    def __init__(self, session_key: str, messages: list[dict]):
        self._entries = {session_key: _FakeSessionEntry("session-1")}
        self._messages = messages
        self.loaded = False

    def _ensure_loaded(self) -> None:
        self.loaded = True

    def load_transcript(self, session_id: str):
        assert session_id == "session-1"
        return self._messages


def _tg_source(chat_id: str = "1"):
    return SimpleNamespace(
        platform=SimpleNamespace(value="telegram"),
        chat_id=chat_id,
        thread_id="",
        user_id="u",
        user_name="user",
        chat_type="dm",
    )


def test_claude_mode_bounce_when_backend_unavailable(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    from hermes_cli.plugins import discover_plugins, invoke_hook
    from hermes_constants import get_hermes_home

    discover_plugins(force=True)
    from hermes_plugins import kaze_claude_mode as mod
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (False, {"backend": "claude-code-cli", "cmd": "", "path": "", "error": "not found"}),
    )
    from hermes_plugins.kaze_claude_mode import set_enabled, state_key_from_source

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    gw = _FakeGateway("agent:main:telegram:dm:1")
    event = SimpleNamespace(text="Please edit files", source=src)

    results = invoke_hook("pre_gateway_dispatch", event=event, gateway=gw, session_store=None)
    assert len(results) == 1
    assert results[0]["action"] == "rewrite"
    assert results[0]["text"].startswith("/kaze-claude-mode-dispatch ")
    # Must not echo the user prompt into the rewritten command (avoid log leakage).
    assert "Please edit files" not in results[0]["text"]
    packet = mod._decode_packet(results[0]["text"].split(maxsplit=1)[1])
    assert packet["args"] == "unavailable"

    # State file is profile-safe (under HERMES_HOME), not ~/.hermes.
    assert (get_hermes_home() / "state" / "kaze_claude_mode.json").exists()


def test_claude_mode_rewrites_plain_message_to_internal_run_when_backend_available(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    from hermes_cli.plugins import discover_plugins, invoke_hook
    discover_plugins(force=True)
    from hermes_plugins import kaze_claude_mode as mod
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )
    from hermes_plugins.kaze_claude_mode import set_enabled, state_key_from_source

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    gw = _FakeGateway("agent:main:telegram:dm:1")
    event = SimpleNamespace(text="Hello", source=src)

    results = invoke_hook("pre_gateway_dispatch", event=event, gateway=gw, session_store=None)
    assert len(results) == 1
    assert results[0]["action"] == "rewrite"
    assert results[0]["text"].startswith("/kaze-claude-mode-run ")
    # Must not embed the user message.
    assert "Hello" not in results[0]["text"]
    token = results[0]["text"].split(maxsplit=1)[1]
    assert token in mod._PENDING
    # No provider-switch masquerade.
    assert gw._session_model_overrides == {}
    assert gw._evicted == []


def test_claude_mode_injects_recent_session_transcript_without_rewrite_leak(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    from hermes_cli.plugins import discover_plugins, invoke_hook
    discover_plugins(force=True)
    from hermes_plugins import kaze_claude_mode as mod
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )
    from hermes_plugins.kaze_claude_mode import set_enabled, state_key_from_source

    src = _tg_source()
    chat_key = state_key_from_source(src)
    session_key = "agent:main:telegram:dm:1"
    set_enabled(chat_key, True, source=src)
    store = _FakeSessionStore(
        session_key,
        [
            {"role": "user", "content": "Earlier user request"},
            {"role": "assistant", "content": "Earlier assistant answer"},
            {"role": "tool", "content": "raw tool JSON should be skipped"},
        ],
    )

    event = SimpleNamespace(text="What did I miss?", source=src)
    results = invoke_hook("pre_gateway_dispatch", event=event, gateway=_FakeGateway(session_key), session_store=store)

    assert len(results) == 1
    assert results[0]["text"].startswith("/kaze-claude-mode-run ")
    assert "Earlier user request" not in results[0]["text"]
    token = results[0]["text"].split(maxsplit=1)[1]
    prompt = mod._PENDING[token].prompt
    assert "<recent_telegram_session_transcript>" in prompt
    assert "Earlier user request" in prompt
    assert "Earlier assistant answer" in prompt
    assert "raw tool JSON" not in prompt
    assert "<latest_user_message>" in prompt
    assert "What did I miss?" in prompt
    assert store.loaded is True


@pytest.mark.asyncio
async def test_claude_mode_smoke_uses_tools_without_leaking_outputs(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    # Provide a fake tool dispatcher so smoke doesn't actually run shell commands.
    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            if name == "write_file":
                return json.dumps({"ok": True})
            if name == "terminal":
                # - In smoke: claude run returns no stdout we care about; verify step reads the temp file.
                if "python -c" in (args.get("command") or ""):
                    return json.dumps({"stdout": "KAZE_CLAUDE_MODE_FILE_OK\\n"})
                return json.dumps({"stdout": "ok\\n", "exit_code": 0})
            return json.dumps({"error": "unexpected tool"})

        # register_* stubs (not used in this test, but plugin register() expects them)
        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import _encode_packet, handle_internal_mode, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet = {"key": chat_key, "args": "smoke", "session_key": "agent:main:telegram:dm:1"}
    reply = await handle_internal_mode(_encode_packet(packet))

    assert "Smoke: **OK**" in reply
    # Must not dump raw tool JSON or arbitrary stdout bodies.
    assert "\"stdout\"" not in reply


@pytest.mark.asyncio
async def test_claude_mode_status_reports_backend_precisely(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            return json.dumps({"error": "not used"})

        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import _encode_packet, handle_internal_mode, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet = {"key": chat_key, "args": "status", "session_key": "agent:main:telegram:dm:1"}
    reply = await handle_internal_mode(_encode_packet(packet))

    assert "backend:" in reply
    assert "claude-code-cli" in reply
    assert "anthropic" not in reply.lower()


@pytest.mark.asyncio
async def test_claude_mode_status_reports_allowed_tools_permission_mode_yolo_and_pending(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)
    monkeypatch.delenv("KAZE_CLAUDE_MODE_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("KAZE_CLAUDE_MODE_PERMISSION_MODE", raising=False)

    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            return json.dumps({"error": "not used"})

        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins

    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import _encode_packet, handle_internal_mode, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet = {"key": chat_key, "args": "status", "session_key": "agent:main:telegram:dm:1"}
    reply = await handle_internal_mode(_encode_packet(packet))

    assert "allowedTools:" in reply
    assert "`Read,Write,Edit,Grep,Glob`" in reply
    assert "permission-mode:" in reply
    assert "`acceptEdits`" in reply
    assert "yolo:" in reply
    assert "**off**" in reply
    assert "pending:" in reply
    assert "**none**" in reply
    assert "max-turns:" in reply
    assert "`90`" in reply


@pytest.mark.asyncio
async def test_claude_mode_env_overrides_allowed_tools_and_permission_mode(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)
    monkeypatch.setenv("KAZE_CLAUDE_MODE_ALLOWED_TOOLS", "Read,Edit")
    monkeypatch.setenv("KAZE_CLAUDE_MODE_PERMISSION_MODE", "auto")

    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            return json.dumps({"error": "not used"})

        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins

    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import _encode_packet, handle_internal_mode, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet = {"key": chat_key, "args": "status", "session_key": "agent:main:telegram:dm:1"}
    reply = await handle_internal_mode(_encode_packet(packet))

    assert "`Read,Edit`" in reply
    assert "`auto`" in reply


@pytest.mark.asyncio
async def test_claude_mode_yolo_toggles_bypass_permission_mode(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)
    monkeypatch.setenv("KAZE_CLAUDE_MODE_PERMISSION_MODE", "auto")

    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            return json.dumps({"error": "not used"})

        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins

    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import _encode_packet, handle_internal_mode, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet_on = {"key": chat_key, "args": "yolo on", "session_key": "agent:main:telegram:dm:1"}
    reply_on = await handle_internal_mode(_encode_packet(packet_on))
    assert "yolo: **on**" in reply_on
    assert "`bypassPermissions`" in reply_on

    packet_off = {"key": chat_key, "args": "yolo off", "session_key": "agent:main:telegram:dm:1"}
    reply_off = await handle_internal_mode(_encode_packet(packet_off))
    assert "yolo: **off**" in reply_off
    assert "`auto`" in reply_off


@pytest.mark.asyncio
async def test_claude_mode_permission_block_creates_pending_approval_and_approve_retries(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)
    monkeypatch.delenv("KAZE_CLAUDE_MODE_ALLOWED_TOOLS", raising=False)

    class _FakeCtx:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            self.calls.append((name, args))
            if name == "write_file":
                return json.dumps({"ok": True})
            if name == "terminal":
                cmd = args.get("command") or ""
                if "Bash(git status *)" in cmd:
                    return json.dumps({"stdout": "ok\\n", "exit_code": 0})
                return json.dumps({"stdout": "Permission blocked: approve Bash(git status *)\\n", "exit_code": 1})
            return json.dumps({"error": "unexpected tool"})

        def register_hook(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

        def register_command(self, *a, **kw):  # noqa: ANN001,ARG002
            return None

    from hermes_cli.plugins import discover_plugins

    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import (
        _encode_packet,
        _pending_approval,
        _stash_pending_prompt,
        handle_internal_mode,
        handle_internal_run,
        state_key_from_source,
        set_enabled,
    )
    from hermes_plugins import kaze_claude_mode as mod

    mod._CTX = _FakeCtx()
    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    secret_prompt = "TOPSECRET: do not leak"
    token = _stash_pending_prompt(chat_key=chat_key, session_key="agent:main:telegram:dm:1", prompt=secret_prompt)
    out = await handle_internal_run(token)
    assert "approval pending" in out.lower()
    assert secret_prompt not in out

    pending = _pending_approval(chat_key)
    assert pending is not None
    assert pending.get("tool_rule") == "Bash(git status *)"
    assert pending.get("prompt_sha256") == hashlib.sha256(secret_prompt.encode("utf-8")).hexdigest()

    # State file must not contain the raw prompt body.
    state_text = (hermes_home / "state" / "kaze_claude_mode.json").read_text(encoding="utf-8")
    assert secret_prompt not in state_text

    approve_packet = {"key": chat_key, "args": "approve", "session_key": "agent:main:telegram:dm:1"}
    out2 = await handle_internal_mode(_encode_packet(approve_packet))
    assert "ok" in out2.lower()
    assert _pending_approval(chat_key) is None


def test_claude_mode_keeps_other_slash_commands_as_escape_hatches(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    from hermes_cli.plugins import discover_plugins

    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import build_pre_dispatch_decision, state_key_from_source, set_enabled
    from hermes_plugins import kaze_claude_mode as mod

    monkeypatch.setattr(
        mod,
        "_resolve_claude_code_cli_backend",
        lambda: (True, {"backend": "claude-code-cli", "cmd": "claude", "path": "/usr/bin/claude"}),
    )

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    # /claude-mode is rewritten (handled by this plugin), but other slash commands must pass through.
    event = SimpleNamespace(text="/help", source=src)
    assert build_pre_dispatch_decision(event, gateway=_FakeGateway("agent:main:telegram:dm:1")) is None
