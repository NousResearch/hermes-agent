from __future__ import annotations

import json
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
    # Ensure no Anthropic credentials leak into this test even if some import
    # side-effect rehydrates them in the worker process.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    from hermes_cli.plugins import discover_plugins, invoke_hook
    from hermes_constants import get_hermes_home

    # Force enabled state without requiring backend credentials.
    discover_plugins(force=True)
    from hermes_plugins import kaze_claude_mode as mod
    monkeypatch.setattr(
        mod,
        "_resolve_toolful_claude_backend",
        lambda **_: (False, "anthropic/claude-sonnet-4.6", {"provider": "anthropic", "api_key": ""}),
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
    assert "Routing this message to normal Hermes/Kaze" in results[0]["text"]

    # State file is profile-safe (under HERMES_HOME), not ~/.hermes.
    assert (get_hermes_home() / "state" / "kaze_claude_mode.json").exists()


def test_claude_mode_sets_gateway_override_when_backend_available(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)

    # Dummy credentials so resolve_runtime_provider reports "toolful backend ok".
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    from hermes_cli.plugins import discover_plugins, invoke_hook
    discover_plugins(force=True)
    from hermes_plugins.kaze_claude_mode import set_enabled, state_key_from_source

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    gw = _FakeGateway("agent:main:telegram:dm:1")
    event = SimpleNamespace(text="Hello", source=src)

    results = invoke_hook("pre_gateway_dispatch", event=event, gateway=gw, session_store=None)
    # No rewrite when backend is available; hook returns [].
    assert results == []
    assert gw._session_model_overrides["agent:main:telegram:dm:1"]["provider"] == "anthropic"
    assert "model" in gw._session_model_overrides["agent:main:telegram:dm:1"]
    assert gw._evicted == ["agent:main:telegram:dm:1"]


@pytest.mark.asyncio
async def test_claude_mode_smoke_uses_tools_without_leaking_outputs(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_test"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    _enable_plugin(hermes_home, "kaze-claude-mode")
    _reset_plugin_singleton(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    # Provide a fake tool dispatcher so smoke doesn't actually run shell commands.
    class _FakeCtx:
        def dispatch_tool(self, name: str, args: dict, **kw):  # noqa: ARG002
            if name == "terminal":
                return json.dumps({"stdout": "KAZE_CLAUDE_MODE_TOOL_OK\\n"})
            if name == "write_file":
                return json.dumps({"ok": True})
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

    src = _tg_source()
    chat_key = state_key_from_source(src)
    set_enabled(chat_key, True, source=src)

    packet = {"key": chat_key, "args": "smoke", "session_key": "agent:main:telegram:dm:1"}
    reply = await handle_internal_mode(_encode_packet(packet))

    assert "Smoke: **OK**" in reply
    assert "KAZE_CLAUDE_MODE_TOOL_OK" in reply
    # Must not dump raw tool JSON or arbitrary stdout bodies.
    assert "\"stdout\"" not in reply
