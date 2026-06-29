from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "aituber-kit"


def load_plugin():
    package_name = "aituber_kit_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def load_core():
    spec = importlib.util.spec_from_file_location(
        "aituber_kit_core_test",
        PLUGIN_DIR / "core.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["aituber_kit_core_test"] = module
    spec.loader.exec_module(module)
    return module


def test_register_exposes_tools_and_cli_command():
    plugin = load_plugin()

    class Ctx:
        def __init__(self):
            self.tools = []
            self.commands = []
            self.cli_commands = []
            self.llm = object()

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    plugin.register(ctx)

    assert {tool["name"] for tool in ctx.tools} == {
        "aituber_kit_status",
        "aituber_kit_configure",
        "aituber_kit_install",
        "aituber_kit_prepare",
        "aituber_kit_start",
        "aituber_kit_stop",
        "aituber_kit_speak",
        "aituber_kit_chat",
        "aituber_kit_stop_playback",
        "aituber_kit_bridge_start",
        "aituber_kit_bridge_stop",
    }
    assert all(tool["toolset"] == "aituber_kit" for tool in ctx.tools)
    assert ctx.commands[0][0][0] == "aituber-kit"
    assert ctx.cli_commands[0]["name"] == "aituber-kit"


def test_status_without_repo(tmp_path, monkeypatch):
    core = load_core()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with patch.object(core.dev_server, "dev_status", return_value={"ok": True, "repo": {"valid": False}}):
        payload = json.loads(core.handle_status({}))
    assert payload["ok"] is True
    assert "install" in " ".join(payload.get("recommended", []))


def test_prepare_writes_env_local(tmp_path, monkeypatch):
    core = load_core()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = home / "workspace" / "aituber-kit"
    repo.mkdir(parents=True)
    (repo / "package.json").write_text(
        json.dumps({"name": "aituber-kit", "version": "0.1.0"}),
        encoding="utf-8",
    )
    (repo / ".env.example").write_text("NEXT_PUBLIC_CLIENT_ID=\"\"\n", encoding="utf-8")

    with patch.object(core, "save_config", return_value={"ok": True}):
        payload = core.prepare({"repo_root": str(repo), "client_id": "test-client"})

    assert payload["ok"] is True
    env_local = (repo / ".env.local").read_text(encoding="utf-8")
    assert "NEXT_PUBLIC_CLIENT_ID=\"test-client\"" in env_local
    assert "NEXT_PUBLIC_MESSAGE_RECEIVER_ENABLED=\"true\"" in env_local


def test_speak_requires_api_key(tmp_path, monkeypatch):
    core = load_core()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("AITUBERKIT_API_KEY", raising=False)
    payload = core.speak({"text": "hello"})
    assert payload["ok"] is False
    assert "AITUBERKIT_API_KEY" in payload["error"]


def test_bridge_start_rejects_public_host_without_confirmation(monkeypatch):
    core = load_core()
    monkeypatch.setattr(core.dev_server, "tailscale_status", lambda: {})

    payload = core.start_bridge({"host": "0.0.0.0"})

    assert payload["ok"] is False
    assert payload["confirmation_required"] is True
    assert "noauth WebSocket" in payload["reason"]


def test_bridge_start_rejects_tailscale_bind_without_confirmation(monkeypatch):
    core = load_core()
    monkeypatch.setattr(core.dev_server, "tailscale_status", lambda: {"ipv4": "100.64.1.2"})

    payload = core.start_bridge({"tailscale": True})

    assert payload["ok"] is False
    assert payload["confirmation_required"] is True
    assert payload["host"] == "0.0.0.0"
