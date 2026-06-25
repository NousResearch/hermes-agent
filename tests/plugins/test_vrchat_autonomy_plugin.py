from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "vrchat-autonomy"


def load_plugin():
    package_name = "vrchat_autonomy_test_plugin"
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


def load_movement():
    spec = importlib.util.spec_from_file_location(
        "vrchat_autonomy_movement_test",
        PLUGIN_DIR / "movement.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrchat_autonomy_movement_test"] = module
    spec.loader.exec_module(module)
    return module


def test_register_exposes_tools_and_cli():
    plugin = load_plugin()

    class Ctx:
        def __init__(self):
            self.tools = []
            self.commands = []
            self.cli_commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_command(self, *args, **kwargs):
            self.commands.append((args, kwargs))

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

    ctx = Ctx()
    plugin.register(ctx)

    assert {tool["name"] for tool in ctx.tools} == {
        "vrchat_autonomy_plugin_status",
        "vrchat_autonomy_plugin_chatbox",
        "vrchat_autonomy_plugin_move",
        "vrchat_autonomy_plugin_tick",
        "vrchat_autonomy_plugin_enqueue",
        "vrchat_autonomy_plugin_neuro_status",
        "vrchat_autonomy_plugin_neuro_bootstrap",
        "vrchat_autonomy_plugin_neuro_handle_action",
    }
    assert ctx.tools[0]["toolset"] == "vrchat_autonomy"
    assert ctx.cli_commands[0]["name"] == "vrchat-autonomy"


def test_send_move_unknown_direction():
    movement = load_movement()
    result = movement.send_move("sideways")
    assert result["success"] is False
    assert "unknown_direction" in result["error"]


def test_send_move_forward_pulses_and_resets():
    movement = load_movement()
    calls: list[tuple[str, list]] = []

    def fake_send(address: str, args: list):
        calls.append((address, args))
        return {"success": True}

    with patch("tools.vrchat_osc_tool.vrchat_send_osc", side_effect=fake_send):
        result = movement.send_move("forward", value=1.0, duration_ms=0)

    assert result["success"] is True
    assert calls[0] == ("/input/MoveForward", [1.0])


def load_core():
    spec = importlib.util.spec_from_file_location(
        "vrchat_autonomy_core_test",
        PLUGIN_DIR / "core.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrchat_autonomy_core_test"] = module
    spec.loader.exec_module(module)
    return module


def test_profile_gate_blocks_dry_run():
    core = load_core()
    fake_loaded = {"success": True, "profile": {"dry_run": True, "mode": "private_test"}}
    with patch("tools.openclaw.vrchat_autonomy.load_autonomy_profile", return_value=fake_loaded):
        blocked = core._profile_gate(need_chatbox=True)
    assert blocked is not None
    assert blocked["error"] == "dry_run_enabled"


def load_neuro():
    spec = importlib.util.spec_from_file_location(
        "vrchat_autonomy_neuro_test",
        PLUGIN_DIR / "neuro.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrchat_autonomy_neuro_test"] = module
    spec.loader.exec_module(module)
    return module


def test_neuro_status_reports_vendor_and_game(tmp_path, monkeypatch):
    neuro = load_neuro()
    sdk_path = tmp_path / "neuro-sdk"
    api_path = sdk_path / "API"
    api_path.mkdir(parents=True)
    (api_path / "SPECIFICATION.md").write_text("# Spec\n", encoding="utf-8")
    (sdk_path / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
    monkeypatch.setattr("tools.openclaw.neuro_bridge.NEURO_SDK_PATH", sdk_path)

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        '{"enabled": true, "mode": "private_test", "dry_run": true, '
        '"allow_chatbox": true, "allow_voice": false, "allowed_avatar_actions": []}',
        encoding="utf-8",
    )
    config = {"neuro_game": "Test Game", "neuro_ws_url": "ws://127.0.0.1:9999"}

    result = neuro.neuro_status(profile=profile_path, config=config)

    assert result["ok"] is True
    assert result["game"] == "Test Game"
    assert result["ws_url"] == "ws://127.0.0.1:9999"
    assert result["vendor"]["success"] is True


def test_neuro_bootstrap_includes_register_actions(tmp_path, monkeypatch):
    neuro = load_neuro()
    sdk_path = tmp_path / "neuro-sdk"
    (sdk_path / "API").mkdir(parents=True)
    (sdk_path / "API" / "SPECIFICATION.md").write_text("# Spec\n", encoding="utf-8")
    (sdk_path / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
    monkeypatch.setattr("tools.openclaw.neuro_bridge.NEURO_SDK_PATH", sdk_path)

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        '{"enabled": true, "mode": "private_test", "dry_run": true, '
        '"allow_chatbox": true, "allowed_avatar_actions": ["wave"]}',
        encoding="utf-8",
    )

    result = neuro.neuro_bootstrap(profile=profile_path, context="hello neuro")

    commands = [message["command"] for message in result["messages"]]
    assert commands == ["startup", "context", "actions/register"]
    action_names = [action["name"] for action in result["messages"][2]["data"]["actions"]]
    assert "vrchat_autonomy_turn" in action_names


def test_neuro_handle_action_dry_run_chatbox(tmp_path):
    neuro = load_neuro()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        '{"enabled": true, "mode": "private_test", "dry_run": true, '
        '"allow_chatbox": true, "allow_voice": false, "allowed_avatar_actions": []}',
        encoding="utf-8",
    )
    message = {
        "command": "action",
        "data": {
            "id": "n1",
            "name": "vrchat_chatbox",
            "data": '{"text": "plugin neuro"}',
        },
    }

    result = neuro.neuro_handle_action(message, profile=profile_path)

    assert result["success"] is True
    assert result["turn"]["dry_run"] is True
    assert result["turn"]["planned_actions"][0]["kind"] == "chatbox"


def test_status_includes_neuro_readiness(monkeypatch, tmp_path):
    core = load_core()
    fake_loaded = {"success": True, "profile": {"dry_run": True, "mode": "observe"}}
    fake_readiness = {"ready": True}
    monkeypatch.setattr(core, "plugin_config", lambda: {"neuro_game": "Hermes VRChat"})
    monkeypatch.setattr(core, "profile_path", lambda _cfg=None: tmp_path / "profile.json")
    monkeypatch.setattr(core, "worker_status", lambda: {"running": False})
    monkeypatch.setattr(core, "check_available", lambda: True)
    with patch("tools.openclaw.vrchat_autonomy.load_autonomy_profile", return_value=fake_loaded):
        with patch("tools.openclaw.vrchat_autonomy.vrchat_autonomy_readiness", return_value=fake_readiness):
            payload = core.status()

    assert "neuro" in payload
    assert "neuro_readiness" in payload
    assert payload["neuro"]["game"] == "Hermes VRChat"


def test_neuro_readiness_reports_missing_vendor(monkeypatch, tmp_path):
    neuro = load_neuro()
    missing = tmp_path / "missing-neuro-sdk"
    monkeypatch.setattr("tools.openclaw.neuro_bridge.NEURO_SDK_PATH", missing)

    result = neuro.neuro_readiness({"neuro_game": "Hermes VRChat"})

    assert result["vendor_ok"] is False
    assert "neuro-sdk" in result["hint"]


def test_neuro_vendor_status_includes_init_command(monkeypatch, tmp_path):
    neuro = load_neuro()
    monkeypatch.setattr("tools.openclaw.neuro_bridge.NEURO_SDK_PATH", tmp_path / "empty")

    result = neuro.neuro_vendor_status()

    assert result["ok"] is False
    assert "git submodule update --init vendor/neuro-sdk" in result["init_command"]


def test_neuro_build_messages_appends_force(tmp_path, monkeypatch):
    neuro = load_neuro()
    sdk_path = tmp_path / "neuro-sdk"
    (sdk_path / "API").mkdir(parents=True)
    (sdk_path / "API" / "SPECIFICATION.md").write_text("# Spec\n", encoding="utf-8")
    (sdk_path / "LICENSE.md").write_text("MIT License\n", encoding="utf-8")
    monkeypatch.setattr("tools.openclaw.neuro_bridge.NEURO_SDK_PATH", sdk_path)

    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        '{"enabled": true, "mode": "private_test", "dry_run": true, '
        '"allow_chatbox": true, "allowed_avatar_actions": []}',
        encoding="utf-8",
    )

    result = neuro.neuro_build_messages(
        profile=profile_path,
        force_action_names=["vrchat_autonomy_turn"],
        force_query="say hello",
        force_state="idle",
    )

    commands = [message["command"] for message in result["messages"]]
    assert commands[-1] == "actions/force"
    assert result["messages"][-1]["data"]["query"] == "say hello"


def test_doctor_core_ok_without_neuro_vendor(monkeypatch, tmp_path):
    core = load_core()
    fake_loaded = {"success": True, "profile": {"dry_run": True, "mode": "observe", "allow_voice": False}}
    fake_readiness = {"ready": True}
    monkeypatch.setattr(core, "check_available", lambda: True)
    monkeypatch.setattr(core, "plugin_config", lambda: {})
    monkeypatch.setattr(core, "profile_path", lambda _cfg=None: tmp_path / "profile.json")
    monkeypatch.setattr(core, "worker_status", lambda: {"running": False})
    monkeypatch.setattr(
        "tools.openclaw.neuro_bridge.neuro_sdk_vendor_status",
        lambda: {"success": False, "path": str(tmp_path / "neuro-sdk")},
    )
    with patch("tools.openclaw.vrchat_autonomy.load_autonomy_profile", return_value=fake_loaded):
        with patch("tools.openclaw.vrchat_autonomy.vrchat_autonomy_readiness", return_value=fake_readiness):
            with patch("tools.openclaw.vrchat_preflight.build_preflight_bundle", return_value={"success": True}):
                payload = core.doctor()

    assert payload["core_ok"] is True
    assert payload["ok"] is True
    assert payload["neuro_ready"] is False
    assert payload["neuro_bridge_ready"] is False
