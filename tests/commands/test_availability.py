"""Availability and dependency contracts for the canonical command owner."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from commands import (
    COMMAND_REGISTRY, VALID_BUSY_POLICIES, available_commands,
    command_desktop_meta, is_gateway_available, resolve_command,
)


def test_names_and_aliases_share_one_unambiguous_lookup():
    keys = [key for cmd in COMMAND_REGISTRY for key in (cmd.name, *cmd.aliases)]
    assert len(keys) == len(set(keys))
    for cmd in COMMAND_REGISTRY:
        for key in (cmd.name, *cmd.aliases):
            assert resolve_command("/" + key.upper()) is cmd
        assert cmd.busy_policy in VALID_BUSY_POLICIES
        assert command_desktop_meta(cmd)["desktop"] == cmd.desktop


def test_availability_uses_only_supplied_gates():
    gated = [cmd for cmd in COMMAND_REGISTRY if cmd.cli_only and cmd.gateway_config_gate]
    assert gated
    for cmd in gated:
        assert not is_gateway_available(cmd)
        assert is_gateway_available(cmd, {cmd.name})
        assert not is_gateway_available(cmd, {cmd.gateway_config_gate})
        assert resolve_command(cmd.name) is cmd  # discovery never changes identity/dispatch
    gates = {cmd.name for cmd in gated}
    assert available_commands("gateway", enabled_config_gates=iter(gates)) == tuple(
        cmd for cmd in COMMAND_REGISTRY if not cmd.cli_only or cmd.name in gates
    )
    assert available_commands("cli") == available_commands("desktop") == tuple(
        cmd for cmd in COMMAND_REGISTRY if not cmd.gateway_only
    )
    with pytest.raises(ValueError):
        available_commands("unknown")


def test_import_and_builtin_queries_need_no_cli_or_discovery():
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc, sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'hermes_cli', 'gateway', 'agent', 'model_tools', 'prompt_toolkit', 'utils'} or fullname == 'plugin_runtime.api':
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Boundary())
import commands
from plugin_runtime.host_bindings import get_plugin_host_callback
assert get_plugin_host_callback('command_resolver') is commands.resolve_command
assert commands.resolve_command('/Q').name == 'queue'
assert commands.available_commands('gateway')
assert commands.desktop_surface_registry()
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gateway_configuration_gates_follow_current_profile(tmp_path):
    from gateway.command_presentation import gateway_help_lines, resolve_config_gates
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    gated = [cmd for cmd in COMMAND_REGISTRY if cmd.gateway_config_gate]
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, enabled in zip(homes, (True, False)):
        home.mkdir()
        config = {}
        for cmd in gated:
            parts = cmd.gateway_config_gate.split(".")
            node = config
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = enabled
        (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    for home, enabled in ((homes[0], True), (homes[1], False), (homes[0], True)):
        token = set_hermes_home_override(home)
        try:
            gates = resolve_config_gates()
            assert gates == ({cmd.name for cmd in gated} if enabled else set())
            for cmd in gated:
                assert is_gateway_available(cmd, gates) == (not cmd.cli_only or enabled)
            lines = gateway_help_lines(allowed={cmd.name for cmd in gated})
            assert len(lines) == sum(is_gateway_available(cmd, gates) for cmd in gated)
        finally:
            reset_hermes_home_override(token)


def test_dynamic_plugin_metadata_is_lazy_and_not_a_builtin_registry(monkeypatch):
    import plugin_runtime.api
    from commands import is_gateway_known_command, plugin_command_entries

    calls = []
    def published():
        calls.append(True)
        return {"phase7-example": {"description": "Example", "args_hint": " <value> "}}
    monkeypatch.setattr(plugin_runtime.api, "get_plugin_commands", published)
    assert resolve_command("phase7-example") is None
    assert not calls
    assert plugin_command_entries() == [("phase7-example", "Example", "<value>")]
    assert is_gateway_known_command("phase7-example")
    assert resolve_command("phase7-example") is None
