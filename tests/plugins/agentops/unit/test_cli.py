from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace

from plugins.agentops import register
from plugins.agentops.cli import agentops_command, register_cli


def test_doctor_json_is_machine_readable_and_does_not_create_missing_db(tmp_path, capsys):
    config_path = tmp_path / "missing.yaml"
    rc = agentops_command(Namespace(agentops_command="doctor", config=str(config_path), json=True))

    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report["authority_mode"] == "observe_only"
    assert not (tmp_path / "state.db").exists()


def test_plugin_registers_only_operator_cli_surface():
    class RecordingContext:
        def __init__(self):
            self.commands = []
            self.hooks = []
            self.tools = []

        def register_cli_command(self, **kwargs):
            self.commands.append(kwargs["name"])

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

        def register_tool(self, *args, **kwargs):
            self.tools.append((args, kwargs))

    ctx = RecordingContext()
    register(ctx)

    assert ctx.commands == ["agentops"]
    assert ctx.hooks == []
    assert ctx.tools == []


def test_opted_in_plugin_discovery_exposes_agentops_cli(monkeypatch):
    import hermes_cli.plugins as plugin_system

    monkeypatch.setattr(plugin_system, "_get_enabled_plugins", lambda: {"agentops"})
    try:
        plugin_system.discover_plugins(force=True)
        manager = plugin_system.get_plugin_manager()
        assert manager._cli_commands["agentops"]["name"] == "agentops"
        assert manager._plugins["agentops"].hooks_registered == []
        assert manager._plugins["agentops"].tools_registered == []
    finally:
        plugin_system._plugin_manager = None


def test_cli_parser_exposes_only_daemon_and_doctor(tmp_path):
    parser = ArgumentParser()
    register_cli(parser)

    args = parser.parse_args(["doctor", "--json", "--config", str(tmp_path / "agentops.yaml")])

    assert args.agentops_command == "doctor"
    assert args.json is True
    assert args.config == tmp_path / "agentops.yaml"
