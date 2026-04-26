"""Tests for memory plugin CLI command discovery."""

from __future__ import annotations


def test_discovers_cli_always_memory_plugin_without_active_provider(monkeypatch, tmp_path):
    import plugins.memory as memory_plugins

    provider_dir = tmp_path / "agentmemory"
    provider_dir.mkdir()
    (provider_dir / "__init__.py").write_text("from agent.memory_provider import MemoryProvider\n")
    (provider_dir / "plugin.yaml").write_text(
        "name: agentmemory\n"
        "description: AgentMemory commands\n"
        "cli_always: true\n"
    )
    (provider_dir / "cli.py").write_text(
        "def register_cli(subparser):\n"
        "    subparser.set_defaults(func=agentmemory_command)\n"
        "def agentmemory_command(args):\n"
        "    return None\n"
    )

    monkeypatch.setattr(memory_plugins, "_MEMORY_PLUGINS_DIR", tmp_path)
    monkeypatch.setattr(memory_plugins, "_get_active_memory_provider", lambda: None)

    commands = memory_plugins.discover_plugin_cli_commands()

    assert [cmd["name"] for cmd in commands] == ["agentmemory"]
    assert commands[0]["help"] == "AgentMemory commands"
