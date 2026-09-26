from __future__ import annotations

import json
from pathlib import Path

from plugins.plugin_doctor import core, register


class _FakeContext:
    def __init__(self) -> None:
        self.tools = {}
        self.commands = {}
        self.cli_commands = {}

    def register_tool(self, name, **kwargs):
        self.tools[name] = kwargs

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs


def _write_plugin(root: Path, name: str, manifest: str, init: str = "def register(ctx):\n    pass\n") -> None:
    plugin = root / name
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(manifest, encoding="utf-8")
    (plugin / "__init__.py").write_text(init, encoding="utf-8")


def test_registers_tool_slash_and_cli_command() -> None:
    ctx = _FakeContext()
    register(ctx)

    assert "plugin_doctor_scan" in ctx.tools
    assert "plugin-doctor" in ctx.commands
    assert "plugin-doctor" in ctx.cli_commands


def test_scan_plugins_reports_valid_plugin(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "demo_plugin",
        """
name: demo-plugin
version: 0.1.0
description: Demo plugin.
kind: standalone
provides_tools:
  - demo_tool
provides_cli:
  - demo
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["ok"] is True
    assert payload["plugin_count"] == 1
    assert payload["plugins"][0]["import"]["ok"] is True


def test_scan_plugins_flags_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["ok"] is False
    assert "missing plugin.yaml" in payload["plugins"][0]["errors"]


def test_scan_plugins_flags_duplicate_tools(tmp_path: Path) -> None:
    manifest = """
name: {name}
version: 0.1.0
description: Demo plugin.
kind: standalone
provides_tools:
  - duplicate_tool
""".strip()
    _write_plugin(tmp_path, "plugin_a", manifest.format(name="plugin-a"))
    _write_plugin(tmp_path, "plugin_b", manifest.format(name="plugin-b"))

    payload = core.scan_plugins({"plugins_dir": str(tmp_path), "include_import_check": False})

    assert payload["ok"] is False
    assert payload["tool_conflicts"] == {"duplicate_tool": ["plugin_a", "plugin_b"]}


def test_handle_slash_returns_json(tmp_path: Path) -> None:
    result = json.loads(core.handle_slash(str(tmp_path)))

    assert result["ok"] is True
    assert result["plugins_dir"] == str(tmp_path)
