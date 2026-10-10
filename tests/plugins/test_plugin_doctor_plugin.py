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


def test_scan_plugins_accepts_manifest_without_kind(tmp_path: Path) -> None:
    """The loader defaults a missing kind to standalone, so the scan must too."""
    _write_plugin(
        tmp_path,
        "kindless",
        """
name: kindless
version: 0.1.0
description: Plugin without an explicit kind.
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["ok"] is True
    assert payload["plugins"][0]["kind"] == "standalone"
    assert payload["plugins"][0]["manifest_ok"] is True


def test_scan_plugins_warns_on_unknown_kind(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "weird_kind",
        """
name: weird-kind
version: 0.1.0
description: Plugin with a bogus kind.
kind: not-a-kind
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    entry = payload["plugins"][0]
    assert entry["kind"] == "standalone"
    assert entry["manifest_ok"] is True
    assert any("unknown kind" in warning for warning in entry["warnings"])


def test_scan_plugins_descends_into_category_directories(tmp_path: Path) -> None:
    """<root>/<category>/<plugin>/plugin.yaml must be discovered, not flagged."""
    _write_plugin(
        tmp_path,
        "image_gen/openai",
        """
name: openai
version: 0.1.0
description: Category-nested backend.
kind: backend
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["ok"] is True
    assert payload["plugin_count"] == 1
    assert payload["plugins"][0]["name"] == "openai"


def test_scan_plugins_stops_at_category_depth_cap(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "a/b/c",
        """
name: toodeep
version: 0.1.0
description: Beyond the loader depth cap.
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["plugin_count"] == 0


def test_scan_plugins_does_not_execute_plugin_code_by_default(tmp_path: Path) -> None:
    """The default scan is read-only: module-level side effects must not run."""
    marker = tmp_path / "side_effect.txt"
    _write_plugin(
        tmp_path,
        "sideeffect",
        """
name: sideeffect
version: 0.1.0
description: Writes a marker at import time.
""".strip(),
        init=(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "def register(ctx):\n"
            "    pass\n"
        ),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    assert payload["plugins"][0]["import"]["mode"] == "static"
    assert payload["plugins"][0]["import"]["ok"] is True
    assert not marker.exists()


def test_scan_plugins_executes_only_when_unsafe_opt_in(tmp_path: Path) -> None:
    marker = tmp_path / "side_effect.txt"
    _write_plugin(
        tmp_path,
        "sideeffect",
        """
name: sideeffect
version: 0.1.0
description: Writes a marker at import time.
""".strip(),
        init=(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "def register(ctx):\n"
            "    pass\n"
        ),
    )

    payload = core.scan_plugins(
        {"plugins_dir": str(tmp_path), "unsafe_execute_import_check": True}
    )

    assert payload["plugins"][0]["import"]["mode"] == "execute"
    assert marker.read_text(encoding="utf-8") == "executed"


def test_scan_plugins_flags_unparseable_manifest(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "malformed",
        "name: [unclosed\n  version: 0.1.0\n",
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    entry = payload["plugins"][0]
    assert entry["manifest_ok"] is False
    assert any("failed to parse" in error for error in entry["errors"])


def test_scan_plugins_flags_invalid_list_field(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path,
        "badlist",
        """
name: badlist
version: 0.1.0
description: provides_tools is not a list.
provides_tools: not-a-list
""".strip(),
    )

    payload = core.scan_plugins({"plugins_dir": str(tmp_path)})

    entry = payload["plugins"][0]
    assert entry["manifest_ok"] is False
    assert "provides_tools must be a list" in entry["errors"]
