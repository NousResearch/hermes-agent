"""``hermes plugins validate`` warns (never fails) on imports of private Hermes core modules."""

from __future__ import annotations

from pathlib import Path

import hermes_yaml as yaml

from hermes_cli.plugin_validate import validate_plugin_dir


def _plugin(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "plugin.yaml").write_text(yaml.safe_dump({"name": root.name, "version": "1.0.0", "description": "x"}),
                                      encoding="utf-8")
    files = {"__init__.py": "def register(ctx):\n    pass\n", **files}
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _private_warnings(report) -> list[str]:
    return [w for w in report.warnings if w.startswith("private core imports")]


def test_private_core_import_warns_with_file_and_line_but_validate_passes(tmp_path):
    plugin = _plugin(tmp_path / "router", {"lib/agent_runner.py": (
        "import json\n"
        "def run():\n"
        "    from tools.tirith_security import ensure_installed\n"
        "    from hermes_cli.config import _private_helper\n"
        "    return ensure_installed, _private_helper\n"
    )})

    report = validate_plugin_dir(plugin)

    assert report.ok, report.failures
    [warning] = _private_warnings(report)
    assert "lib/agent_runner.py:3 tools.tirith_security.ensure_installed" in warning
    assert "lib/agent_runner.py:4 hermes_cli.config._private_helper" in warning
    assert "catalog rule 9" in warning


def test_documented_public_stdlib_and_own_package_imports_are_not_flagged(tmp_path):
    plugin = _plugin(tmp_path / "clean", {
        "__init__.py": (
            "import os, json\n"
            "from typing import TYPE_CHECKING\n"
            "import requests\n"
            "from agent.memory_provider import MemoryProvider\n"  # documented provider base class
            "from gateway.platforms import base\n"  # documented module, imported from its package
            "from hermes_cli.plugins import PluginContext\n"  # what register() receives
            "from clean_lib.helpers import local_helper\n"  # the plugin's own package
            "from .client import Client\n"
            "if TYPE_CHECKING:\n"
            "    from run_agent import AIAgent\n"
            "def register(ctx):\n"
            "    from hermes_cli.plugins import get_plugin_manager\n"  # same module, undocumented name
            "    return get_plugin_manager\n"
        ),
        "clean_lib/__init__.py": "",
        "clean_lib/helpers.py": "def local_helper():\n    pass\n",
        "client.py": "class Client:\n    pass\n",
    })

    [warning] = _private_warnings(validate_plugin_dir(plugin))

    sites = warning.split("release): ", 1)[1].split(". Use ctx.*")[0].split("; ")
    assert sites == ["__init__.py:12 hermes_cli.plugins.get_plugin_manager"], sites
