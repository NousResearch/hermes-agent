"""Integration coverage for loading the feedback plugin through Hermes discovery."""

from __future__ import annotations

import builtins
import shutil
import sys
from pathlib import Path


def test_discovery_registers_command_without_host_github_identity_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """An external install owns its identity boundary and does not require a fork-only core module."""

    from hermes_cli.plugins import PluginManager

    hermes_home = tmp_path / ".hermes"
    installed = hermes_home / "plugins" / "github-pr-feedback"
    shutil.copytree(
        Path(__file__).resolve().parents[1],
        installed,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "tests"),
    )
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled: [github-pr-feedback]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    plugin_source = str(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != plugin_source])
    for module_name in tuple(sys.modules):
        if module_name == "hermes_cli.github_identity" or module_name.startswith(
            ("github_pr_feedback", "hermes_plugins.github_pr_feedback")
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    real_import = builtins.__import__

    def import_without_fork_identity(name, *args, **kwargs):
        if name == "hermes_cli.github_identity":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fork_identity)

    manager = PluginManager(scope_key=str(hermes_home))
    manager.discover_and_load()

    loaded = manager._plugins["github-pr-feedback"]
    assert loaded.enabled, loaded.error
    assert "github-pr-feedback" in manager._cli_commands
