"""Regression tests for plugin CLI names colliding with builtin subcommands."""

from __future__ import annotations

import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _hermes_env(hermes_home: Path) -> dict[str, str]:
    return {**os.environ, "HERMES_HOME": str(hermes_home)}


def _enable_plugin(hermes_home: Path, name: str) -> None:
    cfg_path = hermes_home / "config.yaml"
    cfg: dict = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    plugins_cfg = cfg.setdefault("plugins", {})
    enabled = plugins_cfg.setdefault("enabled", [])
    if isinstance(enabled, list) and name not in enabled:
        enabled.append(name)
    cfg_path.write_text(yaml.safe_dump(cfg))


def test_register_cli_command_rejects_builtin_name():
    mgr = PluginManager()
    manifest = PluginManifest(name="conflict-test", version="0.0.1", description="test")
    ctx = PluginContext(manifest, mgr)

    def _setup(parser: ArgumentParser) -> None:
        parser.add_argument("--noop", action="store_true")

    ctx.register_cli_command(
        name="dashboard",
        help="should not register",
        setup_fn=_setup,
        handler_fn=lambda _args: 0,
    )

    assert "dashboard" not in mgr._cli_commands


def test_main_survives_conflicting_plugin_cli_name(tmp_path, monkeypatch):
    """Plugin discovery on non-builtin argv must not break core ``dashboard``."""
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    plugins_dir = hermes_home / "plugins"
    plugin_dir = plugins_dir / "cli-conflict-plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.dump(
            {
                "name": "cli-conflict-plugin",
                "version": "0.0.1",
                "description": "builtin CLI name collision probe",
            }
        )
    )
    (plugin_dir / "__init__.py").write_text(
        "\n".join(
            [
                "def register(ctx):",
                "    def _setup(parser):",
                "        sub = parser.add_subparsers(dest='action', required=True)",
                "        sub.add_parser('probe')",
                "    ctx.register_cli_command(",
                "        name='dashboard',",
                "        help='conflict probe',",
                "        setup_fn=_setup,",
                "        handler_fn=lambda args: 0,",
                "    )",
                "    ctx.register_cli_command(",
                "        name='conflict-probe',",
                "        help='non-conflicting probe',",
                "        setup_fn=_setup,",
                "        handler_fn=lambda args: 0,",
                "    )",
            ]
        )
    )
    _enable_plugin(hermes_home, "cli-conflict-plugin")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli", "conflict-probe", "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=_hermes_env(hermes_home),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "usage:" in combined.lower()


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / "plugins" / "desktop-dashboard").is_dir(),
    reason="bundled desktop-dashboard plugin not present",
)
def test_desk_widget_cli_registers(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    _enable_plugin(hermes_home, "desktop-dashboard")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "hermes_cli", "desk-widget", "--help"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
        env=_hermes_env(hermes_home),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "desk-widget" in combined or "start" in combined
