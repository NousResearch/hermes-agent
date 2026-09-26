"""Startup regressions exercised against real source trees on the current host.

Source freshness itself is the compiler's receipt (scripts/build/freshness.mjs), covered in tests-js."""

import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop as desktop


def _tree(tmp_path):
    root = tmp_path / "checkout"
    app = root / "apps" / "desktop"
    app.mkdir(parents=True)
    (root / ".gitignore").write_text("dist/\nrelease/\nnode_modules/\n")
    (root / "package.json").write_text("{}")
    (app / "package.json").write_text("{}")
    source = app / "app.ts"
    source.write_text("export const x = 1")
    return root, app, source


def test_current_packaged_launch_does_not_require_npm(tmp_path, monkeypatch):
    root, app, _ = _tree(tmp_path)
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    executable = app / "Hermes.exe"
    args = argparse.Namespace(source=False, skip_build=False, force_build=False, build_only=False)
    with patch.object(desktop, "_desktop_launch_env", return_value=({}, [])), \
         patch.object(desktop, "_desktop_packaged_executable", return_value=executable), \
         patch.object(desktop, "_desktop_build_needed", return_value=False), \
         patch.object(desktop, "_register_linux_desktop_entry"), \
         patch.object(desktop, "_packaged_desktop_launch_command", return_value=[str(executable)]), \
         patch("hermes_cli.main_install_repair._resolve_node_runtime_npm", return_value=None) as npm, \
         patch.object(desktop.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run, \
         pytest.raises(SystemExit) as exited:
        desktop.cmd_gui(args)
    assert exited.value.code == 0
    npm.assert_not_called()
    run.assert_called_once()

