"""Tests for the detached-dashboard-action interpreter choice (#90026).

Under an SSH remote backend the web server runs on the **uv base
interpreter** with the venv's site-packages injected into ``sys.path`` at
startup. A detached action spawned from ``sys.executable`` (the base
interpreter) inherits no injected path and no PYTHONPATH, so it dies on the
first third-party import. The spawner must prefer the install's own venv
interpreter when it differs from ``sys.executable``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import hermes_cli.web_server as web_server
import hermes_cli.web_server_gateway as _web_server_gateway


class TestDashboardSpawnExecutable:
    def test_same_interpreter_returns_sys_executable(self, tmp_path):
        """When sys.executable IS the venv python (normal launch), the
        behavior is unchanged — same path back, verbatim."""
        fake_venv = tmp_path / "venv" / "bin" / "python"
        fake_venv.parent.mkdir(parents=True)
        fake_venv.touch()
        with (
            patch.object(web_server, "PROJECT_ROOT", tmp_path),
            patch.object(sys, "executable", str(fake_venv)),
        ):
            assert _web_server_gateway._dashboard_spawn_executable() == str(fake_venv)

    def test_base_interpreter_replaced_by_venv_python(self, tmp_path):
        """sys.executable pointing at the dependency-less uv base
        interpreter (SSH remote backend) resolves to the install's venv
        python instead (#90026)."""
        fake_venv = tmp_path / "venv" / "bin" / "python"
        fake_venv.parent.mkdir(parents=True)
        fake_venv.touch()
        base_interp = tmp_path / "uv-base" / "python"
        with (
            patch.object(web_server, "PROJECT_ROOT", tmp_path),
            patch.object(sys, "executable", str(base_interp)),
        ):
            chosen = _web_server_gateway._dashboard_spawn_executable()
        assert chosen == str(fake_venv)

    def test_windows_layout_resolved(self, tmp_path):
        """The Windows venv layout (Scripts/python.exe) is honored."""
        fake_venv = tmp_path / "venv" / "Scripts" / "python.exe"
        fake_venv.parent.mkdir(parents=True)
        fake_venv.touch()
        base_interp = tmp_path / "uv-base" / "python.exe"
        with (
            patch.object(web_server, "PROJECT_ROOT", tmp_path),
            patch.object(sys, "executable", str(base_interp)),
        ):
            chosen = _web_server_gateway._dashboard_spawn_executable()
        assert Path(chosen).name == "python.exe"
        assert "Scripts" in chosen

    def test_no_venv_falls_back_to_sys_executable(self, tmp_path):
        """Exotic layouts without an install venv keep the old behavior."""
        base_interp = tmp_path / "uv-base" / "python"
        with (
            patch.object(web_server, "PROJECT_ROOT", tmp_path),
            patch.object(sys, "executable", str(base_interp)),
        ):
            assert _web_server_gateway._dashboard_spawn_executable() == str(base_interp)

    def test_venv_symlink_to_base_is_still_preferred_unresolved(self, tmp_path):
        """The Linux-standard layout: venv/bin/python is a SYMLINK to the
        base interpreter. The chooser must return the UNRESOLVED venv path —
        resolving it would compare equal to the base interpreter (missing
        the swap) or spawn the base directly (bypassing pyvenv.cfg). This is
        the exact layout of the #90026 report."""
        base = tmp_path / "uv-base" / "python"
        base.parent.mkdir(parents=True)
        base.touch()
        venv_py = tmp_path / "venv" / "bin" / "python"
        venv_py.parent.mkdir(parents=True)
        venv_py.symlink_to(base)
        with (
            patch.object(web_server, "PROJECT_ROOT", tmp_path),
            patch.object(sys, "executable", str(base)),
        ):
            chosen = _web_server_gateway._dashboard_spawn_executable()
        assert chosen == str(venv_py), (
            "must return the unresolved venv path, not the symlink target"
        )
