"""Tests for _verify_core_dependencies_installed.

Regression coverage for the partial-install bug where uv's incremental
resolver silently failed to land ``pathspec`` (and similar newly-added
base deps) during ``hermes update``, leaving the venv in a broken state
that only surfaced hours later when a downstream subprocess imported the
missing module.

The verification step:
  1. Reads pyproject.toml's [project.dependencies] directly.
  2. Filters by environment markers so cross-platform exclusions don't
     false-positive (e.g. ``ptyprocess ; sys_platform != 'win32'`` on Windows).
  3. Probes ``importlib.metadata.version()`` in the venv interpreter.
  4. Reinstalls with --reinstall, then per-package, if anything's missing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_pyproject(tmp_path, monkeypatch):
    """Point hermes_cli.main.PROJECT_ROOT at a tmp dir with a minimal pyproject.

    The verification helper opens ``PROJECT_ROOT / 'pyproject.toml'`` directly;
    redirecting PROJECT_ROOT keeps the test hermetic.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(textwrap.dedent("""\
        [project]
        name = "fake"
        version = "0.0.0"
        dependencies = [
          "pathspec==1.1.1",
          "pydantic==2.13.4",
          "ptyprocess>=0.7.0,<1; sys_platform != 'win32'",
          "tzdata>=2024.1; sys_platform == 'win32'",
        ]
    """))
    import hermes_cli.main as main_mod
    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_venv_python(tmp_path):
    """Create a fake venv python shim path that exists on disk."""
    venv_root = tmp_path / "venv"
    scripts = venv_root / "Scripts"
    scripts.mkdir(parents=True)
    py = scripts / "python.exe"
    py.write_text("#!/bin/sh\necho fake python")
    return py, venv_root


class TestVerifyCoreDependencies:



    def test_skips_deps_excluded_by_environment_markers(self, temp_pyproject, fake_venv_python):
        """A dep whose ``sys_platform`` marker excludes THIS host must not be
        probed (and so never reported missing). Without marker evaluation the
        verification step would false-positive on every cross-platform
        exclusion and chase its tail installing something inapplicable here.

        Deliberately host-invariant rather than ``windows_only``: the subject
        is ``packaging``'s marker *evaluation*, not any OS facility. The
        pyproject fixture declares one dep gated to non-Windows and one gated
        to Windows, so exactly one of the pair is filtered on any host — the
        old ``patch("sys.platform", "win32")`` bought nothing but a fake host.
        """
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}
        captured_argv: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_argv.append(list(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main_install_repair._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main_install_repair.subprocess.run", side_effect=fake_subprocess_run), \
             patch("hermes_cli.main_install_repair._run_install_with_heartbeat"):

            from hermes_cli.main_install_repair import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

        # Find the probe argv — it's the call that passed the dep names.
        probe = next(
            (argv for argv in captured_argv if any("importlib.metadata" in str(a) for a in argv)),
            None,
        )
        assert probe is not None, "verification probe should have run"
        # The dep names are tacked on after the -c script. Exactly one of the
        # marker-gated pair applies to this host; the other must be filtered.
        on_windows = sys.platform == "win32"
        assert ("ptyprocess" in probe) is not on_windows, (
            "ptyprocess is gated by sys_platform != 'win32', so it must be "
            f"probed off Windows and filtered on it; probe argv was: {probe}"
        )
        assert ("tzdata" in probe) is on_windows, (
            "tzdata is gated by sys_platform == 'win32', so it must be probed "
            f"on Windows and filtered elsewhere; probe argv was: {probe}"
        )
        assert "pathspec" in probe, "core deps without markers must be checked"

    def test_no_pyproject_is_noop(self, tmp_path, monkeypatch):
        """If pyproject.toml is missing (unusual but possible in some test
        envs), the verification step must short-circuit, not crash."""
        import hermes_cli.main as main_mod
        monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
        # No pyproject.toml in tmp_path.
        with patch("hermes_cli.main_install_repair._resolve_install_target_python") as mock_resolve, \
             patch("hermes_cli.main_install_repair._run_install_with_heartbeat") as mock_install:
            from hermes_cli.main_install_repair import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env={})
            assert not mock_resolve.called
            assert not mock_install.called



class TestResolveInstallTargetPython:
    def test_uses_virtual_env_from_environment(self, tmp_path):
        """When VIRTUAL_ENV is set, the verification step must probe THAT
        venv's interpreter — not the outer Python that drove `hermes update`.
        If we probed sys.executable instead, we'd false-positive every dep
        the outer interpreter happens to lack."""
        venv_root = tmp_path / "newvenv"
        scripts = venv_root / "Scripts"
        scripts.mkdir(parents=True)
        py = scripts / "python.exe"
        py.write_text("fake")

        with patch("hermes_cli.main_install_repair._is_windows", return_value=True):
            from hermes_cli.main_install_repair import _resolve_install_target_python
            result = _resolve_install_target_python(
                ["uv", "pip"], env={"VIRTUAL_ENV": str(venv_root)}
            )
            assert result == py

    def test_returns_none_when_venv_python_missing(self, tmp_path):
        """If the path we'd point at doesn't exist (uv install failed before
        the python shim landed), return None so the verification step
        cleanly short-circuits instead of crashing on FileNotFoundError."""
        with patch("hermes_cli.main_install_repair._is_windows", return_value=True):
            from hermes_cli.main_install_repair import _resolve_install_target_python
            result = _resolve_install_target_python(
                ["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path / "does_not_exist")}
            )
            assert result is None
