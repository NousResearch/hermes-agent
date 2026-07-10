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
    def test_no_action_when_all_deps_present(self, temp_pyproject, fake_venv_python):
        """The happy path: nothing missing, no repair install fires."""
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}

        with patch("hermes_cli.main._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main.subprocess.run") as mock_run, \
             patch("hermes_cli.main._run_install_with_heartbeat") as mock_install:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

            # Probe ran, repair install did not.
            assert mock_run.called, "verification probe should have run"
            assert not mock_install.called, "repair install should not fire when nothing is missing"

    def test_triggers_reinstall_when_dep_missing(self, temp_pyproject, fake_venv_python):
        """The regression: one base dep is missing → trigger --reinstall."""
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}

        # First probe reports pathspec missing; after repair, probe is clean.
        probe_calls = {"count": 0}

        def fake_subprocess_run(cmd, **kwargs):
            # The probe subprocess returns stdout = newline-joined missing names.
            probe_calls["count"] += 1
            if probe_calls["count"] == 1:
                return MagicMock(returncode=0, stdout="pathspec\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main.subprocess.run", side_effect=fake_subprocess_run), \
             patch("hermes_cli.main._run_install_with_heartbeat") as mock_install:

            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

            assert mock_install.called, "repair install must fire when a dep is missing"
            # First repair must use --reinstall to force re-resolution.
            first_repair = mock_install.call_args_list[0]
            args = first_repair[0][0]  # positional: install_cmd
            assert "--reinstall" in args, f"repair install should pass --reinstall, got {args}"
            assert "-e" in args and "." in args, "first repair should be base group reinstall"

    def test_falls_back_to_per_package_install_when_reinstall_did_not_help(
        self, temp_pyproject, fake_venv_python
    ):
        """If --reinstall doesn't repair the partial install (uv resolver
        thinks the env is satisfied), force-install each missing dep with
        its declared pin spec. This is the last-ditch path."""
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}

        probe_calls = {"count": 0}

        def fake_subprocess_run(cmd, **kwargs):
            probe_calls["count"] += 1
            # 1st probe: pathspec missing
            # 2nd probe (after --reinstall): still missing
            # 3rd probe (after per-package): clean
            if probe_calls["count"] in (1, 2):
                return MagicMock(returncode=0, stdout="pathspec\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main.subprocess.run", side_effect=fake_subprocess_run), \
             patch("hermes_cli.main._run_install_with_heartbeat") as mock_install:

            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

            assert mock_install.call_count >= 2, (
                "expected at least 2 repair installs (reinstall + per-package), "
                f"got {mock_install.call_count}"
            )
            # Second repair call should pass the pinned spec for the missing dep.
            second_repair_args = mock_install.call_args_list[1][0][0]
            assert any("pathspec==1.1.1" in str(a) for a in second_repair_args), (
                f"second repair should pin pathspec from pyproject; got {second_repair_args}"
            )

    def test_skips_deps_excluded_by_environment_markers(self, temp_pyproject, fake_venv_python):
        """``ptyprocess ; sys_platform != 'win32'`` should NOT be reported as
        missing on Windows. Without marker evaluation, the verification step
        would false-positive on every cross-platform exclusion and chase its
        tail forever trying to install something that can't apply here."""
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}
        captured_argv: list[list[str]] = []

        def fake_subprocess_run(cmd, **kwargs):
            captured_argv.append(list(cmd))
            return MagicMock(returncode=0, stdout="", stderr="")

        # Force sys.platform to look like Windows so the marker filters
        # ptyprocess out. (We need the actual marker.evaluate() to see win32.)
        with patch("hermes_cli.main._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main.subprocess.run", side_effect=fake_subprocess_run), \
             patch("hermes_cli.main._run_install_with_heartbeat"), \
             patch("sys.platform", "win32"):

            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

        # Find the probe argv — it's the call that passed the dep names.
        probe = next(
            (argv for argv in captured_argv if any("importlib.metadata" in str(a) for a in argv)),
            None,
        )
        assert probe is not None, "verification probe should have run"
        # The dep names are tacked on after the -c script.
        assert "ptyprocess" not in probe, (
            "ptyprocess is gated by sys_platform != 'win32' and must be filtered "
            f"out on Windows; full probe argv was: {probe}"
        )
        assert any(str(a) == "pathspec" or str(a).startswith("pathspec==") for a in probe), (
            "core deps without markers must be checked (bare name or exact pin); "
            f"full probe argv was: {probe}"
        )

    def test_no_pyproject_is_noop(self, tmp_path, monkeypatch):
        """If pyproject.toml is missing (unusual but possible in some test
        envs), the verification step must short-circuit, not crash."""
        import hermes_cli.main as main_mod
        monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
        # No pyproject.toml in tmp_path.
        with patch("hermes_cli.main._resolve_install_target_python") as mock_resolve, \
             patch("hermes_cli.main._run_install_with_heartbeat") as mock_install:
            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env={})
            assert not mock_resolve.called
            assert not mock_install.called

    def test_repair_reinstall_quarantines_running_shim_on_windows(
        self, temp_pyproject, fake_venv_python
    ):
        """Regression: the ``--reinstall -e .`` repair must
        quarantine the running ``hermes.exe`` on Windows before installing.

        That reinstall rewrites the editable entry-point shims, and on Windows
        pip can't overwrite the live launcher — so without quarantine the shim
        is left missing and ``hermes`` drops off PATH. Previously this path
        called ``_run_install_with_heartbeat`` directly, bypassing the
        quarantine that the primary install path performs.
        """
        py, venv_root = fake_venv_python
        env = {"VIRTUAL_ENV": str(venv_root)}

        probe_calls = {"count": 0}

        def fake_subprocess_run(cmd, **kwargs):
            probe_calls["count"] += 1
            # 1st probe: pathspec missing → triggers --reinstall repair.
            # 2nd probe (after repair): clean → stops before per-package path.
            if probe_calls["count"] == 1:
                return MagicMock(returncode=0, stdout="pathspec\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")

        fake_scripts = venv_root / "Scripts"  # created by fake_venv_python

        with patch("hermes_cli.main._resolve_install_target_python", return_value=py), \
             patch("hermes_cli.main.subprocess.run", side_effect=fake_subprocess_run), \
             patch("hermes_cli.main._is_windows", return_value=True), \
             patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts), \
             patch("hermes_cli.main._run_install_with_heartbeat"), \
             patch("hermes_cli.main._quarantine_running_hermes_exe", return_value=[]) as mock_quar:

            from hermes_cli.main import _verify_core_dependencies_installed
            _verify_core_dependencies_installed(["uv", "pip"], env=env)

            assert mock_quar.called, (
                "the --reinstall -e . repair must quarantine the running "
                "hermes.exe on Windows"
            )
            assert mock_quar.call_args[0][0] == fake_scripts


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

        with patch("hermes_cli.main._is_windows", return_value=True):
            from hermes_cli.main import _resolve_install_target_python
            result = _resolve_install_target_python(
                ["uv", "pip"], env={"VIRTUAL_ENV": str(venv_root)}
            )
            assert result == py

    def test_returns_none_when_venv_python_missing(self, tmp_path):
        """If the path we'd point at doesn't exist (uv install failed before
        the python shim landed), return None so the verification step
        cleanly short-circuits instead of crashing on FileNotFoundError."""
        with patch("hermes_cli.main._is_windows", return_value=True):
            from hermes_cli.main import _resolve_install_target_python
            result = _resolve_install_target_python(
                ["uv", "pip"], env={"VIRTUAL_ENV": str(tmp_path / "does_not_exist")}
            )
            assert result is None
