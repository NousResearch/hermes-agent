"""Tests for _verify_console_scripts_installed (issue #52931)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_pyproject(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        textwrap.dedent(
            """\
        [project]
        name = "fake"
        version = "0.0.0"

        [project.scripts]
        hermes = "hermes_cli.main:main"
        hermes-agent = "run_agent:main"
        hermes-acp = "acp_adapter.entry:main"
    """
        )
    )
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_scripts_dir(tmp_path):
    scripts = tmp_path / "venv" / "Scripts"
    scripts.mkdir(parents=True)
    return scripts


class TestVerifyConsoleScriptsInstalled:
    def test_no_action_when_all_shims_present(self, temp_pyproject, fake_scripts_dir):
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (fake_scripts_dir / f"{name}.exe").write_bytes(b"fake")

        with patch("hermes_cli.main._is_windows", return_value=True), \
             patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts_dir), \
             patch("hermes_cli.main._run_quarantined_install") as mock_install:
            from hermes_cli.main import _verify_console_scripts_installed

            _verify_console_scripts_installed(["uv", "pip"], env={})

        mock_install.assert_not_called()




    def test_quarantine_shims_include_declared_console_scripts(
        self, temp_pyproject, fake_scripts_dir
    ):
        import hermes_cli.main as main_mod

        with patch("hermes_cli.main._is_windows", return_value=True):
            names = {path.name for path in main_mod._hermes_exe_shims(fake_scripts_dir)}

        assert {"hermes.exe", "hermes-agent.exe", "hermes-acp.exe"} <= names
        assert "hermes-gateway.exe" in names


class TestWarnIfHermesLauncherBroken:
    """Tests for _warn_if_hermes_launcher_broken (issue #83529).

    A failed base install on POSIX (permission-denied venv files, most often
    from a prior privileged run) can delete ``venv/bin/hermes`` before
    failing to write its replacement, leaving the CLI unusable with no
    indication why. This warning must fire on every platform, not just
    Windows.
    """

    def test_no_warning_when_all_shims_present(
        self, temp_pyproject, fake_scripts_dir, capsys
    ):
        for name in ("hermes", "hermes-agent", "hermes-acp"):
            (fake_scripts_dir / name).write_bytes(b"fake")

        with patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts_dir):
            from hermes_cli.main import _warn_if_hermes_launcher_broken

            _warn_if_hermes_launcher_broken()

        assert capsys.readouterr().out == ""

    def test_warns_when_hermes_shim_missing_on_posix(
        self, temp_pyproject, fake_scripts_dir, capsys
    ):
        # hermes shim missing entirely; the other two survived.
        (fake_scripts_dir / "hermes-agent").write_bytes(b"fake")
        (fake_scripts_dir / "hermes-acp").write_bytes(b"fake")

        with patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts_dir):
            from hermes_cli.main import _warn_if_hermes_launcher_broken

            _warn_if_hermes_launcher_broken()

        out = capsys.readouterr().out
        assert "broken" in out
        assert "hermes" in out
        chown_line = next(line for line in out.splitlines() if "chown" in line)
        assert chown_line.endswith(f'sudo chown -R "$(id -un)" {fake_scripts_dir.parent}')

    def test_no_action_when_not_running_from_a_venv(self, temp_pyproject, capsys):
        with patch("hermes_cli.main._venv_scripts_dir", return_value=None):
            from hermes_cli.main import _warn_if_hermes_launcher_broken

            _warn_if_hermes_launcher_broken()

        assert capsys.readouterr().out == ""

    def test_warns_with_windows_appropriate_remedy_on_windows(
        self, temp_pyproject, fake_scripts_dir, capsys
    ):
        (fake_scripts_dir / "hermes-agent").write_bytes(b"fake")
        (fake_scripts_dir / "hermes-acp").write_bytes(b"fake")

        with patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts_dir), \
             patch("hermes_cli.main._is_windows", return_value=True):
            from hermes_cli.main import _warn_if_hermes_launcher_broken

            _warn_if_hermes_launcher_broken()

        out = capsys.readouterr().out
        assert "ls -la" not in out
        assert "chown" not in out
        assert "icacls" in out
        assert "takeown" in out


class TestInstallPythonDependenciesWithOptionalFallback:
    """The base-install retry used to be unguarded — a failure here skipped
    straight past every recovery check (#83529)."""

    def test_base_install_failure_warns_before_reraising(
        self, temp_pyproject, fake_scripts_dir, monkeypatch
    ):
        import subprocess

        import hermes_cli.main as main_mod

        (fake_scripts_dir / "hermes-agent").write_bytes(b"fake")
        (fake_scripts_dir / "hermes-acp").write_bytes(b"fake")

        calls: list[list[str]] = []

        def fake_run_quarantined_install(cmd, *, env=None, scripts_dir=None):
            calls.append(cmd)
            raise subprocess.CalledProcessError(2, cmd)

        warned = []
        monkeypatch.setattr(main_mod, "_is_windows", lambda: False)
        monkeypatch.setattr(main_mod, "_venv_scripts_dir", lambda: fake_scripts_dir)
        monkeypatch.setattr(
            main_mod, "_run_quarantined_install", fake_run_quarantined_install
        )
        monkeypatch.setattr(
            main_mod,
            "_warn_if_hermes_launcher_broken",
            lambda: warned.append(True),
        )

        with pytest.raises(subprocess.CalledProcessError):
            main_mod._install_python_dependencies_with_optional_fallback(["uv", "pip"])

        assert warned == [True]
        # First attempt was `.[all]`, second (unguarded) was bare `.`.
        assert len(calls) == 2

    def test_base_install_oserror_warns_before_reraising(
        self, temp_pyproject, fake_scripts_dir, monkeypatch
    ):
        """A permission error on file writes (not just a non-zero exit) must
        also trigger the launcher-broken warning, not just CalledProcessError."""
        import subprocess

        import hermes_cli.main as main_mod

        (fake_scripts_dir / "hermes-agent").write_bytes(b"fake")
        (fake_scripts_dir / "hermes-acp").write_bytes(b"fake")

        calls: list[list[str]] = []

        def fake_run_quarantined_install(cmd, *, env=None, scripts_dir=None):
            calls.append(cmd)
            if len(calls) == 1:
                # First attempt (`.[all]`) fails normally, falling through to
                # the unguarded bare-`.` retry below.
                raise subprocess.CalledProcessError(2, cmd)
            raise PermissionError("[Errno 13] Permission denied")

        warned = []
        monkeypatch.setattr(main_mod, "_is_windows", lambda: False)
        monkeypatch.setattr(main_mod, "_venv_scripts_dir", lambda: fake_scripts_dir)
        monkeypatch.setattr(
            main_mod, "_run_quarantined_install", fake_run_quarantined_install
        )
        monkeypatch.setattr(
            main_mod,
            "_warn_if_hermes_launcher_broken",
            lambda: warned.append(True),
        )

        with pytest.raises(PermissionError):
            main_mod._install_python_dependencies_with_optional_fallback(["uv", "pip"])

        assert warned == [True]
