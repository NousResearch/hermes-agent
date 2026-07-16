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

    def test_triggers_reinstall_when_hermes_exe_missing(
        self, temp_pyproject, fake_scripts_dir
    ):
        (fake_scripts_dir / "hermes-agent.exe").write_bytes(b"fake")
        (fake_scripts_dir / "hermes-acp.exe").write_bytes(b"fake")

        with patch("hermes_cli.main._is_windows", return_value=True), \
             patch("hermes_cli.main._venv_scripts_dir", return_value=fake_scripts_dir), \
             patch("hermes_cli.main._run_quarantined_install") as mock_install:
            from hermes_cli.main import _verify_console_scripts_installed

            _verify_console_scripts_installed(["uv", "pip"], env={})

        mock_install.assert_called_once()
        args = mock_install.call_args[0][0]
        assert "--reinstall" in args
        assert "-e" in args and "." in args
        assert mock_install.call_args[1]["scripts_dir"] == fake_scripts_dir

    def test_skips_off_windows(self, temp_pyproject, fake_scripts_dir):
        with patch("hermes_cli.main._is_windows", return_value=False), \
             patch("hermes_cli.main._run_quarantined_install") as mock_install:
            from hermes_cli.main import _verify_console_scripts_installed

            _verify_console_scripts_installed(["uv", "pip"], env={})

        mock_install.assert_not_called()

    def test_load_console_script_names_reads_pyproject(self, temp_pyproject):
        from hermes_cli.main import _load_console_script_names

        names = _load_console_script_names()
        assert names == ["hermes", "hermes-agent", "hermes-acp"]

    def test_primary_install_success_still_verifies_scripts(self):
        import hermes_cli.main as main_mod

        with patch("hermes_cli.main._is_windows", return_value=False), \
             patch("hermes_cli.main._run_quarantined_install") as mock_install, \
             patch("hermes_cli.main._verify_console_scripts_installed") as mock_verify:
            main_mod._install_python_dependencies_with_optional_fallback(
                ["uv", "pip"], env={"VIRTUAL_ENV": "x"}
            )

        mock_install.assert_called_once_with(
            ["uv", "pip", "install", "-e", ".[all]"],
            env={"VIRTUAL_ENV": "x"},
            scripts_dir=None,
        )
        mock_verify.assert_called_once_with(["uv", "pip"], env={"VIRTUAL_ENV": "x"})

    def test_quarantine_shims_include_declared_console_scripts(
        self, temp_pyproject, fake_scripts_dir
    ):
        import hermes_cli.main as main_mod

        with patch("hermes_cli.main._is_windows", return_value=True):
            names = {path.name for path in main_mod._hermes_exe_shims(fake_scripts_dir)}

        assert {"hermes.exe", "hermes-agent.exe", "hermes-acp.exe"} <= names
        assert "hermes-gateway.exe" in names
