"""Windows native-package lock recovery for hermes update.

Regression coverage for the Windows failure mode where a live gateway holds
``cryptography``'s ``_rust.pyd`` during ``hermes update``:

* ``Access is denied (os error 5)`` while replacing native wheels
* broken dist-info RECORD (``uninstall-no-record-file``)
* exact pin mismatch (e.g. required ``cryptography==46.0.7`` but 49.0.0 is left)

These tests force the Windows helpers via patching so they can run on any host.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

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
            dependencies = [
              "cryptography==46.0.7",
              "pathspec==1.1.1",
            ]
            """
        )
    )
    import hermes_cli.main as main_mod

    monkeypatch.setattr(main_mod, "PROJECT_ROOT", tmp_path)
    return tmp_path


def test_is_windows_native_package_lock_error_detects_known_symptoms():
    from hermes_cli.main import _is_windows_native_package_lock_error

    assert _is_windows_native_package_lock_error(
        "error: failed to remove file ... _rust.pyd: Access is denied. (os error 5)"
    )
    assert _is_windows_native_package_lock_error(
        "Cannot uninstall cryptography None\nno RECORD file was found"
    )
    assert not _is_windows_native_package_lock_error("Resolved 99 packages in 1.79s")


def test_quarantine_package_native_binaries_renames_pyd(tmp_path):
    from hermes_cli import main as cli_main

    site = tmp_path / "site-packages"
    pkg = site / "cryptography" / "hazmat" / "bindings"
    pkg.mkdir(parents=True)
    pyd = pkg / "_rust.pyd"
    pyd.write_bytes(b"locked-dll")

    with patch.object(cli_main, "_is_windows", return_value=True):
        moved = cli_main._quarantine_package_native_binaries(site, "cryptography")

    assert len(moved) == 1
    original, quarantined = moved[0]
    assert original == pyd
    assert quarantined.exists()
    assert not pyd.exists()
    assert ".old." in quarantined.name


def test_load_exact_dependency_pins(temp_pyproject):
    from hermes_cli.main import _load_exact_dependency_pins

    pins = _load_exact_dependency_pins()
    assert pins.get("cryptography") == "46.0.7"
    assert pins.get("pathspec") == "1.1.1"


def test_verify_detects_exact_pin_mismatch_and_force_repairs(temp_pyproject, tmp_path):
    """Present-but-wrong cryptography pin must trigger repair on Windows."""
    from hermes_cli.main import _verify_core_dependencies_installed

    venv_root = tmp_path / "venv"
    scripts = venv_root / "Scripts"
    scripts.mkdir(parents=True)
    py = scripts / "python.exe"
    py.write_text("fake")
    env = {"VIRTUAL_ENV": str(venv_root)}

    probe_calls = {"count": 0}

    def fake_subprocess_run(cmd, **kwargs):
        probe_calls["count"] += 1
        # Probe 1: cryptography pin wrong (and remains wrong after --reinstall).
        # Probe 2: still wrong.
        # Probe 3: fixed after ignore-installed force install.
        if probe_calls["count"] in (1, 2):
            return MagicMock(returncode=0, stdout="cryptography\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("hermes_cli.main._resolve_install_target_python", return_value=py), patch(
        "hermes_cli.main.subprocess.run", side_effect=fake_subprocess_run
    ), patch("hermes_cli.main._is_windows", return_value=True), patch(
        "hermes_cli.main._venv_scripts_dir", return_value=scripts
    ), patch(
        "hermes_cli.main._site_packages_for_install_target", return_value=None
    ), patch(
        "hermes_cli.main._run_install_with_heartbeat"
    ) as mock_install, patch(
        "hermes_cli.main._quarantine_running_hermes_exe", return_value=[]
    ):
        _verify_core_dependencies_installed(["uv", "pip"], env=env)

    assert mock_install.call_count >= 2
    last_args = mock_install.call_args_list[-1][0][0]
    assert "--ignore-installed" in last_args
    assert "--no-deps" in last_args
    assert any("cryptography==46.0.7" in str(a) for a in last_args)


def test_primary_install_success_runs_core_dep_verification():
    """Happy-path install must still verify exact pins (not only the fallback path)."""
    from hermes_cli.main import _install_python_dependencies_with_optional_fallback

    with patch("hermes_cli.main._is_windows", return_value=False), patch(
        "hermes_cli.main._run_quarantined_install"
    ) as mock_install, patch(
        "hermes_cli.main._verify_core_dependencies_installed"
    ) as mock_verify, patch(
        "hermes_cli.main._verify_console_scripts_installed"
    ) as mock_scripts:
        _install_python_dependencies_with_optional_fallback(["uv", "pip"], group="all")

    assert mock_install.called
    assert mock_verify.called
    assert mock_scripts.called
