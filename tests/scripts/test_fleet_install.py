from zipfile import ZipFile

import pytest

from scripts.fleet_install import (
    build_runner_bundle,
    liveness_marker_is_active,
    resolve_hermes_executable,
    runner_command,
)


def test_bundle_is_standalone_and_contains_no_checkout_or_secret_files(tmp_path):
    bundle = build_runner_bundle(tmp_path / "fleet-runner.pyz")

    with ZipFile(bundle) as archive:
        names = set(archive.namelist())

    assert "__main__.py" in names
    assert "fleet_runner.py" in names
    assert "hermes_cli/fleet_protocol.py" in names
    assert not any(name.startswith(".git/") or name.endswith(".env") for name in names)


def test_runner_uses_the_exact_verified_executable_and_bundle_command(tmp_path):
    executable = tmp_path / "hermes.exe"
    executable.write_text("stub", encoding="utf-8")
    bundle = tmp_path / "fleet-runner.pyz"
    bundle.write_bytes(b"bundle")

    assert resolve_hermes_executable(executable) == str(executable)
    assert runner_command("python.exe", bundle, "windows") == ["python.exe", str(bundle), "--node-id", "windows"]


def test_runner_refuses_to_start_without_the_desktop_liveness_marker(tmp_path):
    marker = tmp_path / "fleet-live"

    assert liveness_marker_is_active(marker) is False
    marker.write_text("live", encoding="utf-8")
    assert liveness_marker_is_active(marker) is True

    with pytest.raises(ValueError, match="does not exist"):
        resolve_hermes_executable(tmp_path / "missing-hermes.exe")
