"""Behavior coverage for macOS optional-package installation (#72730)."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def macos_package_harness(tmp_path: Path) -> tuple[str, dict[str, str], Path]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise install.sh")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    brew_log = tmp_path / "brew.log"

    _write_executable(fake_bin / "uname", "printf 'Darwin\\n'")
    _write_executable(fake_bin / "tr", "IFS= read -r line; printf '%s\\n' \"$line\"")
    _write_executable(
        fake_bin / "brew",
        "printf '%s\\n' \"$*\" >> \"$BREW_LOG\"",
    )

    env = {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PATH": str(fake_bin),
        "BREW_LOG": str(brew_log),
    }
    return bash, env, brew_log


def _run_ensure_ripgrep(
    harness: tuple[str, dict[str, str], Path], *args: str
) -> subprocess.CompletedProcess[str]:
    bash, env, _ = harness
    return subprocess.run(
        [bash, str(INSTALL_SH), "--ensure", "ripgrep", *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def test_macos_warns_before_homebrew_installs_optional_packages(
    macos_package_harness: tuple[str, dict[str, str], Path],
) -> None:
    result = _run_ensure_ripgrep(macos_package_harness)
    _, _, brew_log = macos_package_harness

    assert result.returncode == 0, result.stderr
    assert "Homebrew is about to install missing optional packages" in result.stdout
    assert "may also install or update transitive dependencies" in result.stdout
    assert "--skip-system-packages" in result.stdout
    assert brew_log.read_text(encoding="utf-8").strip() == "install ripgrep ffmpeg"


def test_skip_system_packages_prevents_homebrew_changes(
    macos_package_harness: tuple[str, dict[str, str], Path],
) -> None:
    result = _run_ensure_ripgrep(
        macos_package_harness,
        "--skip-system-packages",
    )
    _, _, brew_log = macos_package_harness

    assert result.returncode == 0, result.stderr
    assert "Skipping automatic installation of optional system packages" in result.stdout
    assert "brew install ripgrep" in result.stdout
    assert "brew install ffmpeg" in result.stdout
    assert not brew_log.exists()
