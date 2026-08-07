"""Behavioral coverage for installer Node/npm toolchain detection."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
NODE_BOOTSTRAP_SH = REPO_ROOT / "scripts" / "lib" / "node-bootstrap.sh"
BASH = shutil.which("bash") or ""

pytestmark = [
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(not BASH, reason="needs bash"),
    pytest.mark.skipif(sys.platform == "win32", reason="tests POSIX installers"),
]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _fake_toolchain(
    tmp_path: Path,
    *,
    with_npm: bool,
    npm_install_exit: int = 0,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "node",
        'if [ "${1:-}" = "--version" ]; then echo "v22.22.2"; fi',
    )
    _write_executable(
        bin_dir / "uname",
        'case "${1:-}" in -s) echo Darwin ;; -m) echo test-unsupported ;; *) echo Darwin ;; esac',
    )
    _write_executable(bin_dir / "sleep", "exit 0")

    # The sourceable bootstrap helper parses node's version with these tools.
    for name in ("sed", "cut", "tr"):
        resolved = shutil.which(name)
        assert resolved is not None
        _write_executable(bin_dir / name, f'exec "{resolved}" "$@"')

    if with_npm:
        _write_executable(
            bin_dir / "npm",
            (
                'if [ "${1:-}" = "--version" ]; then echo "10.9.0"; exit 0; fi\n'
                f"exit {npm_install_exit}"
            ),
        )
    return bin_dir


def _env(bin_dir: Path, tmp_path: Path) -> dict[str, str]:
    return os.environ | {
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PATH": str(bin_dir),
    }


def test_ensure_node_rejects_node_without_npm(tmp_path: Path) -> None:
    bin_dir = _fake_toolchain(tmp_path, with_npm=False)

    result = subprocess.run(
        [BASH, str(INSTALL_SH), "--ensure", "node"],
        env=_env(bin_dir, tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "has no usable npm on PATH" in result.stdout
    assert "Node.js v22.22.2 found" not in result.stdout


def test_ensure_node_accepts_complete_toolchain(tmp_path: Path) -> None:
    bin_dir = _fake_toolchain(tmp_path, with_npm=True)

    result = subprocess.run(
        [BASH, str(INSTALL_SH), "--ensure", "node"],
        env=_env(bin_dir, tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Node.js v22.22.2 found" in result.stdout
    assert "test-unsupported" not in result.stdout


def test_sourceable_bootstrap_requires_npm_with_node(tmp_path: Path) -> None:
    bin_dir = _fake_toolchain(tmp_path, with_npm=False)

    result = subprocess.run(
        [BASH, "-c", 'source "$1"; _nb_have_modern_node', "_", str(NODE_BOOTSTRAP_SH)],
        env=_env(bin_dir, tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0


@pytest.mark.parametrize(
    ("package_dir", "failure_message", "false_success"),
    [
        (".", "npm install failed or timed out", "Node.js dependencies installed"),
        ("ui-tui", "TUI npm install failed or timed out", "TUI dependencies installed"),
    ],
)
def test_node_deps_does_not_claim_success_after_npm_failure(
    tmp_path: Path,
    package_dir: str,
    failure_message: str,
    false_success: str,
) -> None:
    bin_dir = _fake_toolchain(tmp_path, with_npm=True, npm_install_exit=1)
    install_dir = tmp_path / "checkout"
    target = install_dir if package_dir == "." else install_dir / package_dir
    target.mkdir(parents=True)
    (target / "package.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        [
            BASH,
            str(INSTALL_SH),
            "--stage",
            "node-deps",
            "--dir",
            str(install_dir),
            "--skip-browser",
            "--non-interactive",
        ],
        env=_env(bin_dir, tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert failure_message in result.stdout
    assert false_success not in result.stdout
