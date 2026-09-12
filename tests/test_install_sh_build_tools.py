"""Regression coverage for Debian build-tool installation in install.sh."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _install_deps_function() -> str:
    source = INSTALL_SH.read_text(encoding="utf-8")
    start = source.index("install_deps() {\n")
    end = source.index("\n}\n\n", start) + len("\n}\n")
    return source[start:end]


def _run_root_build_tool_install(
    tmp_path: Path, missing_package: str
) -> tuple[subprocess.CompletedProcess[str], list[str], list[str]]:
    apt_log = tmp_path / "apt.log"
    dpkg_log = tmp_path / "dpkg.log"
    install_dir = tmp_path / "install"
    install_dir.mkdir()

    harness = f"""
set -eu
APT_LOG={shlex.quote(str(apt_log))}
DPKG_LOG={shlex.quote(str(dpkg_log))}
MISSING_PACKAGE={shlex.quote(missing_package)}
: > "$APT_LOG"
: > "$DPKG_LOG"

log_info() {{ :; }}
log_success() {{ :; }}
log_warn() {{ :; }}
id() {{ echo 0; }}
dpkg() {{
    printf '%s\\n' "$2" >> "$DPKG_LOG"
    [ "$2" = "$MISSING_PACKAGE" ] && return 1
    return 0
}}
apt-get() {{ printf '%s\\n' "$*" >> "$APT_LOG"; }}
sudo() {{ return 1; }}
uv() {{ [ "$1" = "sync" ]; }}

DISTRO=debian
USE_VENV=false
INSTALL_DIR={shlex.quote(str(install_dir))}
UV_CMD=uv

{_install_deps_function()}

install_deps
"""
    harness_path = tmp_path / "run-install-deps.sh"
    harness_path.write_text(harness, encoding="utf-8")
    proc = subprocess.run(
        ["bash", str(harness_path)],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        proc,
        apt_log.read_text(encoding="utf-8").splitlines(),
        dpkg_log.read_text(encoding="utf-8").splitlines(),
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires bash")
@pytest.mark.parametrize("missing_package", ["g++", "make"])
def test_root_debian_installs_complete_node_gyp_toolchain(
    tmp_path: Path, missing_package: str
) -> None:
    """Root installs must not depend on sudo to repair node-gyp prerequisites."""
    proc, apt_calls, dpkg_calls = _run_root_build_tool_install(
        tmp_path, missing_package
    )

    assert proc.returncode == 0, proc.stderr
    assert missing_package in dpkg_calls
    assert apt_calls == [
        "update -qq",
        "install -y -qq build-essential python3-dev libffi-dev",
    ]
