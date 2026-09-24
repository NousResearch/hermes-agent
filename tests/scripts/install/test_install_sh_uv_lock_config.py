"""Regression coverage for project-aware locked syncs in Unix installers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
INSTALL_SCRIPTS = (
    REPO_ROOT / "scripts" / "install.sh",
    REPO_ROOT / "setup-hermes.sh",
)
_HELPER_START = "run_locked_uv_sync() {\n"


def _locked_sync_helper(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_HELPER_START)
    assert marker, f"{path.name} is missing run_locked_uv_sync()"
    body, end, _ = rest.partition("\n}\n")
    assert end, f"{path.name} has an unterminated run_locked_uv_sync()"
    return marker + body + end


def _bash_path(path: Path, bash: str) -> str:
    """Path form to pass to THIS bash on Windows.

    MSYS-family bashes (Git Bash, MSYS2, Cygwin) accept Windows drive
    paths directly, so probe uname instead of assuming WSL: handing a
    Git Bash a /mnt/<drive>/... path fails with exit 127 on every
    developer Windows box, while CI's WSL bash needs exactly that form.
    """
    if os.name != "nt":
        return str(path)
    probe = subprocess.run(
        [bash, "-c", "uname -s"], capture_output=True, text=True, check=False,
    )
    uname = probe.stdout.upper()
    if "MINGW" in uname or "MSYS" in uname or "CYGWIN" in uname:
        return path.as_posix()
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def test_locked_sync_helper_sanitizes_only_its_subprocess(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is unavailable")

    record = tmp_path / "uv-args.txt"
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        """#!/bin/sh
test -z "${UV_NO_CONFIG+x}" || exit 10
test -z "${UV_CONFIG_FILE+x}" || exit 11
test -d "$XDG_CONFIG_HOME" || exit 12
test "$XDG_CONFIG_HOME" = "$XDG_CONFIG_DIRS" || exit 13
printf '%s\n' "$@" > "$RECORD"
""",
        encoding="utf-8",
        newline="\n",
    )

    harness = tmp_path / "harness.sh"
    harness.write_text(
        """#!/bin/bash
set -eu
UV_CMD="sh $1"
RECORD="$2"
export UV_CMD RECORD
export UV_NO_CONFIG=1
export UV_CONFIG_FILE=/poison/uv.toml
export XDG_CONFIG_HOME=/poison/user
export XDG_CONFIG_DIRS=/poison/system
"""
        + _locked_sync_helper(INSTALL_SCRIPTS[0])
        + """
run_locked_uv_sync /tmp/hermes-venv
test "$UV_NO_CONFIG" = 1
test "$UV_CONFIG_FILE" = /poison/uv.toml
test "$XDG_CONFIG_HOME" = /poison/user
test "$XDG_CONFIG_DIRS" = /poison/system
""",
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(
        [bash, _bash_path(harness, bash), _bash_path(fake_uv, bash),
         _bash_path(record, bash)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert record.read_text(encoding="utf-8").splitlines() == [
        "sync",
        "--extra",
        "all",
        "--locked",
    ]


