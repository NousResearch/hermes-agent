"""Regression coverage for project-aware locked syncs in Unix installers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
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


def test_installers_keep_bootstrap_isolation_but_restore_project_config_for_lock() -> None:
    install_text = INSTALL_SCRIPTS[0].read_text(encoding="utf-8")
    setup_text = INSTALL_SCRIPTS[1].read_text(encoding="utf-8")

    assert "export UV_NO_CONFIG=1" in install_text
    assert "export UV_NO_CONFIG=1" in setup_text
    assert _locked_sync_helper(INSTALL_SCRIPTS[0]) == _locked_sync_helper(
        INSTALL_SCRIPTS[1]
    )

    helper = _locked_sync_helper(INSTALL_SCRIPTS[0])
    assert "unset UV_NO_CONFIG UV_CONFIG_FILE" in helper
    assert 'export XDG_CONFIG_HOME="$isolated_uv_config"' in helper
    assert 'export XDG_CONFIG_DIRS="$isolated_uv_config"' in helper
    assert "$UV_CMD sync --extra all --locked" in helper
    assert 'run_locked_uv_sync "$INSTALL_DIR/venv"' in install_text
    assert 'run_locked_uv_sync "$SCRIPT_DIR/venv"' in setup_text


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


@pytest.mark.skipif(os.name == "nt", reason="Unix installer behavior")
def test_real_uv_accepts_lock_when_project_config_is_restored(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is unavailable")

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """[project]
name = "installer-lock-regression"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
all = []

[tool.uv]
package = false
exclude-newer = "14 days"
""",
        encoding="utf-8",
    )

    isolated_config = tmp_path / "isolated-config"
    isolated_config.mkdir()
    clean_env = os.environ.copy()
    clean_env.pop("UV_NO_CONFIG", None)
    clean_env.pop("UV_CONFIG_FILE", None)
    clean_env["XDG_CONFIG_HOME"] = str(isolated_config)
    clean_env["XDG_CONFIG_DIRS"] = str(isolated_config)
    clean_env["UV_OFFLINE"] = "1"

    locked = subprocess.run(
        [uv, "lock"],
        cwd=project,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert locked.returncode == 0, locked.stderr

    hidden_config_env = clean_env.copy()
    hidden_config_env["UV_NO_CONFIG"] = "1"
    rejected = subprocess.run(
        [uv, "lock", "--check"],
        cwd=project,
        env=hidden_config_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode != 0

    accepted = subprocess.run(
        [uv, "lock", "--check"],
        cwd=project,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
