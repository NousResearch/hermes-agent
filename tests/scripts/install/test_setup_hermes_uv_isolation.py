"""Guard the manual developer bootstrap's uv isolation contract.

The migration of a legacy ``$HERMES_HOME/bin/uv`` + ``bin/uvx`` into the
private ``$HERMES_HOME/uv`` dir is pure shell and is lifted into a bash
harness against a fake ``$HERMES_HOME`` and actually run (behavior test,
sharing the pattern from tests/scripts/install/test_install_sh_uv_isolation.py).

The install switch (UV_UNMANAGED_INSTALL), the python-store pinning
(UV_PYTHON_*), and the state-dir pins (UV_CACHE_DIR / UV_TOOL_DIR) cannot run
hermetic here (the install path downloads + executes a remote installer), so
those remain source-text guards.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

# POSIX installer bash-harness: lifts the shell function out of setup-hermes.sh
# and runs it via `bash`. Windows uses install.ps1 (covered by the PowerShell CI
# harness + tests/hermes_cli/test_managed_uv.py), and native git-bash cannot
# resolve the WSL-style `/mnt/<drive>/...` harness path — so skip on Windows.
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX installer bash-harness: not runnable on native Windows",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = REPO_ROOT / "setup-hermes.sh"

_MIGRATE_SIG = "migrate_managed_uv_binaries() {\n"


def _migrate_fn() -> str:
    text = SETUP_SH.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_MIGRATE_SIG)
    assert marker, "setup-hermes.sh is missing migrate_managed_uv_binaries()"
    body, end, _ = rest.partition("\n}\n")
    assert end, "setup-hermes.sh has an unterminated migrate_managed_uv_binaries()"
    return marker + body + end


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _run_migrate_harness(home: Path) -> None:
    harness = home / "harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"HERMES_HOME='{_bash_path(home)}'\n"
        + _migrate_fn()
        + "\nmigrate_managed_uv_binaries\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", _bash_path(harness)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_migration_moves_legacy_uv_and_uvx_into_private_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    for name in ("uv", "uvx"):
        (home / "bin" / name).write_text(f"#fake {name}", encoding="utf-8")

    _run_migrate_harness(home)

    assert (home / "uv" / "uv").is_file()
    assert (home / "uv" / "uvx").is_file()
    assert not (home / "bin" / "uv").exists()
    assert not (home / "bin" / "uvx").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_migration_removes_stale_legacy_when_private_copy_exists(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "bin").mkdir(parents=True)
    (home / "uv").mkdir(parents=True)
    (home / "bin" / "uv").write_text("#stale", encoding="utf-8")
    (home / "uv" / "uv").write_text("#fresh", encoding="utf-8")

    _run_migrate_harness(home)

    assert (home / "uv" / "uv").read_text(encoding="utf-8") == "#fresh"
    assert not (home / "bin" / "uv").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_migration_noop_when_no_legacy_binaries(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / "uv").mkdir(parents=True)

    _run_migrate_harness(home)

    assert not (home / "uv" / "uv").exists()
    assert not (home / "uv" / "uvx").exists()


def test_setup_hermes_installs_uv_in_private_dir_without_path_write():
    body = SETUP_SH.read_text(encoding="utf-8")

    assert 'MANAGED_UV_DIR="$HERMES_HOME/uv"' in body
    assert 'UV_UNMANAGED_INSTALL="$MANAGED_UV_DIR" sh "$_uv_installer"' in body
    assert 'UV_CMD="$MANAGED_UV_DIR/uv"' in body
    assert 'command -v uv' not in body


_UV_STATE_SIG = "uv_isolated_state_env() {\n"


def _uv_state_env_fn() -> str:
    text = SETUP_SH.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_UV_STATE_SIG)
    assert marker, "setup-hermes.sh is missing uv_isolated_state_env()"
    body, end, _ = rest.partition("\n}\n")
    assert end, "setup-hermes.sh has an unterminated uv_isolated_state_env()"
    return marker + body + end


def _run_state_env_harness(home: Path, layout: str) -> str:
    """Run setup-hermes.sh's uv_isolated_state_env() against hostile inherited
    UV_* values and print the resulting axes — the values are proven by
    execution, not by grepping the script."""
    home.mkdir(parents=True, exist_ok=True)
    harness = home / "state-env-harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"HERMES_HOME='{_bash_path(home)}'\n"
        "export UV_CACHE_DIR=/evil/cache UV_TOOL_DIR=/evil/tools\n"
        "export UV_PYTHON_INSTALL_DIR=/evil/python UV_PYTHON_INSTALL_BIN=1\n"
        "export UV_PYTHON_INSTALL_REGISTRY=1 UV_PYTHON_BIN_DIR=/evil/bin\n"
        + _uv_state_env_fn()
        + f"\nuv_isolated_state_env {layout}\n"
        + 'echo "$UV_CACHE_DIR|$UV_TOOL_DIR|$UV_PYTHON_INSTALL_DIR|'
        '$UV_PYTHON_INSTALL_BIN|$UV_PYTHON_INSTALL_REGISTRY|${UV_PYTHON_BIN_DIR:-unset}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", _bash_path(harness)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_setup_user_layout_defeats_hostile_inherited_uv_state(tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    out = _run_state_env_harness(home, "user")
    assert out == f"{_bash_path(home)}/cache/uv|{_bash_path(home)}/uv/tools|{_bash_path(home)}/python|0|0|unset"
