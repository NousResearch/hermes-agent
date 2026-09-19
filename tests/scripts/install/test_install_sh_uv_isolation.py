"""Guard: install.sh must install the managed uv with UV_UNMANAGED_INSTALL.

The astral installer only suppresses its shell-profile PATH write
(NO_MODIFY_PATH=1) when UV_UNMANAGED_INSTALL is set. install.sh must keep
passing it — the same no-PATH-write invariant install.ps1's CI harness
asserts on the Windows side (scripts/tests/test_install_ps1_uv_isolation.ps1).
Without it, a fresh POSIX install would have the astral installer append
``$HERMES_HOME/uv`` to ~/.profile/.bashrc and shadow the user's uv.

Two concerns, two test styles:

1. The *migration* of a legacy ``$HERMES_HOME/bin/uv`` + ``bin/uvx`` into the
   private ``$HERMES_HOME/uv`` dir is pure shell — extractable and executable.
   ``migrate_managed_uv_binaries()`` is lifted verbatim into a bash harness
   against a fake ``$HERMES_HOME`` and actually run (behavior test).

2. The *state-dir pins* (``uv_isolated_state_env``) and the layout selection
   that install_uv drives (``apply_uv_isolated_state_env``) are pure shell and
   run in a bash harness against a hostile inherited ``UV_*`` environment.

3. The single remaining static check is the astral installer *switch*
   ``UV_UNMANAGED_INSTALL``: install_uv() downloads and executes a remote
   installer, so it cannot be lifted and run offline.  It is one substring on
   a call site, not logic, and the same switch is checked behaviorally on the
   Windows side by ``scripts/tests/test_install_ps1_uv_isolation.ps1``.
"""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

# These lift a shell function out of the POSIX installers and run it in a
# `bash` harness. install.sh / setup-hermes.sh are the POSIX install path;
# Windows uses install.ps1, and its migration is covered by the PowerShell CI
# harness plus tests/hermes_cli/test_managed_uv.py. On native Windows git-bash
# the harness writes WSL-style `/mnt/<drive>/...` paths that the host `bash`
# cannot resolve, so skip here — the function is not exercised on Windows.
pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX installer bash-harness: not runnable on native Windows",
)


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

_MIGRATE_SIG = "migrate_managed_uv_binaries() {\n"


def _migrate_fn() -> str:
    """Return install.sh's migrate_managed_uv_binaries() body verbatim."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_MIGRATE_SIG)
    assert marker, "install.sh is missing migrate_managed_uv_binaries()"
    body, end, _ = rest.partition("\n}\n")
    assert end, "install.sh has an unterminated migrate_managed_uv_binaries()"
    return marker + body + end


def _bash_path(path: Path) -> str:
    """Map a Windows path into the WSL/bash mount namespace if needed."""
    if os.name != "nt":
        return str(path)
    drive = path.drive.rstrip(":").lower()
    tail = path.as_posix().split(":", 1)[1].lstrip("/")
    return f"/mnt/{drive}/{tail}"


def _run_migrate_harness(home: Path) -> None:
    """Execute install.sh's migrate_managed_uv_binaries() against a fake home."""
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


def _install_uv_body() -> str:
    """Return just the body of install_uv(), bounded by its opening
    signature and the next top-level ``}`` close brace."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    head, _, rest = text.partition("install_uv() {\n")
    assert rest, "Could not find install_uv() in scripts/install.sh"
    body, _, _ = rest.partition("\n}\n")
    assert body, "Could not find install_uv() closing brace"
    return body


_UV_STATE_ENV_SIG = "uv_isolated_state_env() {\n"


def _uv_state_env_fn() -> str:
    """Return install.sh's uv_isolated_state_env() verbatim (pure shell)."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_UV_STATE_ENV_SIG)
    assert marker, "install.sh is missing uv_isolated_state_env()"
    body, end, _ = rest.partition("\n}\n")
    assert end, "install.sh has an unterminated uv_isolated_state_env()"
    return marker + body + end


def _run_uv_state_env_harness(home: Path, layout: str) -> str:
    """Execute uv_isolated_state_env() with HOSTILE inherited UV_* values and
    print the five resulting axes, so tests prove the branch really executes
    and defeats the inherited values (not that the strings merely occur in
    the source)."""
    harness = home / "state-env-harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"HERMES_HOME='{_bash_path(home / 'hermes')}'\n"
        "export UV_CACHE_DIR=/evil/cache UV_TOOL_DIR=/evil/tools\n"
        "export UV_PYTHON_INSTALL_DIR=/evil/python UV_PYTHON_INSTALL_BIN=1\n"
        "export UV_PYTHON_INSTALL_REGISTRY=1 UV_PYTHON_BIN_DIR=/evil/bin\n"
        + _uv_state_env_fn()
        + f"\nuv_isolated_state_env {layout}\n"
        + 'echo "$UV_CACHE_DIR|$UV_TOOL_DIR|$UV_PYTHON_INSTALL_DIR|$UV_PYTHON_INSTALL_BIN|$UV_PYTHON_INSTALL_REGISTRY|${UV_PYTHON_BIN_DIR:-unset}"\n',
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
def test_user_layout_defeats_hostile_inherited_uv_state(tmp_path: Path) -> None:
    """Behavioral guard for the common install path: hostile inherited UV_*
    values must be overridden to the $HERMES_HOME tree."""
    out = _run_uv_state_env_harness(tmp_path, "user")
    home = _bash_path(tmp_path / "hermes")
    assert out == f"{home}/cache/uv|{home}/uv/tools|{home}/python|0|0|unset"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_fhs_root_layout_defeats_hostile_inherited_uv_state(tmp_path: Path) -> None:
    """Behavioral guard for the root-FHS branch (the previous gap): hostile
    inherited UV_* values must be overridden to Hermes' own system tree, with
    the python store staying world-traversable under /usr/local/share (#21457)."""
    out = _run_uv_state_env_harness(tmp_path, "fhs-root")
    assert (
        out
        == "/var/cache/hermes/uv|/usr/local/share/hermes/uv/tools|/usr/local/share/uv/python|0|0|/usr/local/share/uv/bin"
    )


_APPLY_SIG = "apply_uv_isolated_state_env() {\n"


def _apply_env_fn() -> str:
    """Return install.sh's apply_uv_isolated_state_env() verbatim."""
    text = INSTALL_SH.read_text(encoding="utf-8")
    _, marker, rest = text.partition(_APPLY_SIG)
    assert marker, "install.sh is missing apply_uv_isolated_state_env()"
    body, end, _ = rest.partition("\n}\n")
    assert end, "install.sh has an unterminated apply_uv_isolated_state_env()"
    return marker + body + end


def _run_apply_harness(home: Path, root_fhs: str) -> str:
    """Run apply_uv_isolated_state_env() with ROOT_FHS_LAYOUT set, and print the
    resulting axes.  This is the wiring install_uv relies on, proven by
    execution instead of grepping install_uv()'s body."""
    home.mkdir(parents=True, exist_ok=True)
    harness = home / "apply-harness.sh"
    harness.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"HERMES_HOME='{_bash_path(home)}'\n"
        f"ROOT_FHS_LAYOUT='{root_fhs}'\n"
        + _uv_state_env_fn()
        + _apply_env_fn()
        + "\napply_uv_isolated_state_env\n"
        + 'echo "$UV_CACHE_DIR|$UV_TOOL_DIR|$UV_PYTHON_INSTALL_DIR|'
        '$UV_PYTHON_INSTALL_BIN|$UV_PYTHON_INSTALL_REGISTRY|${UV_PYTHON_BIN_DIR:-unset}"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", _bash_path(harness)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_apply_selects_user_layout(tmp_path: Path) -> None:
    home = _bash_path(tmp_path / "hermes")
    assert _run_apply_harness(tmp_path / "hermes", "false") == (
        f"{home}/cache/uv|{home}/uv/tools|{home}/python|0|0|unset")


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is unavailable")
def test_apply_selects_fhs_root_layout(tmp_path: Path) -> None:
    assert _run_apply_harness(tmp_path / "hermes", "true") == (
        "/var/cache/hermes/uv|/usr/local/share/hermes/uv/tools|"
        "/usr/local/share/uv/python|0|0|/usr/local/share/uv/bin")


def test_install_uv_runs_installer_with_uv_unmanaged_install() -> None:
    """The one accepted source-shape check: install_uv() downloads and runs the
    astral installer, so its call site cannot be lifted and executed offline.
    The switch itself is behavioral-only in effect (no PATH write on a fresh
    install); the same invariant is exercised on Windows by
    scripts/tests/test_install_ps1_uv_isolation.ps1."""
    body = _install_uv_body()

    assert 'UV_UNMANAGED_INSTALL="$HERMES_HOME/uv"' in body, (
        "install_uv() must run the astral installer with UV_UNMANAGED_INSTALL "
        "pointing at the private managed dir. On POSIX the astral install.sh "
        "maps it to NO_MODIFY_PATH=1, which is what keeps the managed dir out "
        "of ~/.profile/.bashrc; without it a fresh install would shadow the "
        "user's uv."
    )
