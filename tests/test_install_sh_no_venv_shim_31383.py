"""Regression for #31383: ``install.sh --no-venv`` must still create the shim.

The ``--no-venv`` flag historically dropped ``setup_path()`` into a branch
that searched ``$PATH`` for ``hermes`` via ``which hermes``.  But the uv
install path (``install_deps``) always sets
``UV_PROJECT_ENVIRONMENT="$INSTALL_DIR/venv"`` for ``uv sync``, which
materializes a venv at ``$INSTALL_DIR/venv/`` and writes the entry point
there.  Since the uv-managed venv is not on ``$PATH`` by default,
``which hermes`` returned empty, ``setup_path`` logged a warning and
returned, and ``~/.local/bin/hermes`` was never created.

These tests pin the fix: ``setup_path()`` must check
``$INSTALL_DIR/venv/bin/hermes`` first in the ``--no-venv`` branch and
only fall back to ``which hermes`` when that uv-created entry point is
absent.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_setup_path_function() -> str:
    """Return the install.sh ``setup_path()`` function body up to the shim block."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r"(?P<block>setup_path\(\)\s*\{.*?if \[ ! -x \"\$HERMES_BIN\" \];)",
        text,
        re.DOTALL,
    )
    assert match is not None, (
        "Could not locate setup_path() function header in scripts/install.sh"
    )
    return match["block"]


# ---------------------------------------------------------------------------
# Static guard — the fix mustn't be silently reverted by a refactor.
# ---------------------------------------------------------------------------


def test_setup_path_no_venv_branch_checks_uv_venv_first() -> None:
    """The ``USE_VENV=false`` branch must consult ``$INSTALL_DIR/venv/bin/hermes``
    before falling through to ``which hermes``.

    Pins the #31383 fix.  If a future refactor drops the uv-venv lookup the
    silent-no-shim regression returns immediately on every ``--no-venv``
    install.
    """
    block = _extract_setup_path_function()

    # Strip ``# …`` comments so commentary mentioning ``which hermes`` doesn't
    # spoof the ordering check.  Bash strings never start with ``#`` at the
    # start of a token so a line-anchored strip is safe here.
    code_only = re.sub(r"^\s*#.*$", "", block, flags=re.MULTILINE)

    uv_check_idx = code_only.find('-x "$INSTALL_DIR/venv/bin/hermes"')
    # Match the ACTUAL invocation, not prose that mentions the command.
    which_call_idx = code_only.find('$(which hermes')

    assert uv_check_idx != -1, (
        "setup_path()'s --no-venv branch must check "
        "$INSTALL_DIR/venv/bin/hermes (the uv-managed entry point) before "
        "falling back to `which hermes`. See #31383."
    )
    assert which_call_idx != -1, "expected `which hermes` fallback still present"
    assert uv_check_idx < which_call_idx, (
        "$INSTALL_DIR/venv/bin/hermes lookup must come BEFORE the "
        "`which hermes` fallback — otherwise the uv-created venv is "
        "ignored and the shim is never created. See #31383."
    )


# ---------------------------------------------------------------------------
# Behavioral repro — drive setup_path() against a controlled fixture and
# verify the user-visible artifact (~/.local/bin/hermes) actually appears.
# ---------------------------------------------------------------------------


def _setup_path_test_script(install_dir: Path, link_dir: Path, *, use_venv: bool) -> str:
    """Return a bash snippet that sources just enough of install.sh to call
    ``setup_path`` against a fixture.

    Stubs out everything ``setup_path`` reaches for that we don't care about
    (logging, distro probing, command-link helpers) so the test stays
    hermetic and doesn't touch the host filesystem outside ``tmp_path``.
    """
    install_sh_text = INSTALL_SH.read_text()
    setup_path_match = re.search(
        r"setup_path\(\)\s*\{.*?\n\}\n",
        install_sh_text,
        re.DOTALL,
    )
    assert setup_path_match is not None, "could not isolate setup_path() body"
    setup_path_src = setup_path_match.group(0)

    use_venv_str = "true" if use_venv else "false"
    return f"""
set -e
USE_VENV={use_venv_str}
INSTALL_DIR={install_dir!s}
DISTRO=ubuntu
ROOT_FHS_LAYOUT=false
HOME={install_dir!s}/home
mkdir -p "$HOME"

# Stubs for helpers setup_path() calls that we don't want to exercise.
log_info()    {{ echo "INFO: $*"; }}
log_warn()    {{ echo "WARN: $*"; }}
log_success() {{ echo "OK: $*"; }}
log_error()   {{ echo "ERR: $*"; }}
get_command_link_dir()         {{ echo "{link_dir!s}"; }}
get_command_link_display_dir() {{ echo "~/local/bin"; }}

# Force `which hermes` to return nothing (the bug scenario): the uv-created
# venv is NOT on PATH, and there is no globally-installed hermes.  We
# achieve this by giving the test bash a minimal PATH that excludes any
# directory containing a `hermes` executable.
export PATH=/usr/bin:/bin

{setup_path_src}

setup_path
"""


def test_no_venv_install_with_uv_created_venv_writes_shim(tmp_path: Path) -> None:
    """The ``--no-venv`` flow must create ``$command_link_dir/hermes`` when
    ``uv sync`` left an entry point at ``$INSTALL_DIR/venv/bin/hermes``.

    Pre-fix behaviour: ``setup_path`` saw ``USE_VENV=false``, ran
    ``which hermes`` (empty), warned, returned 0 — no shim.

    Post-fix behaviour: ``setup_path`` finds the uv-managed entry point
    and writes the launcher.
    """
    install_dir = tmp_path / "hermes-agent"
    venv_bin = install_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    uv_entry_point = venv_bin / "hermes"
    uv_entry_point.write_text("#!/usr/bin/env python\n# uv-generated entry point\n")
    uv_entry_point.chmod(uv_entry_point.stat().st_mode | stat.S_IXUSR)

    link_dir = tmp_path / "local_bin"

    script = _setup_path_test_script(install_dir, link_dir, use_venv=False)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"setup_path failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    shim = link_dir / "hermes"
    assert shim.exists(), (
        "setup_path() under --no-venv must create $command_link_dir/hermes "
        "when uv created the venv entry point. The pre-fix code dropped "
        "to `which hermes`, found nothing, and returned without writing "
        "the shim. See #31383.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    shim_text = shim.read_text()
    assert "unset PYTHONPATH" in shim_text
    assert f'exec "{uv_entry_point}"' in shim_text, (
        "Shim must execute the uv-created entry point, not whatever "
        "`which hermes` happened to find."
    )
    assert shim.stat().st_mode & stat.S_IXUSR, "shim must be user-executable"


def test_no_venv_without_uv_venv_falls_back_to_which_hermes(tmp_path: Path) -> None:
    """When uv was bypassed entirely (no ``$INSTALL_DIR/venv/bin/hermes``),
    ``setup_path`` must still attempt the legacy ``which hermes`` lookup so
    operators who pre-installed via system pip aren't silently broken by
    the new check.
    """
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    # Deliberately do NOT create $INSTALL_DIR/venv/bin/hermes.

    # Provide a fake `hermes` somewhere reachable via PATH so `which hermes`
    # succeeds — confirms the fallback is still wired.
    fake_bin_dir = tmp_path / "fake_path"
    fake_bin_dir.mkdir()
    fake_hermes = fake_bin_dir / "hermes"
    fake_hermes.write_text("#!/usr/bin/env bash\n# fake system-pip entry point\n")
    fake_hermes.chmod(fake_hermes.stat().st_mode | stat.S_IXUSR)

    link_dir = tmp_path / "local_bin"

    install_sh_text = INSTALL_SH.read_text()
    setup_path_match = re.search(
        r"setup_path\(\)\s*\{.*?\n\}\n",
        install_sh_text,
        re.DOTALL,
    )
    assert setup_path_match is not None
    setup_path_src = setup_path_match.group(0)

    script = f"""
set -e
USE_VENV=false
INSTALL_DIR={install_dir!s}
DISTRO=ubuntu
ROOT_FHS_LAYOUT=false
HOME={install_dir!s}/home
mkdir -p "$HOME"

log_info()    {{ echo "INFO: $*"; }}
log_warn()    {{ echo "WARN: $*"; }}
log_success() {{ echo "OK: $*"; }}
log_error()   {{ echo "ERR: $*"; }}
get_command_link_dir()         {{ echo "{link_dir!s}"; }}
get_command_link_display_dir() {{ echo "~/local/bin"; }}

# Put the fake hermes on PATH.
export PATH={fake_bin_dir!s}:/usr/bin:/bin

{setup_path_src}

setup_path
"""
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, (
        f"setup_path failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )

    shim = link_dir / "hermes"
    assert shim.exists(), (
        "When the uv venv is missing the legacy `which hermes` fallback "
        "must still create the shim. The fix must not break operators "
        "who already had hermes on PATH via a system-pip install. "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert f'exec "{fake_hermes}"' in shim.read_text(), (
        "Fallback shim must invoke whichever hermes binary `which` returned."
    )


def test_no_venv_without_any_hermes_warns_and_returns(tmp_path: Path) -> None:
    """If neither the uv venv nor PATH have a ``hermes`` entry point,
    ``setup_path`` should warn cleanly and return 0 (don't ``set -e`` blow
    up the entire installer).
    """
    install_dir = tmp_path / "hermes-agent"
    install_dir.mkdir()
    link_dir = tmp_path / "local_bin"

    script = _setup_path_test_script(install_dir, link_dir, use_venv=False)
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, "setup_path must return 0 on missing hermes"
    assert "WARN: hermes not found on PATH after install" in result.stdout
    assert not (link_dir / "hermes").exists(), (
        "No hermes binary anywhere → no shim should be written."
    )
