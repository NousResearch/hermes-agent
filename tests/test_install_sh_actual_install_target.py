"""Regression tests for install.sh actual install target detection.

When ``--no-venv`` is passed, the installer can still end up with a
``$INSTALL_DIR/venv`` if the lockfile-backed ``uv sync`` path materializes one.
The installer must then reuse the *actual* interpreter / entry point it
installed, rather than blindly following the ``USE_VENV`` flag in later steps.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _extract_function(name: str) -> str:
    """Return the full shell function definition for ``name``."""
    text = INSTALL_SH.read_text()
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{.*?^\}}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in scripts/install.sh"
    return match.group(0)


def _make_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_install_followup_steps_use_resolved_install_target() -> None:
    """Static guard: installer follow-up steps should share helper-based lookup."""
    text = INSTALL_SH.read_text()

    assert "resolve_install_python()" in text
    assert "resolve_install_hermes_bin()" in text

    setup_path = _extract_function("setup_path")
    assert 'HERMES_BIN="$(resolve_install_hermes_bin' in setup_path

    run_setup_wizard = _extract_function("run_setup_wizard")
    assert 'install_python="$(resolve_install_python' in run_setup_wizard
    assert '"$install_python" -m hermes_cli.main setup < /dev/tty' in run_setup_wizard

    copy_config_templates = _extract_function("copy_config_templates")
    assert 'install_python="$(resolve_install_python' in copy_config_templates
    assert '"$install_python" "$INSTALL_DIR/tools/skills_sync.py"' in copy_config_templates


def test_resolve_install_hermes_bin_prefers_uv_managed_venv(tmp_path: Path) -> None:
    """Behavioral repro: prefer venv/bin/hermes over a PATH fallback."""
    helper = _extract_function("resolve_install_hermes_bin")

    install_dir = tmp_path / "install"
    venv_bin = install_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    uv_hermes = venv_bin / "hermes"
    _make_executable(uv_hermes, "#!/bin/sh\necho uv-hermes\n")

    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    path_hermes = path_bin / "hermes"
    _make_executable(path_hermes, "#!/bin/sh\necho path-hermes\n")

    script = f"""
set -e
INSTALL_DIR={install_dir!s}
{helper}
resolve_install_hermes_bin
"""
    env = os.environ.copy()
    env["PATH"] = f"{path_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, (
        f"resolve_install_hermes_bin failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == str(uv_hermes)


def test_resolve_install_python_prefers_uv_managed_venv(tmp_path: Path) -> None:
    """Behavioral repro: prefer venv/bin/python over PYTHON_PATH and PATH."""
    helper = _extract_function("resolve_install_python")

    install_dir = tmp_path / "install"
    venv_bin = install_dir / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    uv_python = venv_bin / "python"
    _make_executable(uv_python, "#!/bin/sh\necho uv-python\n")

    explicit_python = tmp_path / "explicit-python"
    _make_executable(explicit_python, "#!/bin/sh\necho explicit-python\n")

    path_bin = tmp_path / "path-bin"
    path_bin.mkdir()
    path_python = path_bin / "python"
    _make_executable(path_python, "#!/bin/sh\necho path-python\n")

    script = f"""
set -e
INSTALL_DIR={install_dir!s}
PYTHON_PATH={explicit_python!s}
{helper}
resolve_install_python
"""
    env = os.environ.copy()
    env["PATH"] = f"{path_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )
    assert result.returncode == 0, (
        f"resolve_install_python failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == str(uv_python)
