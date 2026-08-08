"""The live harness must import THIS worktree, whatever interpreter runs it.

Regression for a real foreman-review failure: the documented command

    .venv/bin/python tests/computer_use/live_cua_launchservices_daemon.py

died with ``ImportError: cannot import name 'resolve_cua_driver_app' from
'tools.computer_use.cua_backend' (/Users/.../.hermes/hermes-agent/tools/...)``.

Running a script by path puts the script's own directory on ``sys.path[0]``, never
the repo root, so ``import tools...`` fell through to whichever hermes-agent
distribution the interpreter had installed — a different checkout entirely under
any shared venv. The harness now pins its own repo root ahead of everything else.

These tests exercise the real entry point in a subprocess under a hostile
environment. They never start a daemon: ``--check-imports`` returns before the
platform gate, so they are safe and meaningful on every OS.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "tests" / "computer_use" / "live_cua_launchservices_daemon.py"


@pytest.fixture
def decoy_checkout(tmp_path: Path) -> Path:
    """A fake ``tools.computer_use.cua_backend`` that wins if path order is wrong.

    It deliberately lacks ``resolve_cua_driver_app``, mirroring the stale checkout
    that produced the original ImportError.
    """
    package = tmp_path / "decoy" / "tools" / "computer_use"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "cua_backend.py").write_text("STALE_DECOY = True\n")
    return tmp_path / "decoy"


def _run(*args: str, cwd: Path, env_overrides: dict[str, str] | None = None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, str(HARNESS), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_harness_imports_this_worktree_from_an_unrelated_cwd(tmp_path):
    proc = _run("--check-imports", cwd=tmp_path)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"cua_backend: {REPO_ROOT}" in proc.stdout
    assert "PASS: imports resolve to this worktree" in proc.stdout


def test_harness_beats_a_stale_checkout_on_pythonpath(tmp_path, decoy_checkout):
    """A conflicting distribution on PYTHONPATH must not win."""
    proc = _run(
        "--check-imports",
        cwd=tmp_path,
        env_overrides={"PYTHONPATH": str(decoy_checkout)},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"cua_backend: {REPO_ROOT}" in proc.stdout
    assert str(decoy_checkout) not in proc.stdout


def test_harness_does_not_depend_on_an_inherited_pythonpath(tmp_path):
    """With PYTHONPATH explicitly emptied, resolution still lands here."""
    proc = _run("--check-imports", cwd=tmp_path, env_overrides={"PYTHONPATH": ""})

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"cua_backend: {REPO_ROOT}" in proc.stdout


def test_check_imports_never_touches_a_daemon(tmp_path):
    """The import probe must stay side-effect free on macOS too."""
    proc = _run("--check-imports", cwd=tmp_path)

    assert "launch argv" not in proc.stdout
    assert "socket:" not in proc.stdout
    assert "/usr/bin/open" not in proc.stdout
