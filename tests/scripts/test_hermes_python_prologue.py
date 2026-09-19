"""The self-activating shebang prologue: when does it re-activate?

``scripts/_hermes-python`` is the POSIX shebang target. Its one decision worth
pinning is staleness: ``__HERMES_ACTIVATED`` holds the installed-state file the
environment was built against, so any of ``uv.lock`` / ``pyproject.toml`` /
``pm/lock.json`` being newer than that file means the inherited environment
predates its inputs. Invert that either way and the cost is invisible — a
re-sync on every run, or a stale environment that looks fine.

A ``python3`` shim goes on PATH because the prologue execs ``python3`` by name,
which does not exist on stock Windows; the subject here is the staleness
branch, not interpreter resolution.
"""

from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROLOGUE = REPO_ROOT / "scripts" / "_hermes-python"

LONG_AGO = "2019-01-01 00:00:00"
STAMP_TIME = "2020-06-01 00:00:00"
JUST_AFTER = "2021-01-01 00:00:00"
INPUTS = ("uv.lock", "pyproject.toml", "pm/lock.json")


def _posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def _bash() -> str:
    found = shutil.which("bash")
    if found and "windowsapps" not in str(found).lower():
        return found
    if sys.platform == "win32":
        for rel in (("Git", "bin", "bash.exe"), ("Git", "usr", "bin", "bash.exe")):
            cand = Path(os.environ.get("ProgramFiles", r"C:\Program Files")).joinpath(*rel)
            if cand.exists():
                return str(cand)
    return found or "bash"


def _set_mtime(path: Path, stamp: str) -> None:
    when = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()
    os.utime(path, (when, when))


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A repo-shaped tree whose stub activate announces each sourcing."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "pm").mkdir()
    (root / "shim").mkdir()
    shutil.copy2(PROLOGUE, root / "scripts" / PROLOGUE.name)
    for name in ("uv.lock", "pyproject.toml"):
        (root / name).touch()
    (root / "pm" / "lock.json").touch()
    (root / "stamp").touch()

    # Quote in posix form: a /bin/sh script treats backslashes in an unquoted
    # word as escapes (same pattern as tests/pm/test_activate_scripts.py).
    shim = root / "shim" / "python3"
    shim.write_text("#!/bin/sh\nexec '%s' \"$@\"\n" % _posix(Path(sys.executable)), encoding="utf-8")
    shim.chmod(0o755)

    (root / "scripts" / "probe.py").write_text(
        "import os, sys\n"
        "print('target ran; sentinel =', os.environ.get('__HERMES_ACTIVATED'))\n",
        encoding="utf-8",
    )
    (root / "activate").write_text(
        "echo 'ACTIVATED' >&2\n"
        "export __HERMES_ACTIVATED=\"%s/stamp\"\n"
        "export PATH=\"%s/shim:$PATH\"\n" % (_posix(root), _posix(root)),
        encoding="utf-8",
    )
    return root


def _run(root: Path, sentinel: str | None) -> str:
    """Drive the prologue as the kernel would; return stderr, assert it ran."""
    env = {**os.environ, "PATH": f"{_posix(root / 'shim')}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.pop("__HERMES_ACTIVATED", None)
    if sentinel is not None:
        env["__HERMES_ACTIVATED"] = sentinel.replace("{root}", _posix(root))
    result = subprocess.run(
        [_bash(), _posix(root / "scripts" / PROLOGUE.name), _posix(root / "scripts" / "probe.py")],
        capture_output=True, text=True, cwd=_posix(root), env=env, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "target ran" in result.stdout, result.stderr
    return result.stderr


def test_current_environment_is_left_alone(checkout: Path):
    """Inputs older than the stamp: re-syncing on every run is the cost of
    getting this wrong, so the prologue must stay out of the way."""
    _set_mtime(checkout / "stamp", STAMP_TIME)
    for name in INPUTS:
        _set_mtime(checkout / name, LONG_AGO)
    assert "ACTIVATED" not in _run(checkout, "{root}/stamp")


@pytest.mark.parametrize("input_name", INPUTS)
def test_input_newer_than_stamp_reactivates(checkout: Path, input_name: str):
    _set_mtime(checkout / "stamp", STAMP_TIME)
    for name in INPUTS:
        _set_mtime(checkout / name, LONG_AGO)
    _set_mtime(checkout / input_name, JUST_AFTER)
    assert "ACTIVATED" in _run(checkout, "{root}/stamp")


@pytest.mark.parametrize("sentinel", [None, "{root}/gone", "1"])
def test_unusable_sentinel_activates(checkout: Path, sentinel: str | None):
    """Cold, dangling stamp, or the bare ``1`` an older activate exported —
    each must activate rather than read as current."""
    for name in INPUTS:
        _set_mtime(checkout / name, LONG_AGO)
    assert "ACTIVATED" in _run(checkout, sentinel)