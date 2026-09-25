"""The self-activating shebang prologue: when does it re-activate?

``scripts/_hermes-python`` is the POSIX shebang target. Its one decision worth
pinning is staleness: pm stamps each dependency input's mtime beside the
installed-state file ``__HERMES_ACTIVATED`` names, and any input whose mtime
differs from its stamp means the inherited environment may not match its
inputs. Invert that either way and the cost is invisible — a re-sync on every
run, or a stale environment that looks fine.

The stamps come from the real ``pm.environments`` writer, so these tests pin
the contract between what pm records and what the prologue reads.

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

from pm.environments import ACTIVATION_INPUTS, activation_input_mtimes, record_activation_inputs

REPO_ROOT = Path(__file__).resolve().parents[2]
PROLOGUE = REPO_ROOT / "scripts" / "_hermes-python"

EARLIER = "2018-01-01 00:00:00"
LONG_AGO = "2019-01-01 00:00:00"
STAMP_TIME = "2020-06-01 00:00:00"
JUST_AFTER = "2021-01-01 00:00:00"


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
    """A repo-shaped tree whose stub activate announces each sourcing; its
    installed state lives in ``state/`` (facts.json beside the input stamps)."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "shim").mkdir()
    (root / "state").mkdir()
    shutil.copy2(PROLOGUE, root / "scripts" / PROLOGUE.name)
    shutil.copy2(REPO_ROOT / "scripts" / "_activation.sh", root / "scripts" / "_activation.sh")
    for name in ACTIVATION_INPUTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
        _set_mtime(root / name, LONG_AGO)
    (root / "state" / "facts.json").touch()
    _set_mtime(root / "state" / "facts.json", STAMP_TIME)

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
        "export __HERMES_ACTIVATED=\"%s/state/facts.json\"\n"
        "export PATH=\"%s/shim:$PATH\"\n" % (_posix(root), _posix(root)),
        encoding="utf-8",
    )
    return root


def _record(root: Path, *, test_environment: bool = True) -> None:
    """What a successful ``pm install`` leaves behind."""
    record_activation_inputs(root / "state" / "inputs", activation_input_mtimes(root), root,
                             test_environment=test_environment)


def _run(root: Path, sentinel: str | None = "{root}/state/facts.json") -> str:
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


def test_foreign_checkout_with_matching_input_times_reactivates(checkout: Path):
    _record(checkout)
    other = checkout.parent / "other"
    shutil.copytree(checkout, other, copy_function=shutil.copy2)
    assert "ACTIVATED" in _run(other, str(checkout / "state" / "facts.json"))


def test_recorded_inputs_are_left_alone(checkout: Path):
    """A checkout rewrote every input after facts.json was last written, then a
    no-op sync recorded them: the environment is current. Re-syncing on every
    run is the cost of getting this wrong."""
    for name in ACTIVATION_INPUTS:
        _set_mtime(checkout / name, JUST_AFTER)
    _record(checkout)
    assert "ACTIVATED" not in _run(checkout)


def test_runtime_only_install_cannot_mark_test_environment_current(checkout: Path):
    marker = checkout / "state" / "inputs" / ".test-environment"
    _record(checkout)
    assert marker.is_file()
    _record(checkout, test_environment=False)
    assert not marker.exists()
    assert "ACTIVATED" not in _run(checkout)  # The app is current; the test env isn't.


@pytest.mark.parametrize("moved_to", [JUST_AFTER, EARLIER], ids=["newer", "older"])
@pytest.mark.parametrize("input_name", ACTIVATION_INPUTS)
def test_input_mtime_differing_from_its_stamp_reactivates(checkout: Path, input_name: str, moved_to: str):
    """A branch switch can move an input's mtime either way; both mean the
    recorded install was not verified against what is on disk now."""
    _record(checkout)
    _set_mtime(checkout / input_name, moved_to)
    assert "ACTIVATED" in _run(checkout)


@pytest.mark.parametrize("sentinel", [None, "{root}/gone", "1", "{root}/state/facts.json"],
                         ids=["cold", "dangling", "legacy-1", "no-stamps"])
def test_unusable_sentinel_activates(checkout: Path, sentinel: str | None):
    """Cold, dangling, or incomplete activation state must not read as current."""
    assert "ACTIVATED" in _run(checkout, sentinel)
