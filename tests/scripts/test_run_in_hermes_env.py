"""``scripts/run-in-hermes-env``: run a command in the Hermes environment, syncing only when needed.

The runner is what repo scripts hand themselves to from their shebang and what
``scripts/run_tests.sh`` re-executes under. Its decision worth pinning is
staleness: pm stamps each dependency input's mtime beside the installed-state
file ``__HERMES_ACTIVATED`` names, and any input whose mtime differs from its
stamp means the inherited environment may not match its inputs. Invert that
either way and the cost is invisible — a re-sync on every run, or a stale
environment that looks fine.

The stamps come from the real ``pm.environments`` writer, so these tests pin
the contract between what pm records and what the runner reads. Setup and the
pm environment reader are stubbed at their process boundary: setup announces
each run, the reader emits a sentinel and a PATH shim.

A ``python3`` shim goes on PATH because scripts run under the runner name
``python3``, which does not exist on stock Windows; the subject here is the
staleness branch, not interpreter resolution.
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
from tests.pm.activation_support import bash, posix

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run-in-hermes-env"

EARLIER = "2018-01-01 00:00:00"
LONG_AGO = "2019-01-01 00:00:00"
STAMP_TIME = "2020-06-01 00:00:00"
JUST_AFTER = "2021-01-01 00:00:00"


def _set_mtime(path: Path, stamp: str) -> None:
    when = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").timestamp()
    os.utime(path, (when, when))


def _shim(path: Path) -> None:
    """Quote in posix form: a /bin/sh script treats backslashes in an unquoted
    word as escapes (same pattern as tests/pm/activation_support.py)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexec '%s' \"$@\"\n" % posix(Path(sys.executable)), encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A repo-shaped tree whose stub setup announces each run; its installed
    state lives in ``state/`` (facts.json beside the input stamps)."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "pm").mkdir()
    (root / "state").mkdir()
    for name in (RUNNER.name, "_activation.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, root / "scripts" / name)
    for name in ACTIVATION_INPUTS:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
        _set_mtime(root / name, LONG_AGO)
    (root / "state" / "facts.json").touch()
    _set_mtime(root / "state" / "facts.json", STAMP_TIME)

    _shim(root / "shim" / "python3")
    _shim(root / ".venv" / "bin" / "python")  # the bootstrap interpreter that reads the pm env

    (root / "setup-hermes.sh").write_text(
        "echo SYNCED >&2\n"
        f'test ! -e "{posix(root)}/fail"\n', encoding="utf-8",
    )
    (root / "pm" / "__init__.py").touch()
    (root / "pm" / "environments.py").write_text(
        "import os, shlex, sys\n"
        "assert sys.argv[1:] == ['--format', 'sh'], sys.argv\n"
        f"print(\"export __HERMES_ACTIVATED='{posix(root)}/state/facts.json'\")\n"
        f"print('export PATH=\"{posix(root)}/shim:$PATH\"')\n"
        "print('export HERMES_HOME=' + shlex.quote(os.environ['HERMES_HOME']))\n",
        encoding="utf-8",
    )
    return root


def _record(root: Path, *, test_environment: bool = True) -> None:
    """What a successful ``pm install`` leaves behind."""
    record_activation_inputs(root / "state" / "inputs", activation_input_mtimes(root), root,
                             test_environment=test_environment)


def _run_in(
    root: Path,
    *command: str,
    sentinel: str | None = "{root}/state/facts.json",
    env_overrides: dict[str, str] | None = None,
):
    """Drive the runner as a caller would; the shim dir is on PATH like any tool dir."""
    env = {**os.environ, "PATH": f"{posix(root / 'shim')}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.update(env_overrides or {})
    env.pop("__HERMES_ACTIVATED", None)
    if sentinel is not None:
        env["__HERMES_ACTIVATED"] = sentinel.replace("{root}", posix(root))
    return subprocess.run(
        [bash(), posix(root / "scripts" / RUNNER.name), *command],
        capture_output=True, text=True, cwd=posix(root), env=env, timeout=60,
    )


def _synced(root: Path, sentinel: str | None = "{root}/state/facts.json") -> bool:
    run = _run_in(root, "true", sentinel=sentinel)
    assert run.returncode == 0, run.stderr
    return "SYNCED" in run.stderr


def test_foreign_checkout_with_matching_input_times_resyncs(checkout: Path):
    _record(checkout)
    other = checkout.parent / "other"
    shutil.copytree(checkout, other, copy_function=shutil.copy2)
    assert _synced(other, str(checkout / "state" / "facts.json"))


def test_recorded_inputs_are_left_alone(checkout: Path):
    """A checkout rewrote every input after facts.json was last written, then a
    no-op sync recorded them: the environment is current. Re-syncing on every
    run is the cost of getting this wrong."""
    for name in ACTIVATION_INPUTS:
        _set_mtime(checkout / name, JUST_AFTER)
    _record(checkout)
    assert not _synced(checkout)


def test_install_that_skipped_the_test_environment_is_stale(checkout: Path):
    """Every environment the runner hands out includes the test environment,
    which a plain runtime install cannot mark current."""
    marker = checkout / "state" / "inputs" / ".test-environment"
    _record(checkout)
    assert marker.is_file()
    _record(checkout, test_environment=False)
    assert not marker.exists()
    assert _synced(checkout)


@pytest.mark.parametrize("moved_to", [JUST_AFTER, EARLIER], ids=["newer", "older"])
@pytest.mark.parametrize("input_name", ACTIVATION_INPUTS)
def test_input_mtime_differing_from_its_stamp_resyncs(checkout: Path, input_name: str, moved_to: str):
    """A branch switch can move an input's mtime either way; both mean the
    recorded install was not verified against what is on disk now."""
    _record(checkout)
    _set_mtime(checkout / input_name, moved_to)
    assert _synced(checkout)


@pytest.mark.parametrize("sentinel", [None, "{root}/gone", "1", "{root}/state/facts.json"],
                         ids=["cold", "dangling", "legacy-1", "no-stamps"])
def test_unusable_sentinel_syncs(checkout: Path, sentinel: str | None):
    """Cold, dangling, or incomplete activation state must not read as current."""
    assert _synced(checkout, sentinel)


def test_command_runs_in_the_environment_with_its_arguments_and_exit_status(checkout: Path):
    run = _run_in(checkout, "bash", "-c", 'echo "$__HERMES_ACTIVATED|$0|$1|$2"; exit 7', "zero", "b c", "d",
                  sentinel=None)
    assert run.returncode == 7, run.stderr
    assert run.stdout.strip() == f"{posix(checkout)}/state/facts.json|zero|b c|d"


def test_unicode_home_survives_legacy_python_stdout_encoding(checkout: Path):
    unicode_home = str(checkout.parent / "Hermes Ω 空間")
    run = _run_in(
        checkout,
        "bash",
        "-c",
        'printf %s "$HERMES_HOME"',
        sentinel=None,
        env_overrides={
            "HERMES_HOME": unicode_home,
            "PYTHONIOENCODING": "cp1252:strict",
        },
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout == unicode_home


def test_failed_sync_runs_nothing(checkout: Path):
    (checkout / "fail").touch()
    ran = checkout / "ran"
    run = _run_in(checkout, "touch", posix(ran), sentinel=None)
    assert run.returncode != 0
    assert not ran.exists()


def test_no_command_is_a_usage_error(checkout: Path):
    run = _run_in(checkout, sentinel=None)
    assert run.returncode == 2
    assert "usage" in run.stderr
