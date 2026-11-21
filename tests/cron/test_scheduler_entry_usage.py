"""`python -m cron.scheduler` must never run a tick by accident (#152710).

Before this guard the entry point only built its parser when ``--external-worker-file`` was on the
command line; every other invocation — including ``--help`` — fell straight through to
``tick(verbose=True, headless=True)`` and fired every due job on the home it found. Asking for help
ran the scheduler.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(args: list[str], home: pathlib.Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
        # Keep the child from inheriting a caller's profile/tool overrides.
        "NO_COLOR": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "cron.scheduler", *args],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=120, check=False,
    )


def _tick_ran(home: pathlib.Path) -> bool:
    return (home / "cron" / ".tick.lock").exists()


@pytest.mark.parametrize(
    "args, rc",
    [
        (["--help"], 0),
        (["-h"], 0),
        ([], 2),
        (["--nope"], 2),
    ],
)
def test_help_and_unknown_invocations_never_tick(tmp_path, args, rc):
    done = _run(args, tmp_path)
    stream = done.stdout if rc == 0 else done.stderr
    assert done.returncode == rc, (done.stdout, done.stderr)
    assert "usage: python -m cron.scheduler" in stream
    assert not _tick_ran(tmp_path), "a bare/help invocation ran a tick"


def test_the_explicit_tick_still_runs(tmp_path):
    done = _run(["--tick"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert _tick_ran(tmp_path), "--tick no longer runs the scheduler"


def test_help_wins_even_with_the_worker_flag(tmp_path):
    """kvnloo's case: `--external-worker-file X --help` must print usage and exit 0, not argparse-error.

    Handled before the worker branch on purpose: booting the worker may relaunch the process, which
    must not happen for a help request.
    """
    done = _run(["--external-worker-file", str(tmp_path / "payload.json"), "--help"], tmp_path)
    assert done.returncode == 0, done.stderr
    assert "usage: python -m cron.scheduler" in done.stdout
    assert not _tick_ran(tmp_path)


def test_the_worker_flag_keeps_its_own_parser(tmp_path):
    """The gateway launches this path; it must still refuse a half-specified worker, not tick."""
    done = _run(["--external-worker-file", str(tmp_path / "payload.json")], tmp_path)
    assert done.returncode == 2
    assert "--ack-file" in done.stderr
    assert not _tick_ran(tmp_path)
