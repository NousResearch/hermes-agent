"""A build that fails under update custody is reported by its own command.

While the checkout lock is held, ``contained_command`` starts the build behind a launcher (POSIX:
the reap-tree script; Windows inside an update: the job-joining one). The failure used to carry
that launcher's argv, so ``⚠ <step> failed`` printed the launcher's Python source, and the owed
follow-up line (cut at 500 characters) never reached the build command or its exit code.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from hermes_cli import update_lock as ul
from hermes_cli.source_build import _failure_text, run_in_custody


@pytest.mark.platforms("any")  # each OS has its own launcher
# Windows: the job-joining launcher's argv reads as `hermes update` to the live-system guard; the
# build here is a no-op against a tmp_path checkout (as in test_update_lock_windows_live.py).
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("verbose", ["1", "0"], ids=["streamed", "captured"])
def test_a_failed_build_in_custody_names_its_own_command(tmp_path, monkeypatch, verbose):
    monkeypatch.setenv("HERMES_VERBOSE", verbose)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    build = [sys.executable, "-c", "raise SystemExit(3)"]
    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=checkout)
    assert lock.acquire()
    try:
        with pytest.raises(subprocess.CalledProcessError) as failed:
            run_in_custody(checkout, build, "probe build", stdin=subprocess.DEVNULL)
    finally:
        lock.release()
    assert failed.value.returncode == 3
    assert failed.value.cmd == build
    assert _failure_text(failed.value) == f"{sys.executable} -c raise SystemExit(3) exited 3"
