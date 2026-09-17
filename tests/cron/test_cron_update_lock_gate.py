"""A due cron job must not fire while `hermes update` holds the install.

``tick()`` advances ``next_run_at`` BEFORE dispatch, and the code swap happens after that, so a job
firing in that window runs against a tree whose source has moved but whose venv/env has not. The
failure presents as an ImportError/ModuleNotFoundError naming a symbol that IS present on disk, or
a dead interpreter — and for a weekly job the next real run is a week away.

The gate returns BEFORE ``get_due_jobs()``, so deferring mutates nothing: no ``next_run_at``
advance, no execution row, no failure streak. The assertions are behaviour contracts, never
snapshots of message text.
"""

import json
import os
from pathlib import Path

import pytest

from cron import scheduler_preflight


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME (root) and the profile-level default root."""
    root = tmp_path / ".hermes"
    (root / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


def _write_marker(path: Path, pid: int) -> None:
    import time

    path.write_text(f"{pid}\n{int(time.time())}\n", encoding="utf-8")


def test_live_marker_defers_and_a_dead_one_never_wedges(hermes_home):
    """The two halves of the contract in one place: defer while live, never block when stale."""
    marker = hermes_home / ".hermes-update-in-progress"

    _write_marker(marker, 999_999_999)  # crashed updater
    assert scheduler_preflight.update_in_progress() is False
    assert not marker.exists(), "a dead-pid marker must be reaped, not honoured forever"

    _write_marker(marker, os.getpid())
    assert scheduler_preflight.update_in_progress() is True


def test_profile_gateway_sees_the_root_marker(tmp_path, monkeypatch):
    """A CLIENT PROFILE gateway must still see an update.

    ``hermes update`` writes the marker at the ROOT Hermes home, but a profile gateway runs with
    ``HERMES_HOME=<root>/profiles/<name>``. Probing only the process home silently no-ops for
    every client profile — exactly the case this gate exists for.
    """
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "sample-client"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    _write_marker(root / ".hermes-update-in-progress", os.getpid())
    assert scheduler_preflight.update_in_progress() is True


def test_deferring_leaves_the_job_store_untouched(hermes_home):
    """The outcome contract: no dispatch, and the schedule is not consumed.

    A deferred job must fire on a later tick, so ``next_run_at`` has to be byte-identical.
    """
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    jobs.create_job(name="weekly-report", prompt="do the thing", schedule="0 7 * * 1")
    before = json.loads((hermes_home / "cron" / "jobs.json").read_text())["jobs"]

    _write_marker(hermes_home / ".hermes-update-in-progress", os.getpid())
    fired = scheduler.tick(verbose=False, adapters=[], loop=None, sync=True)

    after = json.loads((hermes_home / "cron" / "jobs.json").read_text())["jobs"]
    assert fired == 0
    assert [j.get("next_run_at") for j in after] == [j.get("next_run_at") for j in before]
