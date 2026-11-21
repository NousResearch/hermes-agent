"""Regression tests for the orphaned fire-owner count (Part B).

``cron/scheduler.sweep_stale_inflight`` only reaped ``_running_job_ids``
while ``get_running_job_ids()`` — the count the gateway shutdown drain
reports as ``active_agents`` — is the union with ``_running_fire_owners``.
An orphaned fire-owner entry pinned the count at >=1 with no sweeper able
to reap it. The sweep now also reaps local-home fire-owner entries whose
own executions-ledger row is terminal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import cron.scheduler as sched
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _Scope:
    """Bind the cron home scope the way ``_profile_cron_scope`` does for a tick."""

    def __init__(self, home):
        self._home = str(home)

    def __enter__(self):
        self._token = set_hermes_home_override(self._home)
        return self

    def __exit__(self, *_exc):
        reset_hermes_home_override(self._token)
        return False


@pytest.fixture(autouse=True)
def _clean_inflight():
    """Reset the in-memory running sets so tests are isolated."""
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    sched._running_registration_owners.clear()
    sched._running_allowance_s.clear()
    sched._forced_releases.clear()
    sched._forced_release_count = 0
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    sched._running_registration_owners.clear()
    sched._running_allowance_s.clear()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def test_sweep_reaps_terminal_fire_owner(tmp_path):
    """(1) The sweep reaps a planted fire-owner entry whose ledger row is terminal."""
    home = tmp_path / "profile"
    home.mkdir()
    job_id = "orphaned-fire"
    key = sched._inflight_key(job_id, home)
    sched._running_fire_owners[key] = {object(): ("owner-1", home.resolve())}
    row = {"status": "completed", "id": "exec-1", "claimed_at": _now_iso()}
    with _Scope(home), \
            patch("cron.executions.latest_executions", return_value={job_id: row}), \
            patch.object(sched, "mark_job_run") as mark:
        released = sched.sweep_stale_inflight([])
    assert released == [job_id]
    assert sched._running_fire_owners == {}
    assert sched.get_running_job_ids() == frozenset()
    mark.assert_not_called()  # ledger-terminal must not clobber the real outcome
    assert sched.get_inflight_guard_stats()["forced_releases"] == 1


def test_sweep_keeps_live_fire_owner_and_claim(tmp_path):
    """(2) Negative control: a still-running fire-owner entry (non-terminal
    ledger row) and a live claim keep the count >= 1; the sweep reaps neither."""
    home = tmp_path / "profile"
    home.mkdir()
    fire_id, claim_id = "live-fire", "live-claim"
    with _Scope(home):
        assert sched.try_register_running_job(claim_id) is True
    try:
        sched._running_fire_owners[sched._inflight_key(fire_id, home)] = {
            object(): ("owner-9", home.resolve())
        }
        rows = {fire_id: {"status": "running", "id": "exec-live", "claimed_at": _now_iso()}}
        with _Scope(home), \
                patch("cron.executions.latest_executions", return_value=rows), \
                patch.object(sched, "mark_job_run") as mark:
            released = sched.sweep_stale_inflight([])
        assert released == []
        assert sched.get_running_job_ids() == frozenset({claim_id, fire_id})
        mark.assert_not_called()
    finally:
        sched.release_running_job(claim_id, home=home)
