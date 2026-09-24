"""A released stale claim must already have a durable terminal outcome on disk.

``sweep_stale_inflight`` used to mutate the in-flight guard (``_running_job_ids`` and friends) in
its classification loop and only afterwards write the release record — and for an ``age`` release of
a job with a finite repeat budget it deliberately wrote no ``last_error`` at all. A released claim
whose ledger row was still ``claimed``/``running`` therefore left nobody able to say whether its
side effects ran (#115692: "Owner process is still alive but the claim outlived the derived stale
bound ... whether side effects ran is unknown").

These tests pin the ordering: the durable outcome is written BEFORE the claim is dropped, an open
ledger row is terminalised first, and a claim whose outcome cannot be written stays held (fail
closed) instead of being silently released.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import cron.scheduler as sched
from cron import executions as executions_mod


def _job(job_id="wedged", minutes=60):
    return {
        "id": job_id,
        "name": job_id,
        "schedule": {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"},
    }


@pytest.fixture(autouse=True)
def _clean_inflight():
    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_since.clear()
    sched._running_futures.clear()


def _claim(job, home, *, age_seconds=5 * 60 * 60):
    """Register an aged, submit-hung claim for ``job`` and return (key, started)."""
    key = sched._inflight_key(job["id"], home)
    started = time.time() - age_seconds
    sched._running_job_ids.add(key)
    sched._running_since[key] = started
    sched._running_futures[key] = sched._FUTURE_PENDING
    return key, started


def test_durable_outcome_is_written_before_the_claim_is_released(tmp_path):
    """The claim is still held while its outcome is recorded — after the release nothing can
    inspect the run, so the answer must already be on disk."""
    job = _job()
    key, started = _claim(job, tmp_path)
    seen = {}

    def _record(job_id, claim_started):
        seen["held"] = key in sched._running_job_ids
        seen["started"] = sched._running_since.get(key)
        seen["job_id"] = job_id
        seen["claim_started"] = claim_started

    with patch.object(sched, "_terminalise_claim_if_open", side_effect=_record), \
         patch.object(sched, "mark_job_run"), \
         patch.object(sched, "_get_hermes_home", return_value=tmp_path):
        assert sched.sweep_stale_inflight([job]) == [job["id"]]

    assert seen == {"held": True, "started": started, "job_id": job["id"],
                    "claim_started": started}
    assert key not in sched._running_job_ids


@pytest.mark.parametrize("failing", ["_terminalise_claim_if_open", "_record_stale_release"])
def test_claim_is_retained_when_the_durable_outcome_cannot_be_written(tmp_path, failing):
    """Fail closed: a claim whose outcome could not be persisted is NOT released — the next sweep
    retries rather than losing the attempt with no record of how it ended."""
    job = _job()
    key, _started = _claim(job, tmp_path)

    patches = [
        patch.object(sched, "mark_job_run"),
        patch.object(sched, "_get_hermes_home", return_value=tmp_path),
    ]
    if failing == "_terminalise_claim_if_open":
        patches.append(patch.object(sched, "_terminalise_claim_if_open",
                                    side_effect=RuntimeError("ledger locked")))
    else:
        patches.append(patch.object(sched, "_terminalise_claim_if_open"))
        patches.append(patch.object(sched, "_record_stale_release",
                                    side_effect=RuntimeError("jsonl unwritable")))

    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        assert sched.sweep_stale_inflight([job]) == []

    assert key in sched._running_job_ids
    assert job["id"] in sched.get_running_job_ids()


def test_age_release_terminalises_the_open_ledger_row(tmp_path):
    """An age release leaves the ledger with a terminal 'unknown' row and the #115692 reason, so a
    reader can tell the attempt ended and that its side effects are unknown."""
    job = _job()
    row = executions_mod.create_execution(job["id"], source="builtin")
    assert row["status"] == "claimed"
    key, _started = _claim(job, tmp_path)

    with patch.object(sched, "mark_job_run"), \
         patch.object(sched, "_get_hermes_home", return_value=tmp_path):
        assert sched.sweep_stale_inflight([job]) == [job["id"]]

    after = executions_mod.get_execution(row["id"])
    assert after["status"] == "unknown"
    assert after["finished_at"]
    assert "115692" in after["error"]
    assert key not in sched._running_job_ids


def test_mark_claim_unknown_leaves_a_row_from_a_different_claim_alone(tmp_path):
    """Only the claim's own row is terminalised: a newer attempt's row must not be reaped."""
    row = executions_mod.create_execution("job-x", source="builtin")

    assert executions_mod.mark_claim_unknown("job-x", claimed_after=time.time() + 3600) is None
    assert executions_mod.get_execution(row["id"])["status"] == "claimed"

    # ... and the matching claim's row is the one that gets the terminal state.
    updated = executions_mod.mark_claim_unknown("job-x", claimed_after=time.time() - 3600,
                                                reason="released stale (#115692)")
    assert updated is not None and updated["id"] == row["id"]
    assert updated["status"] == "unknown"
    assert updated["error"] == "released stale (#115692)"
    # Terminal rows are immutable: a second call finds nothing open to touch.
    assert executions_mod.mark_claim_unknown("job-x", claimed_after=time.time() - 3600) is None
    assert executions_mod.get_execution(row["id"])["error"] == "released stale (#115692)"