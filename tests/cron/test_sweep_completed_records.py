"""The completed-record sweep must consider every schedule that can never fire again.

A recurring job with a finite ``repeat.times`` reaches ``state == "completed"`` like a one-shot does,
but the sweep used to look only at ``schedule.kind == "once"``, so those records were immortal. The
widened rule has to stay conservative in the other direction: an unbounded cron has no budget, so
even a ``completed`` state must not be enough to delete it.
"""
from datetime import UTC, datetime, timedelta

import pytest

from cron import jobs


NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
OLD = (NOW - timedelta(days=30)).isoformat()
FRESH = (NOW - timedelta(days=1)).isoformat()


def _record(job_id="job-1", *, kind="once", repeat=None, last_run_at=OLD, state="completed"):
    rec = {
        "id": job_id,
        "name": f"record {job_id}",
        "state": state,
        "schedule": {"kind": kind, "expr": "0 9 30 9 *"} if kind else None,
        "last_run_at": last_run_at,
    }
    if repeat is not None:
        rec["repeat"] = repeat
    return rec


@pytest.fixture(autouse=True)
def _fixed_retention(monkeypatch):
    """Pin the window so the cases test the schedule rule, not the operator's config."""
    monkeypatch.setattr(jobs, "_completed_oneshot_retention_days", lambda: 7.0)


@pytest.mark.parametrize(
    "kind,repeat",
    [
        ("once", None),                                  # the historical case, unchanged
        ("cron", {"times": 1, "completed": 1}),          # budget spent
        ("cron", {"times": 3, "completed": 3}),          # spent, and spent exactly
        ("cron", {"times": 3, "completed": 4}),          # over-spent (a replay cannot un-finish it)
    ],
)
def test_finished_schedule_is_swept(kind, repeat):
    raw = [_record(kind=kind, repeat=repeat)]
    removed = set()
    assert jobs._sweep_completed_oneshots(raw, NOW, removed_ids=removed) is True
    assert raw == []
    assert removed == {"job-1"}


@pytest.mark.parametrize(
    "kind,repeat",
    [
        ("cron", None),                                  # unbounded: no repeat block at all
        ("cron", {"times": 0, "completed": 0}),           # 0 == unbounded in this store
        ("cron", {"times": None, "completed": 5}),        # unbounded, explicitly null
        ("cron", {"times": 5, "completed": 2}),           # budget remains
        ("cron", {"times": "x", "completed": 1}),         # unreadable budget
        ("cron", {}),                                     # unreadable, empty
        ("cron", "nonsense"),                             # unreadable, wrong type
        (None, {"times": 1, "completed": 1}),             # no schedule at all
    ],
)
def test_unfinished_or_unreadable_schedule_is_kept(kind, repeat):
    raw = [_record(kind=kind, repeat=repeat)]
    assert jobs._sweep_completed_oneshots(raw, NOW) is False
    assert raw != []


def test_finished_but_fresh_record_is_kept():
    raw = [_record(kind="cron", repeat={"times": 1, "completed": 1}, last_run_at=FRESH)]
    assert jobs._sweep_completed_oneshots(raw, NOW) is False
    assert raw != []


def test_finished_record_without_parseable_last_run_is_kept():
    raw = [_record(kind="cron", repeat={"times": 1, "completed": 1}, last_run_at="not a timestamp")]
    assert jobs._sweep_completed_oneshots(raw, NOW) is False
    assert raw != []


def test_non_completed_state_is_kept_regardless_of_budget():
    raw = [_record(kind="cron", repeat={"times": 1, "completed": 1}, state="scheduled")]
    assert jobs._sweep_completed_oneshots(raw, NOW) is False
    assert raw != []


def test_mixed_store_sweeps_only_the_finished_records():
    finished = _record("done-cron", kind="cron", repeat={"times": 2, "completed": 2})
    immortal = _record("live-cron", kind="cron", repeat={"times": 0, "completed": 9})
    one_shot = _record("done-once", kind="once")
    raw = [finished, immortal, one_shot]
    removed = set()
    assert jobs._sweep_completed_oneshots(raw, NOW, removed_ids=removed) is True
    assert raw == [immortal]
    assert removed == {"done-cron", "done-once"}


def test_disabled_retention_sweeps_nothing(monkeypatch):
    monkeypatch.setattr(jobs, "_completed_oneshot_retention_days", lambda: 0.0)
    raw = [_record(kind="cron", repeat={"times": 1, "completed": 1})]
    assert jobs._sweep_completed_oneshots(raw, NOW) is False
    assert raw != []
