"""Regression tests for a non-numeric interval ``minutes``.

``schedule["minutes"]`` comes straight from ``jobs.json``, which is routinely
hand-edited — the id / schedule / next_run_at / last_run_at repair passes in
``_get_due_jobs_locked`` exist for exactly that. A quoted number
(``"minutes": "60"``) made every consumer raise ``TypeError`` on the arithmetic:

* ``compute_next_run`` — ``timedelta(minutes="60")``; its ``except`` fallback
  could not help because it re-evaluates the same failing expression.
* ``_compute_grace_seconds`` — ``"60" * 60`` then ``// 2``.

In the due scan the per-job structural guard swallowed that, so the job was
skipped on EVERY tick — silently disabled forever, the failure mode the sibling
repair passes exist to prevent. ``advance_next_run`` / ``mark_job_run`` have no
such guard and raised outright.
"""
from datetime import datetime, timedelta, timezone

from cron.jobs import (
    _coerce_interval_minutes,
    _compute_grace_seconds,
    compute_next_run,
)


def _past(minutes: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


class TestCoerceIntervalMinutes:
    def test_numbers_pass_through(self):
        assert _coerce_interval_minutes(60) == 60.0
        assert _coerce_interval_minutes(1.5) == 1.5

    def test_quoted_number_is_coerced(self):
        assert _coerce_interval_minutes("60") == 60.0
        assert _coerce_interval_minutes(" 60 ") == 60.0

    def test_bool_is_rejected(self):
        # bool is an int subclass; "minutes": true is corruption, not 1 minute.
        assert _coerce_interval_minutes(True) is None

    def test_garbage_is_rejected(self):
        for bad in ("", "later", {}, [], None, object()):
            assert _coerce_interval_minutes(bad) is None


class TestComputeNextRunWithStringMinutes:
    def test_quoted_minutes_still_schedules(self):
        got = compute_next_run({"kind": "interval", "minutes": "60"}, _past())
        assert got is not None
        assert datetime.fromisoformat(got) > datetime.now(timezone.utc)

    def test_quoted_minutes_without_last_run(self):
        assert compute_next_run({"kind": "interval", "minutes": "30"}) is not None

    def test_garbage_minutes_returns_none_instead_of_raising(self):
        for bad in ("", "soon", {}, True):
            assert compute_next_run({"kind": "interval", "minutes": bad}, _past()) is None


class TestGraceSecondsWithStringMinutes:
    def test_quoted_minutes_grace_matches_numeric(self):
        assert (
            _compute_grace_seconds({"kind": "interval", "minutes": "60"})
            == _compute_grace_seconds({"kind": "interval", "minutes": 60})
        )

    def test_garbage_minutes_falls_back_to_min_grace(self):
        assert _compute_grace_seconds({"kind": "interval", "minutes": {}}) == 120


class TestDueScanSelfHeals:
    def test_job_with_quoted_minutes_becomes_due_and_is_repaired(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import importlib

        import cron.jobs as jobs_mod

        importlib.reload(jobs_mod)
        jobs_mod.save_jobs([{
            "id": "j1", "name": "nightly", "enabled": True,
            "schedule": {"kind": "interval", "minutes": "60"},
            "next_run_at": _past(), "prompt": "hi",
        }])

        due = jobs_mod.get_due_jobs()
        assert [d["id"] for d in due] == ["j1"], "job was silently skipped"

        # Repaired and persisted, so it never degrades again.
        healed = jobs_mod.load_jobs()[0]["schedule"]["minutes"]
        assert healed == 60 and isinstance(healed, int)

        # The paths that previously raised now succeed.
        assert jobs_mod.advance_next_run("j1") is True
        jobs_mod.mark_job_run("j1", True)
