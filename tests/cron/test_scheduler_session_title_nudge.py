"""Tests for cron/scheduler.py — cron session titling and the repeated-failure review nudge."""

from unittest.mock import MagicMock, patch


class TestSetCronSessionTitle:
    """Robust cron session titling: #50535/#50536/#50537."""

    def test_dedupes_on_duplicate_title(self):
        # First write collides (ValueError); helper falls back to lineage #N.
        from cron.scheduler import _set_cron_session_title
        db = MagicMock()
        db.set_session_title.side_effect = [ValueError("in use"), True]
        db.get_next_title_in_lineage.return_value = "Nightly Synthesis #2"
        out = _set_cron_session_title(db, "sess-1", "Nightly Synthesis")
        assert out == "Nightly Synthesis #2"
        db.get_next_title_in_lineage.assert_called_once_with("Nightly Synthesis")

class TestFailureStreakNudge:
    """Poke-inspired repeated-failure review nudge (_failure_streak_nudge)."""

    def _job(self, streak, kind="cron", name="scout"):
        return {
            "id": "j1",
            "name": name,
            "failure_streak": streak,
            "schedule": {"kind": kind},
        }

    def test_silent_below_threshold(self):
        from cron.scheduler import _failure_streak_nudge
        with patch("cron.scheduler.load_config", return_value={}):
            assert _failure_streak_nudge(self._job(0)) == ""
            assert _failure_streak_nudge(self._job(1)) == ""

    def test_oneshot_never_nudges(self):
        from cron.scheduler import _failure_streak_nudge
        with patch("cron.scheduler.load_config", return_value={}):
            assert _failure_streak_nudge(self._job(10, kind="once")) == ""

    def test_config_threshold_and_disable(self):
        from cron.scheduler import _failure_streak_nudge
        cfg5 = {"cron": {"failure_nudge_threshold": 5}}
        with patch("cron.scheduler.load_config", return_value=cfg5):
            assert _failure_streak_nudge(self._job(3)) == ""
            assert _failure_streak_nudge(self._job(4))
        with patch("cron.scheduler.load_config", return_value={"cron": {"failure_nudge_threshold": 0}}):
            assert _failure_streak_nudge(self._job(50)) == ""

    def test_missing_streak_field_backcompat(self):
        from cron.scheduler import _failure_streak_nudge
        job = {"id": "old", "schedule": {"kind": "interval"}}  # pre-field job
        with patch("cron.scheduler.load_config", return_value={}):
            assert _failure_streak_nudge(job) == ""
