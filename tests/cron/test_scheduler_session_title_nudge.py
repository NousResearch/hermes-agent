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


class TestCronTitleFallbacks:
    """Recovery candidates never raise and keep the non-blank guarantee (#146855)."""

    def test_lineage_candidate_wins_when_available(self):
        from cron.scheduler import _cron_title_fallbacks
        db = MagicMock()
        db.get_next_title_in_lineage.return_value = "Nightly Synthesis #2"
        out = _cron_title_fallbacks(db, "j1", "sess-1")
        assert out == ["Nightly Synthesis #2", "cron j1 sess-1", "cron j1"]

    def test_raising_lineage_lookup_does_not_propagate(self):
        # The lineage lookup shares the bounded DB with the write that just failed and can
        # raise; recovery must survive it instead of aborting run_job's cleanup.
        from cron.scheduler import _cron_title_fallbacks
        db = MagicMock()
        db.get_next_title_in_lineage.side_effect = RuntimeError("session finalization disabled")
        out = _cron_title_fallbacks(db, "j1", "sess-1")
        assert out == ["cron j1 sess-1", "cron j1"]

    def test_missing_lineage_support_still_yields_candidates(self):
        from cron.scheduler import _cron_title_fallbacks
        db = MagicMock(spec=[])  # no get_next_title_in_lineage at all
        out = _cron_title_fallbacks(db, "j1", "sess-1")
        assert out == ["cron j1 sess-1", "cron j1"]

    def test_unset_session_id_still_guarantees_nonblank(self):
        from cron.scheduler import _cron_title_fallbacks
        db = MagicMock()
        db.get_next_title_in_lineage.side_effect = RuntimeError("session finalization disabled")
        out = _cron_title_fallbacks(db, "j1", None)
        assert out == ["cron j1"]
        assert all(candidate.strip() for candidate in out)
