"""``sessions.write_patience_s`` / ``sessions.transcript_write_patience_s`` (config.yaml).

The two write budgets are compiled defaults — 20s for routine writes, 60s for a turn's transcript
append. On a host whose ``state.db`` is large or contended, a sibling Hermes process legitimately
holds the single SQLite write lock for minutes (VACUUM after auto-prune, TRUNCATE checkpoint at
close, a long FTS pass), and the default budget then aborts the turn with
``session_persistence_failed`` even though the store is healthy and merely busy. Raising the budget
preserves the turn's already-done model work; failing and retrying re-runs — and re-pays for — the
whole turn. This suite pins the contract of the knob:

- unset => the class defaults stand and stay monkeypatchable, so nothing changes by default;
- a configured value lands on the handle and ACTUALLY lengthens the wait (proved against a lock
  hold the default budget cannot survive);
- a zero/negative/unparseable value is ignored rather than disabling the wait;
- the activity budget and the compression-lease wait stay compiled-in (scope boundary).
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


def _hold_write_lock(db_path, hold_s, started_evt):
    """Hold the SQLite write lock on *db_path* for *hold_s* seconds (a sibling maintenance process)."""
    conn = sqlite3.connect(str(db_path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started_evt.set()
        time.sleep(hold_s)
        conn.execute("COMMIT")
    finally:
        conn.close()


class TestResolution:
    def test_defaults_when_unconfigured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_WRITE_PATIENCE_S", raising=False)
        monkeypatch.delenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", raising=False)
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._WRITE_PATIENCE_S == 20.0
            assert db._TRANSCRIPT_WRITE_PATIENCE_S == 60.0
            # No instance attribute is materialised, so the class default (and a test's monkeypatch
            # of it) still governs every unconfigured host.
            assert "_WRITE_PATIENCE_S" not in db.__dict__
            assert "_TRANSCRIPT_WRITE_PATIENCE_S" not in db.__dict__
        finally:
            db.close()

    def test_env_carriers_override_each_budget_independently(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_WRITE_PATIENCE_S", "45")
        monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", "300")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._WRITE_PATIENCE_S == 45.0
            assert db._TRANSCRIPT_WRITE_PATIENCE_S == 300.0
        finally:
            db.close()

    def test_one_carrier_does_not_disturb_the_other_budget(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", raising=False)
        monkeypatch.setenv("HERMES_WRITE_PATIENCE_S", "45")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._WRITE_PATIENCE_S == 45.0
            assert db._TRANSCRIPT_WRITE_PATIENCE_S == 60.0
        finally:
            db.close()

    @pytest.mark.parametrize("value", ["0", "-1", "not-a-number", "", "   "])
    def test_unusable_value_falls_back_to_the_default(self, tmp_path, monkeypatch, value):
        """Patience is RAISED by configuration, never removed: a bad value must not disable the wait."""
        monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", value)
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._TRANSCRIPT_WRITE_PATIENCE_S == 60.0
        finally:
            db.close()

    def test_activity_and_compression_budgets_are_not_configurable(self, tmp_path, monkeypatch):
        """Scope guard: only the routine/transcript budgets are exposed.

        The activity budget is observation-only and must never lengthen a response-critical write;
        the compression-lease wait is a correctness boundary, not a tuning knob.
        """
        monkeypatch.setenv("HERMES_ACTIVITY_WRITE_PATIENCE_S", "9")
        monkeypatch.setenv("HERMES_COMPRESSION_BUSY_WAIT_S", "9")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._ACTIVITY_WRITE_PATIENCE_S == 0.5
            assert db._COMPRESSION_BUSY_WAIT_S == 5.0
        finally:
            db.close()

    def test_config_defaults_carry_both_keys(self):
        """The keys are registered so `hermes config set sessions.<key>` validates and lands."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        sessions = DEFAULT_CONFIG["sessions"]
        assert sessions["write_patience_s"] == 20
        assert sessions["transcript_write_patience_s"] == 60


class TestPrecedenceAndEffect:
    def test_configured_value_beats_the_class_default(self, tmp_path, monkeypatch):
        monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 0.2)
        monkeypatch.setenv("HERMES_WRITE_PATIENCE_S", "30")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._WRITE_PATIENCE_S == 30.0
            assert SessionDB._WRITE_PATIENCE_S == 0.2  # the class default itself is untouched
        finally:
            db.close()

    @staticmethod
    def _budget_seen_by_the_write_path(db, monkeypatch, call):
        """The ``patience_s`` the write path ends up using for *call* — the end of the wiring.

        ``_execute_write`` resolves an explicit ``patience_s`` first and falls back to
        ``self._WRITE_PATIENCE_S`` when the caller passes none, so the spy applies the same fallback.
        """
        seen = {}

        def spy(self, fn, patience_s=None):
            seen["patience_s"] = patience_s if patience_s is not None else self._WRITE_PATIENCE_S
            return real(self, fn, patience_s=patience_s)

        real = SessionDB._execute_write
        monkeypatch.setattr(SessionDB, "_execute_write", spy)
        db.create_session("s1", "cli")
        call(db)
        return seen.get("patience_s")

    def test_configured_transcript_budget_reaches_the_transcript_write(self, tmp_path, monkeypatch):
        """config -> env -> handle -> the value ``append_message`` hands the write path."""
        monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", "300")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert self._budget_seen_by_the_write_path(
                db, monkeypatch, lambda d: d.append_message(session_id="s1", role="user", content="x")
            ) == 300.0
        finally:
            db.close()

    def test_configured_routine_budget_reaches_a_routine_write(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_WRITE_PATIENCE_S", "45")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert self._budget_seen_by_the_write_path(
                db, monkeypatch, lambda d: d.set_meta("k", "v")
            ) == 45.0
        finally:
            db.close()

    def test_unconfigured_writes_get_the_compiled_defaults(self, tmp_path, monkeypatch):
        """The negative control for the two tests above: with nothing configured the defaults arrive."""
        monkeypatch.delenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", raising=False)
        monkeypatch.delenv("HERMES_WRITE_PATIENCE_S", raising=False)
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert self._budget_seen_by_the_write_path(
                db, monkeypatch, lambda d: d.append_message(session_id="s1", role="user", content="x")
            ) == 60.0
            assert self._budget_seen_by_the_write_path(
                db, monkeypatch, lambda d: d.set_meta("k", "v")
            ) == 20.0
        finally:
            db.close()

    def test_configured_budget_is_a_ceiling_that_costs_nothing_uncontended(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", "300")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s2", "cli")
            t0 = time.monotonic()
            db.append_message(session_id="s2", role="user", content="fast")
            assert time.monotonic() - t0 < 5.0
        finally:
            db.close()

    def test_configured_budget_rides_out_a_multi_second_sibling_hold(self, tmp_path, monkeypatch):
        """The production scenario: a sibling maintenance process holds the lock for seconds."""
        monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", "30")
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s3", "cli")
            started = threading.Event()
            holder = threading.Thread(target=_hold_write_lock, args=(db.db_path, 3.0, started))
            holder.start()
            try:
                assert started.wait(5.0)
                t0 = time.monotonic()
                db.append_message(session_id="s3", role="user", content="waited it out")
                assert time.monotonic() - t0 >= 1.0  # it genuinely contended; the append did not race past
            finally:
                holder.join(timeout=15.0)
            assert not holder.is_alive()
            assert any(m["content"] == "waited it out" for m in db.get_messages("s3"))
        finally:
            db.close()

