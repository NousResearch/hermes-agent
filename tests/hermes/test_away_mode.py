"""Tests for plugins.life_ops.away_mode — the away-mode kill switch.

Shares todos.db with plugins.life_ops.todo_store but only touches its own
``away_state`` table (single row, id=1). HERMES_HOME is redirected to a
per-test tmp_path (same explicit-monkeypatch pattern used in
tests/hermes/test_rpe_command.py and tests/hermes/test_todo_store.py).
"""
from __future__ import annotations

import pytest

from plugins.life_ops import away_mode


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


class TestSetAwayWithUntilDate:
    def test_is_away_true_while_today_before_until(self):
        away_mode.set_away(until="2026-07-20")
        assert away_mode.is_away("2026-07-15") is True

    def test_is_away_true_on_boundary_today_equals_until(self):
        away_mode.set_away(until="2026-07-20")
        assert away_mode.is_away("2026-07-20") is True

    def test_is_away_false_after_until_and_auto_clears(self):
        away_mode.set_away(until="2026-07-10")

        assert away_mode.is_away("2026-07-15") is False

        # Auto-clear side effect: away_status must reflect inactive afterward.
        status = away_mode.away_status("2026-07-15")
        assert status == {"active": False, "until": None}

    def test_auto_clear_persists_across_reads(self):
        away_mode.set_away(until="2026-07-10")
        away_mode.is_away("2026-07-15")  # triggers auto-clear

        # A second, independent check should also see it cleared (not just
        # a return-value fluke — verify it stuck in the db).
        assert away_mode.is_away("2026-07-16") is False
        assert away_mode.away_status("2026-07-16") == {"active": False, "until": None}


class TestSetAwayIndefinite:
    def test_until_none_is_always_away(self):
        away_mode.set_away(until=None)
        assert away_mode.is_away("2026-07-15") is True
        assert away_mode.is_away("2099-01-01") is True

    def test_until_none_status_reports_active_until_none(self):
        away_mode.set_away(until=None)
        status = away_mode.away_status("2026-07-15")
        assert status == {"active": True, "until": None}


class TestClearAway:
    def test_manual_clear_turns_off(self):
        away_mode.set_away(until="2099-01-01")
        assert away_mode.is_away("2026-07-15") is True

        away_mode.clear_away()

        assert away_mode.is_away("2026-07-15") is False
        assert away_mode.away_status("2026-07-15") == {"active": False, "until": None}

    def test_clear_away_on_never_set_store_is_safe(self):
        away_mode.clear_away()
        assert away_mode.is_away("2026-07-15") is False


class TestAwayStatusNeverSet:
    def test_status_defaults_inactive(self):
        assert away_mode.away_status("2026-07-15") == {"active": False, "until": None}

    def test_is_away_defaults_false(self):
        assert away_mode.is_away("2026-07-15") is False


class TestSetAwayOverwritesPreviousState:
    def test_setting_again_replaces_until(self):
        away_mode.set_away(until="2026-07-10")
        away_mode.set_away(until="2026-08-01")
        status = away_mode.away_status("2026-07-15")
        assert status == {"active": True, "until": "2026-08-01"}
