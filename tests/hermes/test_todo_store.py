"""Tests for plugins.life_ops.todo_store — the persistent stable-key todo store.

Covers the acceptance criteria for the morning-run reconciliation
(:func:`upsert_from_contract` / :func:`plan_upsert_from_contract`), the
standalone snooze-expiry sweep, the close/done/dismiss/snooze lifecycle, the
read helpers used for Discord/journal injection, and the audit log.

THE core bug fix under test: an open key that simply does not reappear in a
given day's contract (because journal didn't re-propose it, not because it
was closed) must be left completely untouched — not silently dropped, not
re-touched. See ``TestUpsertFromContract.test_open_key_missing_from_contract_left_untouched``.

HERMES_HOME is redirected to a per-test ``tmp_path`` (matching the explicit
``monkeypatch.setenv("HERMES_HOME", ...)`` pattern used in
tests/hermes/test_rpe_command.py) so every test gets an isolated todos.db,
even though tests/conftest.py's autouse fixture already does this implicitly.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from plugins.life_ops import todo_store as ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(key, text="do the fake-widget-42 review", priority="medium", source_dates=None, **kw):
    item = {
        "key": key,
        "text": text,
        "priority": priority,
        "source_dates": source_dates if source_dates is not None else ["2026-07-14"],
    }
    item.update(kw)
    return item


def _row(tmp_path, key) -> sqlite3.Row | None:
    conn = ts.connect(tmp_path / "todos.db")
    try:
        return conn.execute("SELECT * FROM todos WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()


def _row_dict(tmp_path, key) -> dict | None:
    row = _row(tmp_path, key)
    return dict(row) if row is not None else None


def _all_rows(tmp_path) -> list[sqlite3.Row]:
    conn = ts.connect(tmp_path / "todos.db")
    try:
        return conn.execute("SELECT * FROM todos").fetchall()
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# upsert_from_contract
# ---------------------------------------------------------------------------


class TestUpsertFromContract:
    def test_new_key_is_inserted_open(self, tmp_path):
        plan = ts.upsert_from_contract([_item("fake-key-1", text="buy synthetic-widget-42")], "2026-07-14")

        assert plan["inserted"] == ["fake-key-1"]
        assert plan["refreshed"] == []
        row = _row_dict(tmp_path, "fake-key-1")
        assert row is not None
        assert row["status"] == "open"
        assert row["text"] == "buy synthetic-widget-42"
        assert row["first_seen"] == "2026-07-14"
        assert row["last_seen"] == "2026-07-14"

    def test_open_key_reappearing_is_refreshed_status_untouched(self, tmp_path):
        ts.upsert_from_contract(
            [_item("fake-key-1", text="old fake text", priority="low", source_dates=["2026-07-14"])],
            "2026-07-14",
        )
        plan = ts.upsert_from_contract(
            [_item("fake-key-1", text="new fake text", priority="high", source_dates=["2026-07-15"])],
            "2026-07-15",
        )

        assert plan["refreshed"] == ["fake-key-1"]
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["text"] == "new fake text"
        assert row["priority"] == "high"
        assert json.loads(row["source_dates"]) == ["2026-07-15"]
        assert row["last_seen"] == "2026-07-15"
        assert row["status"] == "open"

    def test_open_key_missing_from_contract_left_untouched(self, tmp_path):
        """THE core bug fix: absent from today's contract != closed.

        First call inserts fake-key-1; second call's contract omits it
        entirely (as if journal simply didn't re-propose it that day). The
        row must stay open and every field must be byte-for-byte unchanged.
        """
        ts.upsert_from_contract(
            [_item("fake-key-1", text="fake task alpha", priority="low", source_dates=["2026-07-14"])],
            "2026-07-14",
        )
        before = _row_dict(tmp_path, "fake-key-1")
        assert before["status"] == "open"

        plan = ts.upsert_from_contract([], "2026-07-15")

        after = _row_dict(tmp_path, "fake-key-1")
        assert after == before
        assert after["status"] == "open"
        assert after["last_seen"] == "2026-07-14"  # NOT bumped to 2026-07-15
        # key never appears in any bucket since it wasn't part of the contract
        assert "fake-key-1" not in (
            plan["inserted"] + plan["refreshed"] + plan["reopened"]
            + plan["ignored_snoozed"] + plan["ignored_closed"]
        )

    def test_snoozed_key_past_due_reopens_and_refreshes(self, tmp_path):
        ts.upsert_from_contract([_item("fake-key-1", text="orig fake text")], "2026-07-10")
        close = ts.close_todo("fake-key-1", "snooze", "test", snooze_until="2026-07-12")
        assert close["ok"] is True

        plan = ts.upsert_from_contract(
            [_item("fake-key-1", text="post-snooze fake text", priority="high")], "2026-07-14"
        )

        assert plan["reopened"] == ["fake-key-1"]
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "open"
        assert row["snoozed_until"] is None
        assert row["text"] == "post-snooze fake text"
        assert row["priority"] == "high"

    def test_snoozed_key_future_left_completely_alone(self, tmp_path):
        ts.upsert_from_contract([_item("fake-key-1", text="orig fake text")], "2026-07-10")
        ts.close_todo("fake-key-1", "snooze", "test", snooze_until="2026-07-20")
        before = _row_dict(tmp_path, "fake-key-1")
        assert before["status"] == "snoozed"

        plan = ts.upsert_from_contract(
            [_item("fake-key-1", text="text changed while snoozed!", priority="high")], "2026-07-14"
        )

        assert plan["ignored_snoozed"] == ["fake-key-1"]
        after = _row_dict(tmp_path, "fake-key-1")
        assert after == before
        assert after["text"] == "orig fake text"
        assert after["status"] == "snoozed"
        assert after["snoozed_until"] == "2026-07-20"

    def test_done_key_reappearing_is_ignored_stays_closed(self, tmp_path):
        ts.upsert_from_contract([_item("fake-key-1", text="orig fake text")], "2026-07-10")
        ts.close_todo("fake-key-1", "done", "test")

        plan = ts.upsert_from_contract([_item("fake-key-1", text="changed fake text")], "2026-07-14")

        assert plan["ignored_closed"] == ["fake-key-1"]
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "done"
        assert row["text"] == "orig fake text"

    def test_dismissed_key_reappearing_is_ignored_stays_closed(self, tmp_path):
        ts.upsert_from_contract([_item("fake-key-1", text="orig fake text")], "2026-07-10")
        ts.close_todo("fake-key-1", "dismiss", "test")

        plan = ts.upsert_from_contract([_item("fake-key-1", text="changed fake text")], "2026-07-14")

        assert plan["ignored_closed"] == ["fake-key-1"]
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "dismissed"
        assert row["text"] == "orig fake text"


# ---------------------------------------------------------------------------
# plan_upsert_from_contract — read-only dry run
# ---------------------------------------------------------------------------


class TestPlanUpsertFromContract:
    def test_plan_makes_zero_writes_and_matches_the_real_call(self, tmp_path):
        # Seed a mix of states so the plan has something interesting to say.
        ts.upsert_from_contract(
            [_item("open-key", text="fake open task"), _item("snooze-past", text="fake snoozed past task")],
            "2026-07-01",
        )
        ts.upsert_from_contract([_item("snooze-future", text="fake snoozed future task")], "2026-07-01")
        ts.close_todo("snooze-past", "snooze", "test", snooze_until="2026-07-05")
        ts.close_todo("snooze-future", "snooze", "test", snooze_until="2026-07-30")
        ts.upsert_from_contract([_item("done-key", text="fake done task")], "2026-07-01")
        ts.close_todo("done-key", "done", "test")

        contract = [
            _item("open-key", text="open task refreshed text"),
            _item("snooze-past", text="snooze-past refreshed text"),
            _item("snooze-future", text="snooze-future SHOULD NOT LAND"),
            _item("done-key", text="done SHOULD NOT LAND"),
            _item("brand-new-key", text="brand new fake task"),
        ]

        before_rows = {r["key"]: dict(r) for r in _all_rows(tmp_path)}

        plan = ts.plan_upsert_from_contract(contract, "2026-07-14")

        after_rows = {r["key"]: dict(r) for r in _all_rows(tmp_path)}
        assert after_rows == before_rows  # zero rows changed by the plan call
        assert "brand-new-key" not in after_rows  # nothing inserted either

        # Now do the real call and confirm the plan's buckets match reality.
        real_plan = ts.upsert_from_contract(contract, "2026-07-14")
        assert real_plan == plan

        assert real_plan["inserted"] == ["brand-new-key"]
        assert real_plan["refreshed"] == ["open-key"]
        assert real_plan["reopened"] == ["snooze-past"]
        assert real_plan["ignored_snoozed"] == ["snooze-future"]
        assert real_plan["ignored_closed"] == ["done-key"]


# ---------------------------------------------------------------------------
# reopen_expired_snoozes — standalone sweep
# ---------------------------------------------------------------------------


class TestReopenExpiredSnoozes:
    def test_reopens_only_past_due_leaves_future_alone(self, tmp_path):
        ts.upsert_from_contract(
            [_item("past-key", text="fake past task"), _item("future-key", text="fake future task")],
            "2026-07-01",
        )
        ts.close_todo("past-key", "snooze", "test", snooze_until="2026-07-10")
        ts.close_todo("future-key", "snooze", "test", snooze_until="2026-07-20")

        reopened = ts.reopen_expired_snoozes("2026-07-15")

        assert reopened == ["past-key"]
        past_row = _row_dict(tmp_path, "past-key")
        assert past_row["status"] == "open"
        assert past_row["snoozed_until"] is None
        future_row = _row_dict(tmp_path, "future-key")
        assert future_row["status"] == "snoozed"
        assert future_row["snoozed_until"] == "2026-07-20"

    def test_boundary_snoozed_until_equal_today_reopens(self, tmp_path):
        ts.upsert_from_contract([_item("edge-key", text="fake edge task")], "2026-07-01")
        ts.close_todo("edge-key", "snooze", "test", snooze_until="2026-07-15")

        reopened = ts.reopen_expired_snoozes("2026-07-15")

        assert reopened == ["edge-key"]

    def test_noop_on_empty_store(self, tmp_path):
        reopened = ts.reopen_expired_snoozes("2026-07-15")
        assert reopened == []


# ---------------------------------------------------------------------------
# close_todo — done/dismiss/snooze lifecycle
# ---------------------------------------------------------------------------


class TestCloseTodo:
    def _seed(self, tmp_path, key="fake-key-1", text="fake task text"):
        ts.upsert_from_contract([_item(key, text=text)], "2026-07-14")

    def test_done_sets_status_and_closed_at(self, tmp_path):
        self._seed(tmp_path)
        result = ts.close_todo("fake-key-1", "done", "discord")
        assert result == {"ok": True, "status": "done"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "done"
        assert row["closed_at"] is not None
        assert row["snoozed_until"] is None

    def test_dismiss_sets_status_and_closed_at(self, tmp_path):
        self._seed(tmp_path)
        result = ts.close_todo("fake-key-1", "dismiss", "discord")
        assert result == {"ok": True, "status": "dismissed"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "dismissed"
        assert row["closed_at"] is not None

    def test_snooze_sets_status_and_snoozed_until(self, tmp_path):
        self._seed(tmp_path)
        result = ts.close_todo("fake-key-1", "snooze", "discord", snooze_until="2026-07-20")
        assert result == {"ok": True, "status": "snoozed"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "snoozed"
        assert row["snoozed_until"] == "2026-07-20"
        assert row["closed_at"] is None

    def test_reclosing_same_key_same_action_is_idempotent_success(self, tmp_path):
        self._seed(tmp_path)
        first = ts.close_todo("fake-key-1", "done", "discord")
        second = ts.close_todo("fake-key-1", "done", "discord")
        assert first == {"ok": True, "status": "done"}
        assert second == {"ok": True, "status": "done"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "done"

    def test_reclosing_with_different_action_overrides_state(self, tmp_path):
        self._seed(tmp_path)
        ts.close_todo("fake-key-1", "done", "discord")
        result = ts.close_todo("fake-key-1", "dismiss", "discord")
        assert result == {"ok": True, "status": "dismissed"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "dismissed"

    def test_snooze_after_done_clears_closed_at(self, tmp_path):
        self._seed(tmp_path)
        ts.close_todo("fake-key-1", "done", "discord")
        ts.close_todo("fake-key-1", "snooze", "discord", snooze_until="2026-08-01")
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "snoozed"
        assert row["closed_at"] is None
        assert row["snoozed_until"] == "2026-08-01"

    def test_unknown_key_returns_ok_false_without_raising(self, tmp_path):
        result = ts.close_todo("does-not-exist", "done", "discord")
        assert result == {"ok": False, "error": "unknown key: does-not-exist"}

    def test_invalid_action_returns_ok_false(self, tmp_path):
        self._seed(tmp_path)
        result = ts.close_todo("fake-key-1", "bogus-action", "discord")
        assert result["ok"] is False
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "open"  # untouched

    def test_snooze_without_snooze_until_does_not_silently_succeed(self, tmp_path):
        """Documents the actual (correct, defensive) behavior: snooze with no
        snooze_until is rejected up front rather than silently succeeding
        with a null/empty date. Not a bug — reported for completeness per
        the acceptance-criteria checklist."""
        self._seed(tmp_path)
        result = ts.close_todo("fake-key-1", "snooze", "discord", snooze_until=None)
        assert result == {"ok": False, "error": "snooze requires snooze_until"}
        row = _row_dict(tmp_path, "fake-key-1")
        assert row["status"] == "open"  # untouched, not silently snoozed


# ---------------------------------------------------------------------------
# get_open_keys / get_closed_keys
# ---------------------------------------------------------------------------


class TestGetOpenKeys:
    def test_excludes_snoozed_done_dismissed(self, tmp_path):
        ts.upsert_from_contract(
            [
                _item("open-key", text="fake open"),
                _item("snooze-key", text="fake snoozed"),
                _item("done-key", text="fake done"),
                _item("dismiss-key", text="fake dismissed"),
            ],
            "2026-07-14",
        )
        ts.close_todo("snooze-key", "snooze", "test", snooze_until="2026-08-01")
        ts.close_todo("done-key", "done", "test")
        ts.close_todo("dismiss-key", "dismiss", "test")

        open_keys = {d["key"] for d in ts.get_open_keys()}
        assert open_keys == {"open-key"}

    def test_ordering_priority_then_last_seen_desc(self, tmp_path):
        ts.upsert_from_contract(
            [
                _item("low-old", text="fake low old", priority="low"),
                _item("high-old", text="fake high old", priority="high"),
                _item("medium-key", text="fake medium", priority="medium"),
            ],
            "2026-07-01",
        )
        # Bump high-new's last_seen later than high-old, both priority high.
        ts.upsert_from_contract([_item("high-new", text="fake high new", priority="high")], "2026-07-10")

        keys = [d["key"] for d in ts.get_open_keys()]

        # All 'high' priority keys must precede 'medium' which must precede 'low'.
        assert keys.index("high-old") < keys.index("medium-key")
        assert keys.index("high-new") < keys.index("medium-key")
        assert keys.index("medium-key") < keys.index("low-old")
        # Within the same priority, more-recently-seen sorts first.
        assert keys.index("high-new") < keys.index("high-old")

    def test_unknown_priority_sorts_last(self, tmp_path):
        ts.upsert_from_contract(
            [
                _item("no-priority", text="fake no prio", priority=None),
                _item("low-key", text="fake low", priority="low"),
            ],
            "2026-07-14",
        )
        keys = [d["key"] for d in ts.get_open_keys()]
        assert keys.index("low-key") < keys.index("no-priority")


class TestGetClosedKeys:
    def test_only_done_and_dismissed(self, tmp_path):
        ts.upsert_from_contract(
            [
                _item("open-key", text="fake open"),
                _item("snooze-key", text="fake snoozed"),
                _item("done-key", text="fake done"),
                _item("dismiss-key", text="fake dismissed"),
            ],
            "2026-07-14",
        )
        ts.close_todo("snooze-key", "snooze", "test", snooze_until="2026-08-01")
        ts.close_todo("done-key", "done", "test")
        ts.close_todo("dismiss-key", "dismiss", "test")

        closed = set(ts.get_closed_keys())
        assert closed == {"done-key", "dismiss-key"}


# ---------------------------------------------------------------------------
# record_audit / audit log
# ---------------------------------------------------------------------------


class TestRecordAudit:
    def test_close_todo_writes_one_jsonl_line_per_call_including_repeats(self, tmp_path):
        ts.upsert_from_contract([_item("fake-key-1", text="fake task")], "2026-07-14")

        ts.close_todo("fake-key-1", "done", "discord")
        ts.close_todo("fake-key-1", "done", "discord")  # idempotent repeat
        ts.close_todo("fake-key-1", "dismiss", "discord")  # override

        log_path = tmp_path / "todo-audit.log"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        entries = [json.loads(line) for line in lines]
        assert [e["action"] for e in entries] == ["done", "done", "dismiss"]
        for e in entries:
            assert e["key"] == "fake-key-1"
            assert e["source"] == "discord"
            assert isinstance(e["ts"], str) and e["ts"]

    def test_no_audit_entry_for_unknown_key(self, tmp_path):
        ts.close_todo("does-not-exist", "done", "discord")
        log_path = tmp_path / "todo-audit.log"
        assert not log_path.exists()


# ---------------------------------------------------------------------------
# Key stability across "days"
# ---------------------------------------------------------------------------


class TestKeyStabilityAcrossDays:
    def test_same_key_two_days_one_row_dates_grow_first_seen_fixed(self, tmp_path):
        ts.upsert_from_contract(
            [_item("stable-key", text="day1 fake text", source_dates=["2026-07-14"])], "2026-07-14"
        )
        day1 = _row_dict(tmp_path, "stable-key")
        assert day1["first_seen"] == "2026-07-14"
        assert day1["last_seen"] == "2026-07-14"

        # Day 2: journal re-proposes the same stable key with an accumulated
        # source_dates list (this is what the caller is expected to pass —
        # the store itself just persists whatever source_dates it's given).
        ts.upsert_from_contract(
            [_item("stable-key", text="day2 fake text", source_dates=["2026-07-14", "2026-07-15"])],
            "2026-07-15",
        )
        day2 = _row_dict(tmp_path, "stable-key")

        rows = _all_rows(tmp_path)
        assert len(rows) == 1  # still one row, not a duplicate

        assert day2["text"] == "day2 fake text"
        assert json.loads(day2["source_dates"]) == ["2026-07-14", "2026-07-15"]
        assert day2["first_seen"] == "2026-07-14"  # unchanged
        assert day2["last_seen"] == "2026-07-15"  # advanced


# ---------------------------------------------------------------------------
# get_stale_todos
# ---------------------------------------------------------------------------

_MOCK_TODAY = "2026-07-17"
_EXPECTED_KEYS = {"key", "text", "priority", "recurring", "source_dates", "first_seen", "last_seen"}


@pytest.fixture()
def mock_today(monkeypatch):
    monkeypatch.setattr("plugins.life_ops.todo_store._today_str", lambda: _MOCK_TODAY)


class TestGetStaleTodos:
    """AC: get_stale_todos(threshold_days=5) — issue #64."""

    def _seed(self, key, last_seen, *, text="fake task", priority="medium", recurring=False):
        ts.upsert_from_contract(
            [_item(key, text=text, priority=priority, recurring=recurring)], last_seen
        )

    # AC: stale todo (last_seen strictly older than threshold) is returned
    def test_stale_todo_is_returned(self, tmp_path, mock_today):
        self._seed("stale-key", "2026-07-07")  # 10 days before mock today
        result = ts.get_stale_todos(threshold_days=5)
        assert len(result) == 1
        assert result[0]["key"] == "stale-key"

    # AC: todos within the threshold are excluded
    def test_recent_todo_is_excluded(self, tmp_path, mock_today):
        self._seed("fresh-key", "2026-07-14")  # 3 days before mock today
        result = ts.get_stale_todos(threshold_days=5)
        assert result == []

    # AC: boundary — last_seen exactly at cutoff is NOT stale (strictly older required)
    def test_boundary_exactly_at_threshold_is_excluded(self, tmp_path, mock_today):
        self._seed("boundary-key", "2026-07-12")  # exactly 5 days before mock today
        result = ts.get_stale_todos(threshold_days=5)
        assert result == []

    # AC: results are sorted oldest-first by last_seen
    def test_sort_order_oldest_first(self, tmp_path, mock_today):
        self._seed("seven-days", "2026-07-10", text="7 days ago")
        self._seed("fifteen-days", "2026-07-02", text="15 days ago")
        self._seed("ten-days", "2026-07-07", text="10 days ago")
        result = ts.get_stale_todos(threshold_days=5)
        assert [d["key"] for d in result] == ["fifteen-days", "ten-days", "seven-days"]

    # AC: returns [] when store is empty
    def test_empty_store(self, tmp_path, mock_today):
        result = ts.get_stale_todos()
        assert result == []

    # AC: returns [] when all todos are fresh
    def test_all_fresh_returns_empty(self, tmp_path, mock_today):
        self._seed("fresh-1", "2026-07-16")
        self._seed("fresh-2", "2026-07-15")
        result = ts.get_stale_todos(threshold_days=5)
        assert result == []

    # AC: recurring todos are NOT exempt; included if last_seen exceeds threshold
    def test_recurring_todo_is_eligible(self, tmp_path, mock_today):
        self._seed("recur-key", "2026-07-09", recurring=True)  # 8 days ago
        result = ts.get_stale_todos(threshold_days=5)
        assert len(result) == 1
        assert result[0]["key"] == "recur-key"
        assert result[0]["recurring"] is True

    # AC: every returned dict contains exactly the required keys
    def test_row_shape_matches_get_open_todos(self, tmp_path, mock_today):
        self._seed("shape-key", "2026-07-07")
        result = ts.get_stale_todos(threshold_days=5)
        assert len(result) == 1
        assert set(result[0].keys()) == _EXPECTED_KEYS

    # AC: default parameter behaves as threshold_days=5
    def test_default_parameter_is_five_days(self, tmp_path, mock_today):
        self._seed("six-days-ago", "2026-07-11")  # 6 days before mock today
        result_default = ts.get_stale_todos()
        result_explicit = ts.get_stale_todos(threshold_days=5)
        assert result_default == result_explicit
        assert len(result_default) == 1

    # AC: only open todos are included — closed/snoozed are excluded even if stale
    def test_closed_stale_todos_are_excluded(self, tmp_path, mock_today):
        self._seed("old-done", "2026-07-01")
        self._seed("old-dismissed", "2026-07-01")
        self._seed("old-snoozed", "2026-07-01")
        ts.close_todo("old-done", "done", "test")
        ts.close_todo("old-dismissed", "dismiss", "test")
        ts.close_todo("old-snoozed", "snooze", "test", snooze_until="2026-08-01")
        result = ts.get_stale_todos(threshold_days=5)
        assert result == []

    # AC: get_open_todos() and all other existing function signatures are unmodified
    def test_get_open_todos_unmodified(self, tmp_path, mock_today):
        self._seed("open-key", "2026-07-16")
        self._seed("stale-open", "2026-07-07")
        open_todos = ts.get_open_todos()
        assert len(open_todos) == 2
        for d in open_todos:
            assert set(d.keys()) == _EXPECTED_KEYS
