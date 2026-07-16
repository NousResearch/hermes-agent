"""Tests for plugins.life_ops.discord_commands's /done, /dismiss, /snooze, /away-on,
/away-off handlers and the shared todo-key autocomplete.

These are the pure-logic ``handle_x()`` functions (same shape as /rpe's
``handle_rpe``) that the thin Discord-facing ``register_x_command()``
wrappers call into — see plugins/life_ops/discord_commands.py's module docstring.
Exercised here against a real (tmp HERMES_HOME) plugins.life_ops.todo_store /
plugins.life_ops.away_mode, following the explicit
``monkeypatch.setenv("HERMES_HOME", ...)`` pattern used in
tests/hermes/test_todo_store.py and tests/hermes/test_away_mode.py.
"""
from __future__ import annotations

import json

import pytest

from plugins.life_ops import discord_commands as hdiscord
from plugins.life_ops import away_mode
from plugins.life_ops import todo_store as ts

try:
    import discord as _discord_check  # noqa: F401
    HAS_DISCORD = True
except ImportError:
    HAS_DISCORD = False


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _seed(key="fake-key-1", text="fake task text", priority="medium", for_date="2026-07-14"):
    ts.upsert_from_contract(
        [{"key": key, "text": text, "priority": priority, "source_dates": [for_date]}],
        for_date,
    )


# ---------------------------------------------------------------------------
# handle_done
# ---------------------------------------------------------------------------


class TestHandleDone:
    def test_closes_todo_as_done(self, tmp_path):
        _seed()
        result = hdiscord.handle_done("fake-key-1")
        assert result == {"ok": True, "status": "done"}

        conn = ts.connect(tmp_path / "todos.db")
        try:
            row = conn.execute("SELECT status FROM todos WHERE key=?", ("fake-key-1",)).fetchone()
        finally:
            conn.close()
        assert row["status"] == "done"

    def test_uses_discord_done_source_string_in_audit_log(self, tmp_path):
        _seed()
        hdiscord.handle_done("fake-key-1")

        log_path = tmp_path / "todo-audit.log"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
        assert entries[-1]["source"] == "discord:/done"
        assert entries[-1]["action"] == "done"

    def test_unknown_key_returns_friendly_error_not_exception(self):
        # todo_store.close_todo()'s unknown-key error dict has no "status"
        # key at all (see plugins/life_ops/todo_store.py close_todo) —
        # handle_done() forwards it verbatim rather than raising.
        result = hdiscord.handle_done("does-not-exist")
        assert result == {"ok": False, "error": "unknown key: does-not-exist"}

    def test_empty_key_returns_friendly_error(self):
        result = hdiscord.handle_done("")
        assert result["ok"] is False
        assert "required" in result["error"].lower()

    def test_whitespace_only_key_returns_friendly_error(self):
        result = hdiscord.handle_done("   ")
        assert result["ok"] is False
        assert "required" in result["error"].lower()


# ---------------------------------------------------------------------------
# handle_dismiss
# ---------------------------------------------------------------------------


class TestHandleDismiss:
    def test_closes_todo_as_dismissed(self):
        _seed()
        result = hdiscord.handle_dismiss("fake-key-1")
        assert result == {"ok": True, "status": "dismissed"}

    def test_uses_discord_dismiss_source_string_in_audit_log(self, tmp_path):
        _seed()
        hdiscord.handle_dismiss("fake-key-1")

        log_path = tmp_path / "todo-audit.log"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
        assert entries[-1]["source"] == "discord:/dismiss"
        assert entries[-1]["action"] == "dismiss"

    def test_unknown_key_returns_friendly_error_not_exception(self):
        result = hdiscord.handle_dismiss("does-not-exist")
        assert result["ok"] is False
        assert "does-not-exist" in result["error"]


# ---------------------------------------------------------------------------
# handle_snooze
# ---------------------------------------------------------------------------


class TestHandleSnooze:
    def test_valid_date_computes_correct_snoozed_until(self, tmp_path):
        _seed()
        result = hdiscord.handle_snooze("fake-key-1", "2026-07-22")
        assert result == {"ok": True, "status": "snoozed"}

        conn = ts.connect(tmp_path / "todos.db")
        try:
            row = conn.execute(
                "SELECT status, snoozed_until FROM todos WHERE key=?", ("fake-key-1",)
            ).fetchone()
        finally:
            conn.close()
        assert row["status"] == "snoozed"
        assert row["snoozed_until"] == "2026-07-22"

    def test_uses_discord_snooze_source_string_in_audit_log(self, tmp_path):
        _seed()
        hdiscord.handle_snooze("fake-key-1", "2026-07-22")

        log_path = tmp_path / "todo-audit.log"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").strip().split("\n")]
        assert entries[-1]["source"] == "discord:/snooze"
        assert entries[-1]["action"] == "snooze"

    def test_unparseable_date_returns_friendly_error_not_crash(self):
        _seed()
        result = hdiscord.handle_snooze("fake-key-1", "not-a-date")
        assert result["ok"] is False
        assert "not-a-date" in result["error"]
        assert "YYYY-MM-DD" in result["error"]

    def test_unparseable_date_does_not_touch_the_row(self, tmp_path):
        _seed()
        hdiscord.handle_snooze("fake-key-1", "not-a-date")

        conn = ts.connect(tmp_path / "todos.db")
        try:
            row = conn.execute("SELECT status FROM todos WHERE key=?", ("fake-key-1",)).fetchone()
        finally:
            conn.close()
        assert row["status"] == "open"

    def test_unknown_key_returns_friendly_error(self):
        result = hdiscord.handle_snooze("does-not-exist", "2026-07-22")
        assert result["ok"] is False
        assert "does-not-exist" in result["error"]

    def test_empty_key_returns_friendly_error_before_date_parsing(self):
        result = hdiscord.handle_snooze("", "not-a-date-either")
        assert result["ok"] is False
        assert "required" in result["error"].lower()


# ---------------------------------------------------------------------------
# handle_away_on / handle_away_off
# ---------------------------------------------------------------------------


class TestHandleAwayOnOff:
    def test_away_on_with_until_persists(self):
        result = hdiscord.handle_away_on("2026-07-25")
        assert result == {"ok": True, "until": "2026-07-25"}
        assert away_mode.is_away("2026-07-20") is True
        assert away_mode.away_status("2026-07-20") == {"active": True, "until": "2026-07-25"}

    def test_away_on_indefinite_persists(self):
        result = hdiscord.handle_away_on(None)
        assert result == {"ok": True, "until": None}
        assert away_mode.is_away("2099-01-01") is True

    def test_away_on_empty_string_treated_as_indefinite(self):
        result = hdiscord.handle_away_on("")
        assert result == {"ok": True, "until": None}
        assert away_mode.is_away("2026-07-20") is True

    def test_away_on_unparseable_date_returns_friendly_error(self):
        result = hdiscord.handle_away_on("not-a-date")
        assert result["ok"] is False
        assert "not-a-date" in result["error"]
        # And does not turn away mode on as a side effect.
        assert away_mode.is_away("2026-07-20") is False

    def test_away_off_clears_and_persists(self):
        away_mode.set_away(until="2099-01-01")
        assert away_mode.is_away("2026-07-20") is True

        result = hdiscord.handle_away_off()

        assert result == {"ok": True}
        assert away_mode.is_away("2026-07-20") is False
        assert away_mode.away_status("2026-07-20") == {"active": False, "until": None}

    def test_away_off_on_never_set_store_is_safe(self):
        result = hdiscord.handle_away_off()
        assert result == {"ok": True}
        assert away_mode.is_away("2026-07-20") is False

    def test_away_on_then_off_round_trip_reflected_by_is_away(self):
        hdiscord.handle_away_on("2026-08-01")
        assert away_mode.is_away("2026-07-20") is True
        hdiscord.handle_away_off()
        assert away_mode.is_away("2026-07-20") is False


# ---------------------------------------------------------------------------
# _autocomplete_todo_key
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DISCORD, reason="discord.py not installed")
class TestAutocompleteTodoKey:
    @pytest.mark.asyncio
    async def test_sources_from_open_keys(self):
        _seed(key="alpha-key", text="Alpha task text")
        _seed(key="beta-key", text="Beta task text", for_date="2026-07-14")

        choices = await hdiscord._autocomplete_todo_key(None, "")

        values = {c.value for c in choices}
        assert values == {"alpha-key", "beta-key"}

    @pytest.mark.asyncio
    async def test_excludes_closed_and_snoozed_keys(self):
        _seed(key="open-key", text="Open task")
        _seed(key="done-key", text="Done task")
        ts.close_todo("done-key", "done", "test")

        choices = await hdiscord._autocomplete_todo_key(None, "")

        values = {c.value for c in choices}
        assert values == {"open-key"}

    @pytest.mark.asyncio
    async def test_filters_by_current_query_case_insensitively(self):
        _seed(key="fix-auth-bug", text="Fix the authentication bug")
        _seed(key="buy-milk", text="Buy milk at the store")

        choices = await hdiscord._autocomplete_todo_key(None, "AUTH")

        values = {c.value for c in choices}
        assert values == {"fix-auth-bug"}

    @pytest.mark.asyncio
    async def test_empty_store_does_not_crash_returns_empty_list(self):
        choices = await hdiscord._autocomplete_todo_key(None, "")
        assert choices == []

    @pytest.mark.asyncio
    async def test_caps_at_25_choices(self):
        for i in range(30):
            _seed(key=f"key-{i:02d}", text=f"task {i}", for_date="2026-07-14")

        choices = await hdiscord._autocomplete_todo_key(None, "")

        assert len(choices) == 25
