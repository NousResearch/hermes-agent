"""Tests for `hermes sessions list --duration`.

Behavior-contract style: assert the Duration label formatting and that the
column appears only when --duration is passed. Uses a temp HERMES_HOME with a
seeded SessionDB so no real data is touched.
"""
import os
import time

import pytest


def _make_session(title, started_at, ended_at, source="cli", sid="s1"):
    return {
        "id": sid,
        "source": source,
        "model": "x",
        "title": title,
        "started_at": started_at,
        "ended_at": ended_at,
        "message_count": 1,
        "preview": "hello world this is a preview",
        "last_active": ended_at if ended_at else started_at,
    }


def test_duration_label_formatting():
    from hermes_cli import sessions_cmd

    # ended session: 1h 2m 3s
    s = _make_session("t", 1_000_000, 1_000_000 + 3723)
    assert sessions_cmd._duration_label(s) == "1h 02m"

    # ended session: 5m 09s
    s = _make_session("t", 1_000_000, 1_000_000 + 309)
    assert sessions_cmd._duration_label(s) == "5m 09s"

    # short session: seconds only
    s = _make_session("t", 1_000_000, 1_000_000 + 42)
    assert sessions_cmd._duration_label(s) == "42s"

    # multi-day
    s = _make_session("t", 1_000_000, 1_000_000 + 60 * 60 * 25 + 3 * 3600 + 7 * 60)
    assert sessions_cmd._duration_label(s) == "1d 04h"

    # active session (ended_at None): uses now, so duration >= started->now
    s = _make_session("t", time.time() - 120, None)
    assert sessions_cmd._duration_label(s) == "2m 00s"

    # missing started_at: defensive dash
    s = _make_session("t", None, None)
    assert sessions_cmd._duration_label(s) == "—"


def test_list_duration_column_appears(monkeypatch, capsys):
    from hermes_cli import sessions_cmd

    now = time.time()
    rows = [
        _make_session("Alpha", now - 3600, now - 60, sid="a"),
        _make_session("Beta", now - 120, None, sid="b"),
    ]

    class _FakeDB:
        def list_sessions_rich(self, **kwargs):
            return rows

    import hermes_state
    monkeypatch.setattr(hermes_state, "SessionDB", lambda: _FakeDB())

    class _Args:
        sessions_action = "list"
        source = None
        limit = 20
        workspace = None
        duration = True

    sessions_cmd.cmd_sessions(_Args())
    out = capsys.readouterr().out
    assert "Duration" in out
    assert "1h" in out  # Alpha
    assert "2m" in out  # Beta (active)


def test_list_duration_column_absent_by_default(monkeypatch, capsys):
    from hermes_cli import sessions_cmd

    now = time.time()
    rows = [_make_session("Alpha", now - 3600, now - 60, sid="a")]

    class _FakeDB:
        def list_sessions_rich(self, **kwargs):
            return rows

    monkeypatch.setattr(sessions_cmd, "SessionDB", lambda: _FakeDB())

    class _Args:
        sessions_action = "list"
        source = None
        limit = 20
        workspace = None
        duration = False

    sessions_cmd.cmd_sessions(_Args())
    out = capsys.readouterr().out
    assert "Duration" not in out
