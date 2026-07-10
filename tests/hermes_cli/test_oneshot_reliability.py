from __future__ import annotations

import types

from hermes_cli import oneshot


def test_finalize_oneshot_session_marks_complete(monkeypatch):
    calls = []

    class FakeDB:
        def end_session(self, session_id, end_reason):
            calls.append((session_id, end_reason))

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_state",
        types.SimpleNamespace(get_session_db=lambda: FakeDB()),
    )

    oneshot._finalize_oneshot_session({"session_id": "s1"})

    assert calls == [("s1", "oneshot_complete")]


def test_finalize_oneshot_session_can_mark_failed(monkeypatch):
    calls = []

    class FakeDB:
        def end_session(self, session_id, end_reason):
            calls.append((session_id, end_reason))

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_state",
        types.SimpleNamespace(get_session_db=lambda: FakeDB()),
    )

    oneshot._finalize_oneshot_session({"session_id": "s1"}, "oneshot_failed")

    assert calls == [("s1", "oneshot_failed")]


def test_finalize_oneshot_session_ignores_missing_session_id(monkeypatch):
    calls = []

    class FakeDB:
        def end_session(self, session_id, end_reason):
            calls.append((session_id, end_reason))

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_state",
        types.SimpleNamespace(get_session_db=lambda: FakeDB()),
    )

    oneshot._finalize_oneshot_session({})

    assert calls == []


def test_finalize_oneshot_session_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setitem(
        __import__("sys").modules,
        "hermes_state",
        types.SimpleNamespace(get_session_db=boom),
    )

    oneshot._finalize_oneshot_session({"session_id": "s1"})
