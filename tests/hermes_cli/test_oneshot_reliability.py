from __future__ import annotations

import types

import pytest

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


@pytest.mark.parametrize(
    "failure_signal",
    [
        {"failed": True},
        {"partial": True},
        {"error": "HTTP 502: upstream unavailable"},
    ],
)
def test_run_oneshot_marks_structured_provider_failure_failed(
    monkeypatch, capsys, failure_signal
):
    result = {
        "session_id": "provider-failure-session",
        **failure_signal,
    }
    finalized = []
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (
            "API call failed after 1 retries: HTTP 502: upstream unavailable",
            result,
        ),
    )
    monkeypatch.setattr(
        oneshot,
        "_finalize_oneshot_session",
        lambda run_result, end_reason="oneshot_complete": finalized.append(
            (run_result, end_reason)
        ),
    )

    assert oneshot.run_oneshot("hello") == 2
    captured = capsys.readouterr()
    assert captured.out == (
        "API call failed after 1 retries: HTTP 502: upstream unavailable\n"
    )
    assert finalized == [(result, "oneshot_failed")]


def test_run_oneshot_genuine_text_succeeds(monkeypatch, capsys):
    result = {
        "session_id": "successful-session",
        "completed": True,
        "failed": False,
        "partial": False,
    }
    finalized = []
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: ("genuine answer", result),
    )
    monkeypatch.setattr(
        oneshot,
        "_finalize_oneshot_session",
        lambda run_result, end_reason="oneshot_complete": finalized.append(
            (run_result, end_reason)
        ),
    )

    assert oneshot.run_oneshot("hello") == 0
    captured = capsys.readouterr()
    assert captured.out == "genuine answer\n"
    assert finalized == [(result, "oneshot_complete")]
