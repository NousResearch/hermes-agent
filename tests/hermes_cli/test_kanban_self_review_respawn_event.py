"""A review respawn on the implementer's own assignee/override emits a
``self_review_respawn`` event (#138109).

``kanban_request_review`` without a ``reviewer`` leaves the card on the
implementer's assignee and its ``model_override``/``provider_override``, so the
review phase runs on the same model that did the work with no structural
marker. The dispatcher must make that visible; a distinct reviewer must not
trigger the event.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _spawned(task, workspace, board=None):
    return 4242


def _event_payloads(conn, tid, kind):
    return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_respawn_without_reviewer_records_self_review(kanban_home, all_assignees_spawnable):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="t", assignee="worker",
            model_override="claude-sonnet-5", provider_override="anthropic",
        )
        assert kb.request_review(conn, tid, summary="implemented") is True
        res = kbd.dispatch_once(conn, spawn_fn=_spawned)
        payloads = _event_payloads(conn, tid, "self_review_respawn")

    assert [t for t, _a, _w in res.spawned] == [tid]
    assert payloads == [{
        "assignee": "worker",
        "model_override": "claude-sonnet-5",
        "provider_override": "anthropic",
    }]


def test_respawn_with_distinct_reviewer_records_nothing(kanban_home, all_assignees_spawnable):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="t", assignee="worker",
            model_override="claude-sonnet-5", provider_override="anthropic",
        )
        assert kb.request_review(conn, tid, summary="implemented", reviewer="auditor") is True
        res = kbd.dispatch_once(conn, spawn_fn=_spawned)
        payloads = _event_payloads(conn, tid, "self_review_respawn")

    assert [t for t, _a, _w in res.spawned] == [tid]
    assert payloads == []


def test_respawn_with_reviewer_equal_to_implementer_records_self_review(
    kanban_home, all_assignees_spawnable,
):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="worker")
        assert kb.request_review(conn, tid, summary="implemented", reviewer="worker") is True
        res = kbd.dispatch_once(conn, spawn_fn=_spawned)
        payloads = _event_payloads(conn, tid, "self_review_respawn")

    assert [t for t, _a, _w in res.spawned] == [tid]
    assert payloads and payloads[0]["model_override"] is None
