"""Regression tests for gateway assistant transcript persistence."""

from __future__ import annotations

from typing import Any

from gateway.run import GatewayRunner


class _FakeSessionDB:
    def __init__(self, messages):
        self.messages = list(messages)
        self.appended = []

    def get_messages(self, session_id):
        assert session_id == "session-1"
        return list(self.messages)

    def append_message(self, *, session_id, role, content):
        assert session_id == "session-1"
        row = {"id": len(self.messages) + 1, "role": role, "content": content}
        self.messages.append(row)
        self.appended.append(row)


def _runner_with_db(messages):
    runner = object.__new__(GatewayRunner)
    db = _FakeSessionDB(messages)
    runner_any = runner  # type: Any
    runner_any._session_db = db
    return runner, db


def test_backfills_visible_assistant_response_when_agent_db_flush_missed_it():
    """A visible gateway response must be present in SQLite replay history.

    This covers the regression where the agent persisted the inbound user row
    early, gateway skipped DB fallback writes because a SessionDB existed, and
    the final assistant response was sent to the platform but missing from
    state.db.  Next-turn replay then looked like a user-only backlog.
    """
    runner, db = _runner_with_db(
        [
            {"id": 1, "role": "user", "content": "previous"},
            {"id": 2, "role": "assistant", "content": "previous reply"},
            {"id": 3, "role": "user", "content": "latest user"},
        ]
    )

    runner._ensure_visible_response_persisted(
        session_id="session-1",
        response="latest assistant reply",
        tail_id_before_turn=2,
    )

    assert db.appended == [
        {"id": 4, "role": "assistant", "content": "latest assistant reply"}
    ]


def test_does_not_duplicate_when_agent_already_persisted_assistant_response():
    runner, db = _runner_with_db(
        [
            {"id": 1, "role": "user", "content": "previous"},
            {"id": 2, "role": "assistant", "content": "previous reply"},
            {"id": 3, "role": "user", "content": "latest user"},
            {"id": 4, "role": "assistant", "content": "latest assistant reply"},
        ]
    )

    runner._ensure_visible_response_persisted(
        session_id="session-1",
        response="latest assistant reply",
        tail_id_before_turn=2,
    )

    assert db.appended == []


def test_no_response_means_no_backfill():
    runner, db = _runner_with_db(
        [
            {"id": 1, "role": "user", "content": "previous"},
            {"id": 2, "role": "user", "content": "latest user"},
        ]
    )

    runner._ensure_visible_response_persisted(
        session_id="session-1",
        response=None,
        tail_id_before_turn=1,
    )

    assert db.appended == []
