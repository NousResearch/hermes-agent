"""Regressions for paginating durable ACP sessions before workspace filtering."""

import json
from unittest.mock import MagicMock

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class _PagedSessionDB:
    def __init__(self, rows):
        self.rows = rows
        self.offsets = []

    def list_sessions_rich(self, *, source, limit, offset=0):
        assert source == "acp"
        self.offsets.append(offset)
        return self.rows[offset : offset + limit]


def _row(index, cwd):
    return {
        "id": f"session-{index:04d}",
        "model_config": json.dumps({"cwd": cwd}),
        "model": "test-model",
        "message_count": 1,
        "title": None,
        "preview": f"message {index}",
        "last_active": 2_000_000_000 - index,
        "started_at": 2_000_000_000 - index,
    }


def _manager(rows):
    db = _PagedSessionDB(rows)
    return SessionManager(agent_factory=lambda: MagicMock(), db=db), db


def test_restart_listing_reaches_oldest_workspace_session():
    rows = [_row(index, "/workspace-a") for index in range(1000)]
    rows.append(_row(1000, "/workspace-b"))

    # A fresh manager models the adapter after restart: only persisted rows exist.
    manager, db = _manager(rows)
    listed = manager.list_sessions(cwd="/workspace-b")

    assert [session["session_id"] for session in listed] == ["session-1000"]
    assert db.offsets == [0, 1000]


@pytest.mark.asyncio
async def test_following_protocol_cursors_returns_every_old_workspace_session():
    rows = [_row(index, "/workspace-a") for index in range(1000)]
    rows.extend(_row(index, "/workspace-b") for index in range(1000, 1125))
    manager, _db = _manager(rows)
    agent = HermesACPAgent(session_manager=manager)

    found = []
    cursor = None
    while True:
        page = await agent.list_sessions(cursor=cursor, cwd="/workspace-b")
        found.extend(session.session_id for session in page.sessions)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert found == [f"session-{index:04d}" for index in range(1000, 1125)]


def test_workspace_sessions_in_first_db_page_are_still_listed():
    rows = [_row(index, "/workspace-b") for index in range(3)]
    rows.extend(_row(index, "/workspace-a") for index in range(3, 999))
    manager, db = _manager(rows)

    listed = manager.list_sessions(cwd="/workspace-b")

    assert [session["session_id"] for session in listed] == [
        "session-0000",
        "session-0001",
        "session-0002",
    ]
    assert db.offsets == [0]
