"""``kanban.cancel`` / ``kanban.cancel_all`` (B3): the gateway verbs over the
kernel cancellation.

Contract: ``session:control`` + the owning profile, closed param sets, one card
(+ descendants) or every non-terminal card one origin session created. A live
worker is signalled before the card is marked; cancellation never touches the
breaker and never rewrites persisted work.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from tui_gateway.contracts.registry import CANONICAL_METHODS


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with SessionDB(db_path=home / "state.db") as db:
        db.create_session("s", source="test")
        epoch = begin_runtime_epoch(db, instance_id="current")
        authority = SessionAuthority(
            SimpleNamespace(_draining=False),
            profile_id="owned",
            instance_id="current",
            db=db,
            epoch=epoch,
        )
        authority.sessions["s"] = LiveSession(None, "route")
        connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
        try:
            yield connection, home
        finally:
            await connection.close()


def _seed():
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", session_id="sess-1")
        child = kb.create_task(conn, title="child", parents=[parent], session_id="sess-1")
        other = kb.create_task(conn, title="other", session_id="sess-2")
    return parent, child, other


async def _dispatch(connection, method, params, rid=1):
    return await connection.dispatch({"id": rid, "method": method, "params": params})


@pytest.mark.asyncio
async def test_kanban_cancel_marks_card_and_descendants(owner):
    connection, home = owner
    parent, child, other = _seed()
    response = await _dispatch(connection, "kanban.cancel", {
        "task_id": parent, "reason": "stop", "cancelled_by": "tester",
    })
    assert "error" not in response, response
    CANONICAL_METHODS["kanban.cancel"].result.model_validate(response["result"])
    result = response["result"]
    assert result["cancelled"] is True
    assert result["status"] == "cancelled"
    assert set(result["cancelled_ids"]) == {parent, child}
    assert result["workers_terminated"] == [] and result["workers_survived"] == []
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, parent).cancelled_by == "tester"
        assert kb.get_task(conn, child).status == "cancelled"
        assert kb.get_task(conn, other).status == "ready"


@pytest.mark.asyncio
async def test_kanban_cancel_terminal_is_idempotent(owner):
    connection, home = owner
    parent, _child, _other = _seed()
    first = await _dispatch(connection, "kanban.cancel", {"task_id": parent})
    second = await _dispatch(connection, "kanban.cancel", {"task_id": parent}, rid=2)
    assert first["result"]["cancelled"] is True
    assert "error" not in second, second
    assert second["result"]["cancelled"] is False
    assert second["result"]["status"] == "cancelled"
    assert second["result"]["cancelled_ids"] == []


@pytest.mark.asyncio
async def test_kanban_cancel_all_stops_one_session(owner):
    connection, home = owner
    parent, child, other = _seed()
    response = await _dispatch(connection, "kanban.cancel_all", {
        "origin_session_id": "sess-1", "cancelled_by": "tester",
    })
    assert "error" not in response, response
    CANONICAL_METHODS["kanban.cancel_all"].result.model_validate(response["result"])
    assert set(response["result"]["cancelled_ids"]) == {parent, child}
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, other).status == "ready"


@pytest.mark.asyncio
async def test_kanban_cancel_unknown_task_reports_status_none(owner):
    connection, _home = owner
    response = await _dispatch(connection, "kanban.cancel", {"task_id": "t_nope"})
    assert "error" not in response, response
    assert response["result"]["cancelled"] is False
    assert response["result"]["status"] is None


@pytest.mark.asyncio
async def test_kanban_cancel_requires_control_capability(owner):
    """A read-only actor may observe the board but never cancel work on it."""
    connection, _home = owner
    parent, _child, _other = _seed()
    read_only = AuthorityConnection(
        connection.authority, object(),
        {"user_id": "reader", "capabilities": ["session:read"]},
    )
    try:
        response = await _dispatch(read_only, "kanban.cancel", {"task_id": parent})
    finally:
        await read_only.close()
    assert response["error"]["message"] == "permission_denied", response
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, parent).status == "ready"


@pytest.mark.asyncio
async def test_kanban_cancel_foreign_profile_is_mismatch(owner):
    """Another profile's actor cannot cancel this profile's cards."""
    connection, home = owner
    parent, _child, _other = _seed()
    foreign = AuthorityConnection(
        connection.authority, object(),
        {"user_id": "owner", "profile_id": "someone-else"},
    )
    try:
        response = await _dispatch(foreign, "kanban.cancel", {"task_id": parent})
    finally:
        await foreign.close()
    assert response["error"]["message"] == "profile_mismatch", response
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, parent).status == "ready"


@pytest.mark.asyncio
async def test_kanban_cancel_rejects_unknown_params(owner):
    """The closed param set is enforced before the handler runs."""
    connection, home = owner
    parent, _child, _other = _seed()
    response = await _dispatch(
        connection, "kanban.cancel", {"task_id": parent, "surprise": True},
    )
    assert response["error"]["message"] == "invalid_params", response
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, parent).status == "ready"
