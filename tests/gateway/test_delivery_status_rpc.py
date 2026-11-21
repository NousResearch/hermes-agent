"""``delivery.status`` (Q5, puente B): the read-only delivery-state verb.

Contract: transcript detached-result rows + bot relay receipts, per session or every visible
session; ``session:read`` + owning profile; no writes; answers during the drain. The plugin
outbox is outside the contract by design and is asserted nowhere here.
"""

import json
import sqlite3
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence.conversation.messages import DelegationDelivery, NewMessage
from persistence.conversation.sessions import END_REASON_COMPRESSION
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session
from tui_gateway.contracts.registry import CANONICAL_METHODS

NOW = time.time()


def _delivery_row(db, session_id, delegation_id, timestamp):
    """A detached delegation delivery row, as ``append_delegation_delivery`` records it,
    with a controlled timestamp (that verb takes no timestamp, so the row is written
    with its shape: role ``user``, the same display kind and the same identity key)."""
    return db.messages.append(
        session_id,
        NewMessage(
            role="user",
            content=f"delivery {delegation_id}",
            display_kind="async_delegation_complete",
            display_metadata={"delegation_id": delegation_id},
            timestamp=timestamp,
        ),
    )


def _assistant_row(db, session_id, timestamp):
    return db.messages.append(
        session_id, NewMessage(role="assistant", content="done", timestamp=timestamp)
    )


def _sqlite_snapshot(home):
    """``(message count, newest message id, [(session id, message_count)])`` read on a connection
    of its own, so a caller can prove a verb wrote nothing."""
    conn = sqlite3.connect(str(home / "state.db"))
    try:
        messages = conn.execute("SELECT COUNT(*), MAX(id) FROM messages").fetchone()
        sessions = conn.execute("SELECT id, message_count FROM sessions ORDER BY id").fetchall()
    finally:
        conn.close()
    return messages, sessions


def _receipt(home, delivery_id, session_id, status, created_at=None, profile_home=None):
    """A canonical relay receipt file in the profile home's mailbox."""
    root = home / "runtime" / "bot_live_delivery"
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "delivery_id": delivery_id,
        "admission_id": f"adm-{delivery_id[:8]}",
        "profile_home": profile_home if profile_home is not None else str(home),
        "session_id": session_id,
        "principal_id": "owner",
        "message": "result",
        "status": status,
    }
    if created_at is not None:
        record["created_at"] = created_at
    (root / f"{delivery_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return record


@pytest_asyncio.fixture
async def owner(tmp_path):
    home = tmp_path
    db = open_persistence(profile_scope(home=home))
    try:
        for sid in ("s", "empty", "other"):
            seed_session(db, sid, "test")
        epoch = db.runtime.begin_epoch(instance_id="current")
        authority = SessionAuthority(
            SimpleNamespace(_draining=False),
            profile_id="owned",
            instance_id="current",
            epoch=epoch,
            persistence=db,
        )
        for sid in ("s", "empty", "other"):
            authority.sessions[sid] = LiveSession(None, "route")
        connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
        try:
            yield connection, db, home
        finally:
            await connection.close()
    finally:
        db.close()


@pytest_asyncio.fixture
async def seeded(owner):
    """``s``: one consumed delivery (before the last turn), two pending (after it), one
    stale delivery (before the last turn AND outside the default window); ``empty``: nothing
    at all."""
    connection, db, home = owner
    _assistant_row(db, "s", NOW - 20000)
    _delivery_row(db, "s", "consumed", NOW - 30000)  # before the first turn below
    _assistant_row(db, "s", NOW - 10000)  # last turn: consumes "consumed"
    _delivery_row(db, "s", "pending-1", NOW - 5000)
    _delivery_row(db, "s", "pending-2", NOW - 1000)
    _delivery_row(db, "s", "stale", NOW - 90000)  # outside the 86400 s window
    return connection, db, home


def _row(result, session_id):
    rows = [row for row in result["sessions"] if row["session_id"] == session_id]
    assert len(rows) == 1, result
    return rows[0]


@pytest.mark.asyncio
async def test_counts_pending_consumed_and_recent(seeded):
    connection, _db, _home = seeded
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
    )
    assert "error" not in response, response
    CANONICAL_METHODS["delivery.status"].result.model_validate(response["result"])
    assert response["result"]["scope"] == "session"
    assert response["result"]["window_seconds"] == 86400
    row = _row(response["result"], "s")
    # Two deliveries after the last assistant row; the older one is consumed, the stale
    # one predates the last turn too (old but consumed). Pending is about the last turn,
    # not the window.
    assert row["pending"] == 2, row
    assert row["oldest_pending_age_seconds"] == pytest.approx(5000, abs=120), row
    # Inside the window: consumed, pending-1, pending-2 (the stale row is outside it).
    assert row["delivered_recent"] == 3, row
    assert row["dead_letter"] == 0, row
    assert row["last_delivery_at"] == pytest.approx(NOW - 1000, abs=120), row
    assert row["source"] == {
        "pending": "transcript",
        "delivered_recent": "transcript+relay",
        "dead_letter": "relay",
        "last_delivery_at": "transcript+relay",
    }


@pytest.mark.asyncio
async def test_real_delegation_writer_shape_counts(seeded):
    """Rows written by the real ``append_delegation_delivery`` (not the test's hand-built
    rows) count as deliveries, pending until the next turn."""
    connection, db, _home = seeded
    db.messages.append_delegation_delivery(
        "s", DelegationDelivery(content="real delivery", metadata={"delegation_id": "real-1"})
    )
    response = await connection.dispatch(
        {"id": 2, "method": "delivery.status", "params": {"session_id": "s"}}
    )
    assert _row(response["result"], "s")["pending"] == 3, response


@pytest.mark.asyncio
async def test_session_without_deliveries_returns_zeros(owner):
    connection, _db, _home = owner
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "empty"}}
    )
    assert "error" not in response, response
    row = _row(response["result"], "empty")
    assert row == {
        "session_id": "empty",
        "pending": 0,
        "oldest_pending_age_seconds": None,
        "delivered_recent": 0,
        "dead_letter": 0,
        "last_delivery_at": None,
        "source": {
            "pending": "transcript",
            "delivered_recent": "transcript+relay",
            "dead_letter": "relay",
            "last_delivery_at": "transcript+relay",
        },
    }


@pytest.mark.asyncio
async def test_missing_session_id_and_scope_is_invalid_params(owner):
    connection, _db, _home = owner
    for params in (
        {},
        {"session_id": "s", "scope": "all"},
        {"limit": 10},
        {"session_id": "s", "limit": 0},
        {"session_id": "s", "limit": 201},
        {"session_id": "s", "window_seconds": 0},
        {"session_id": "s", "scope": "everything"},
        {"session_id": ""},
    ):
        response = await connection.dispatch(
            {"id": 1, "method": "delivery.status", "params": params}
        )
        assert response["error"]["code"] == 4001, (params, response)
        assert response["error"]["data"]["reason"] == "invalid_params", (
            params,
            response,
        )
        assert "result" not in response


@pytest.mark.asyncio
async def test_unknown_session_is_not_found(owner):
    connection, _db, _home = owner
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "nope"}}
    )
    assert response["error"]["code"] == 4001, response
    assert response["error"]["message"] == "not_found", response
    assert "result" not in response


@pytest.mark.asyncio
async def test_actor_without_session_read_is_denied(owner, tmp_path):
    connection, _db, _home = owner
    authority = connection.authority
    reader = AuthorityConnection(
        authority, object(), {"user_id": "owner", "capabilities": ["session:submit"]}
    )
    try:
        response = await reader.dispatch(
            {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
        )
    finally:
        await reader.close()
    assert response["error"]["code"] == 4001, response
    assert response["error"]["message"] == "permission_denied", response
    assert "result" not in response


@pytest.mark.asyncio
async def test_other_profile_gets_no_data(owner):
    connection, _db, _home = owner
    authority = connection.authority
    foreign = AuthorityConnection(
        authority, object(), {"user_id": "owner", "profile_id": "other"}
    )
    try:
        response = await foreign.dispatch(
            {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
        )
    finally:
        await foreign.close()
    # The codebase's established refusal for a foreign profile is profile_mismatch (the
    # dispatcher's sibling routing); either way no session data crosses the boundary.
    assert response["error"]["code"] == 4001, response
    assert response["error"]["message"] == "profile_mismatch", response
    assert "result" not in response


@pytest.mark.asyncio
async def test_verb_writes_nothing(seeded):
    connection, db, home = seeded
    before_messages, before_sessions = _sqlite_snapshot(home)
    before_receipts = (
        sorted((home / "runtime" / "bot_live_delivery").glob("*.json"))
        if (home / "runtime" / "bot_live_delivery").exists()
        else []
    )
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
    )
    assert "error" not in response, response
    after_messages, after_sessions = _sqlite_snapshot(home)
    assert after_messages == before_messages
    assert after_sessions == before_sessions
    after_receipts = (
        sorted((home / "runtime" / "bot_live_delivery").glob("*.json"))
        if (home / "runtime" / "bot_live_delivery").exists()
        else []
    )
    assert after_receipts == before_receipts


@pytest.mark.asyncio
async def test_relay_receipts_count_dead_letter_and_recent(seeded):
    connection, _db, home = seeded
    _receipt(home, "a" * 32, "s", "settled", created_at=NOW - 100)
    _receipt(home, "b" * 32, "s", "failed", created_at=NOW - 200)
    _receipt(
        home, "c" * 32, "s", "queued", created_at=NOW - 50
    )  # in flight, not a delivery
    _receipt(home, "d" * 32, "other", "failed", created_at=NOW - 50)  # another session
    _receipt(
        home,
        "e" * 32,
        "s",
        "failed",
        created_at=NOW - 50,
        profile_home="/some/other/home",
    )  # another profile's home: ignored
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
    )
    assert "error" not in response, response
    row = _row(response["result"], "s")
    assert row["dead_letter"] == 1, row
    # Transcript recent (3) + the one settled relay receipt.
    assert row["delivered_recent"] == 4, row
    # The newest delivery is the settled receipt, newer than the newest transcript row.
    assert row["last_delivery_at"] == pytest.approx(NOW - 100, abs=120), row


@pytest.mark.asyncio
async def test_scope_all_lists_visible_sessions(seeded):
    connection, db, _home = seeded
    db.sessions.set_hidden("other", True)
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"scope": "all"}}
    )
    assert "error" not in response, response
    CANONICAL_METHODS["delivery.status"].result.model_validate(response["result"])
    assert response["result"]["scope"] == "all"
    got = {row["session_id"]: row for row in response["result"]["sessions"]}
    assert set(got) == {"s", "empty"}, got  # hidden "other" is not visible
    assert got["s"]["pending"] == 2
    assert got["empty"]["pending"] == 0
    limited = await connection.dispatch(
        {"id": 2, "method": "delivery.status", "params": {"scope": "all", "limit": 1}}
    )
    assert len(limited["result"]["sessions"]) == 1, limited


@pytest.mark.asyncio
async def test_verb_answers_while_draining(seeded):
    connection, _db, _home = seeded
    authority = connection.authority
    authority.runner._draining = True
    operator = AuthorityConnection(
        authority, object(), {"user_id": "owner"}, operator=True
    )
    try:
        response = await operator.dispatch(
            {"id": 1, "method": "delivery.status", "params": {"session_id": "s"}}
        )
    finally:
        await operator.close()
        authority.runner._draining = False
    assert "error" not in response, response
    assert _row(response["result"], "s")["pending"] == 2, response


@pytest.mark.asyncio
async def test_window_seconds_is_honoured(seeded):
    connection, _db, _home = seeded
    response = await connection.dispatch(
        {
            "id": 1,
            "method": "delivery.status",
            "params": {"session_id": "s", "window_seconds": 6000},
        }
    )
    row = _row(response["result"], "s")
    # 6000 s window: pending-1 (5000 s ago) and pending-2 (1000 s ago); the consumed row
    # (30000 s ago) is outside it.
    assert row["delivered_recent"] == 2, row
    narrow = await connection.dispatch(
        {
            "id": 2,
            "method": "delivery.status",
            "params": {"session_id": "s", "window_seconds": 2000},
        }
    )
    assert _row(narrow["result"], "s")["delivered_recent"] == 1, narrow


@pytest.mark.asyncio
async def test_explicit_null_limit_and_window_use_defaults(owner):
    """The contract's null is "not provided": explicit nulls take the defaults."""
    connection, _db, _home = owner
    response = await connection.dispatch(
        {
            "id": 1,
            "method": "delivery.status",
            "params": {"session_id": "empty", "limit": None, "window_seconds": None},
        }
    )
    assert "error" not in response, response
    assert response["result"]["window_seconds"] == 86400, response


@pytest.mark.asyncio
async def test_delivery_on_compressed_parent_counts_for_the_tip(seeded):
    """A delivery recorded before a compaction rotates the session still counts: the verb
    aggregates the resume lineage, not the tip segment alone — detached rows survive
    compaction through the lineage, as the detached-result contract promises."""
    connection, db, _home = seeded
    db.sessions.end("s", END_REASON_COMPRESSION)
    seed_session(db, "s2", "test", parent_session_id="s")
    connection.authority.sessions["s2"] = LiveSession(None, "route")
    assert db.lineage.resolve_resume_target("s") == "s2"
    assert db.lineage.resume_lineage_ids("s2") == ("s", "s2")
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "s2"}}
    )
    assert "error" not in response, response
    row = _row(response["result"], "s2")
    assert row["pending"] == 2, row
    assert row["delivered_recent"] == 3, row
    assert row["last_delivery_at"] == pytest.approx(NOW - 1000, abs=120), row


@pytest.mark.asyncio
async def test_relay_receipt_for_compressed_parent_counts_for_the_tip(owner):
    """A relay receipt filed under the pre-rotation id belongs to the same conversation."""
    connection, db, home = owner
    db.sessions.end("s", END_REASON_COMPRESSION)
    seed_session(db, "s2", "test", parent_session_id="s")
    connection.authority.sessions["s2"] = LiveSession(None, "route")
    _receipt(home, "f" * 32, "s", "failed", created_at=NOW - 50)
    response = await connection.dispatch(
        {"id": 1, "method": "delivery.status", "params": {"session_id": "s2"}}
    )
    assert "error" not in response, response
    assert _row(response["result"], "s2")["dead_letter"] == 1, response
