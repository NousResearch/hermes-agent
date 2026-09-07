"""Tests for #104653 — inbound user turns persisted twice (gateway + agent flush).

Every inbound user turn on a gateway messaging platform was written to
``messages`` twice: once by the gateway on receipt (with
``platform_message_id`` set) and once by the agent runtime at turn flush
(with ``platform_message_id`` NULL). History load does not de-duplicate,
so the model saw each user message twice on every rehydration.

Behavior contract (real SessionDB, no mocks on the write path):

1. The agent flush must skip a user row whose ``platform_message_id``
   already has a live row in the session (the gateway wrote it first).
2. The flush row for the current-turn user must carry the turn's
   ``platform_message_id`` (``_persist_user_message_platform_id``), so the
   second writer shares the join key instead of emitting NULL.
3. Anything without a key still persists (fail-open: never lose content),
   including when the dedup probe itself errors.
"""

import types

import pytest

from agent.session_persistence import (
    _db_flush_collect,
    _db_flush_row,
    _db_flush_write,
)
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    d.create_session("sess-104653", source="gateway")
    yield d
    d.close()


def _agent(db, **attrs):
    """Duck-typed agent stub for the module-level flush phases."""
    agent = types.SimpleNamespace(
        session_id="sess-104653",
        _session_db=db,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _persist_user_message_platform_id=None,
        _pending_cli_user_message=None,
    )
    for key, value in attrs.items():
        setattr(agent, key, value)
    return agent


def _user_rows(db, session_id="sess-104653"):
    return db._conn.execute(
        "SELECT content, platform_message_id FROM messages "
        "WHERE session_id = ? AND role = 'user' AND active = 1 ORDER BY id",
        (session_id,),
    ).fetchall()


def test_flush_skips_user_row_already_persisted_by_gateway(db):
    """Gateway wrote the inbound turn first; the agent flush must not add a copy."""
    db.append_message(
        session_id="sess-104653",
        role="user",
        content="hello from telegram",
        platform_message_id="tg-2146",
        timestamp=1756685929,  # whole seconds, as the platform event carries
    )
    agent = _agent(db)
    live = {
        "role": "user",
        "content": "hello from telegram",
        "platform_message_id": "tg-2146",
        "timestamp": 1756685929.123,
    }
    batch_rows, batch_msgs = _db_flush_collect(agent, [live], None)
    assert batch_rows == [] and batch_msgs == []
    _db_flush_write(agent, batch_rows, batch_msgs)
    rows = _user_rows(db)
    assert len(rows) == 1
    assert tuple(rows[0]) == ("hello from telegram", "tg-2146")


def test_flush_row_carries_turn_platform_id(db):
    """The turn knows the inbound id; the flushed row must share the join key."""
    agent = _agent(db, _persist_user_message_idx=0, _persist_user_message_platform_id="tg-99")
    live = {"role": "user", "content": "question", "timestamp": 1756686000.5}
    row = _db_flush_row(agent, live, True)
    assert row["platform_message_id"] == "tg-99"
    batch_rows, batch_msgs = _db_flush_collect(agent, [live], None)
    _db_flush_write(agent, batch_rows, batch_msgs)
    rows = _user_rows(db)
    assert len(rows) == 1
    assert tuple(rows[0]) == ("question", "tg-99")


def test_flush_does_not_clobber_explicit_row_platform_id(db):
    """A live dict carrying its own id keeps it; the turn override only fills gaps."""
    agent = _agent(db, _persist_user_message_idx=0, _persist_user_message_platform_id="tg-other")
    live = {"role": "user", "content": "q", "platform_message_id": "tg-mine"}
    assert _db_flush_row(agent, live, True)["platform_message_id"] == "tg-mine"


def test_flush_still_writes_user_without_platform_id(db):
    """No key anywhere: the row must still land (dedupe never drops content)."""
    agent = _agent(db)
    live = {"role": "user", "content": "[System note: shutdown]", "timestamp": 1756686100.25}
    batch_rows, batch_msgs = _db_flush_collect(agent, [live], None)
    assert len(batch_rows) == 1
    _db_flush_write(agent, batch_rows, batch_msgs)
    assert len(_user_rows(db)) == 1


def test_flush_writes_user_with_unseen_platform_id(db):
    """A fresh inbound id (no gateway row yet) persists normally."""
    db.append_message(
        session_id="sess-104653", role="user", content="older", platform_message_id="tg-1",
    )
    agent = _agent(db)
    live = {"role": "user", "content": "newer", "platform_message_id": "tg-2"}
    batch_rows, batch_msgs = _db_flush_collect(agent, [live], None)
    assert len(batch_rows) == 1
    _db_flush_write(agent, batch_rows, batch_msgs)
    assert [tuple(r) for r in _user_rows(db)] == [("older", "tg-1"), ("newer", "tg-2")]


def test_flush_probe_failure_fails_open(db, monkeypatch):
    """A broken dedup probe must not lose the user turn."""
    db.append_message(
        session_id="sess-104653", role="user", content="hi", platform_message_id="tg-7",
    )
    agent = _agent(db)

    def _boom(session_id, platform_message_id):
        raise RuntimeError("db probe exploded")

    monkeypatch.setattr(agent._session_db, "has_platform_message_id", _boom)
    live = {"role": "user", "content": "hi", "platform_message_id": "tg-7"}
    batch_rows, batch_msgs = _db_flush_collect(agent, [live], None)
    assert len(batch_rows) == 1
