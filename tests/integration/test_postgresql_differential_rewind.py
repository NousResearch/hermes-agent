"""Differential parity: user-turn rewind — SQLite ``SessionDB`` oracle vs PostgreSQL.

Each scenario runs against both backends and asserts the same behavioral contract
(visible transcript, turn classification, error classes), never the storage
internals.  Receipt visibility is the one backend-specific capability and is
asserted as such.
"""
from __future__ import annotations

import uuid

import pytest

from hermes_state import SessionDB
from hermes_state_rewind import RewindIndeterminateError, RewindTargetUnavailableError, rewind_user_turn
from state_store import MessageRecord, PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def backend(request, tmp_path, monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget):
    if request.param == "sqlite":
        store = SessionDB(db_path=tmp_path / "state.db")
    else:
        monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
        store = PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    try:
        yield request.param, store
    finally:
        store.close()


def _append(store, session_id: str, role: str, content=None, **extra) -> None:
    """Seed a message through whichever append surface the backend exposes."""
    if hasattr(store, "append_message_record"):
        store.append_message_record(session_id, MessageRecord(role=role, content=content, **extra))
    else:
        store.append_message(session_id, role=role, content=content, **extra)


def _seed(store, session_id: str) -> None:
    store.ensure_session(session_id, source="test")
    for text in ("first", "second", "third"):
        _append(store, session_id, "user", text)
        _append(store, session_id, "assistant", f"answer-{text}")


def _visible(store, session_id: str):
    return [(m["role"], m["content"]) for m in store.get_messages_as_conversation(session_id)]


def test_rewind_minus_one(backend):
    kind, store = backend
    sid = f"rewind-one-{uuid.uuid4()}"
    _seed(store, sid)
    assert _visible(store, sid) == [
        ("user", "first"), ("assistant", "answer-first"),
        ("user", "second"), ("assistant", "answer-second"),
        ("user", "third"), ("assistant", "answer-third"),
    ]

    outcome = rewind_user_turn(store, sid, -1)
    assert outcome.rewound_count == 2
    assert outcome.turns_undone == 1
    assert outcome.live_text == "third"
    assert _visible(store, sid) == [
        ("user", "first"), ("assistant", "answer-first"),
        ("user", "second"), ("assistant", "answer-second"),
    ]


def test_rewind_minus_two(backend):
    kind, store = backend
    sid = f"rewind-two-{uuid.uuid4()}"
    _seed(store, sid)

    outcome = rewind_user_turn(store, sid, -2)
    assert outcome.rewound_count == 4
    assert outcome.turns_undone == 2
    assert outcome.live_text == "second"
    assert _visible(store, sid) == [("user", "first"), ("assistant", "answer-first")]


def test_rewind_out_of_range_ordinal_raises_value_error(backend):
    kind, store = backend
    sid = f"rewind-range-{uuid.uuid4()}"
    _seed(store, sid)
    with pytest.raises(RewindTargetUnavailableError) as exc:
        rewind_user_turn(store, sid, 5)
    assert isinstance(exc.value, ValueError)


def test_rewind_negative_ordinal_on_empty_raises_value_error(backend):
    kind, store = backend
    sid = f"rewind-empty-{uuid.uuid4()}"
    store.ensure_session(sid, source="test")
    with pytest.raises(RewindTargetUnavailableError) as exc:
        rewind_user_turn(store, sid, -1)
    assert isinstance(exc.value, ValueError)


def test_rewind_classifies_only_user_originated_turns(backend):
    kind, store = backend
    sid = f"rewind-class-{uuid.uuid4()}"
    store.ensure_session(sid, source="test")
    _append(store, sid, "user", "first")
    _append(store, sid, "assistant", "a1")
    _append(store, sid, "tool", "tool-result", tool_name="t", tool_call_id="call-1")
    _append(store, sid, "user", "second")
    _append(store, sid, "assistant", "a2")

    outcome = rewind_user_turn(store, sid, -1)
    # The tool row is not user-originated: -1 targets "second", never the tool row.
    assert outcome.live_text == "second"
    assert outcome.rewound_count == 2
    assert [m["role"] for m in store.get_messages_as_conversation(sid)] == ["user", "assistant", "tool"]


def test_rewind_repairs_adjacent_durable_user_turns(backend):
    """Rewind addresses the same repaired projection the live resume path exposes."""
    _kind, store = backend
    sid = f"rewind-repaired-{uuid.uuid4()}"
    store.ensure_session(sid, source="test")
    _append(store, sid, "user", "first question")
    _append(store, sid, "user", "second question")
    _append(store, sid, "assistant", "answer")

    outcome = rewind_user_turn(store, sid, -1)

    assert outcome.turns_undone == 1
    assert "first question" in outcome.live_text
    assert "second question" in outcome.live_text
    assert _visible(store, sid) == []


def test_rewind_receipt_visibility(backend):
    kind, store = backend
    sid = f"rewind-receipt-{uuid.uuid4()}"
    _seed(store, sid)
    outcome = rewind_user_turn(store, sid, -1)
    assert outcome.request_id
    if kind == "postgresql":
        receipt = store.get_rewind_receipt(outcome.request_id)
        assert receipt is not None and receipt["retired_count"] == 2
    else:
        assert getattr(store, "get_rewind_receipt", None) is None


def test_rewind_error_class_hierarchy_is_backend_neutral():
    assert issubclass(RewindTargetUnavailableError, ValueError)
    assert issubclass(RewindIndeterminateError, RuntimeError)
