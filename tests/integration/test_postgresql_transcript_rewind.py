"""Real PostgreSQL rewind receipts, CAS, and carrier-retention contract."""
from __future__ import annotations

import uuid

import pytest

from agent.context_compressor import HISTORICAL_TASK_HEADING, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from cli_session_store import PostgreSQLCLISessionStore
from hermes_state_rewind import rewind_user_turn
from state_store import MessageRecord, PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)


def _carrier(ask: str) -> str:
    return f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold\n\n{_SUMMARY_END_MARKER}\n\n{ask}"


def _store(monkeypatch, target: OwnedPostgreSQLTestTarget) -> PostgreSQLStateStore:
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    return PostgreSQLStateStore(_SETTINGS, _DSN, schema=target.schema)


def test_rewind_is_receipted_soft_delete_with_cas_and_recovery(monkeypatch, postgresql_test_target):
    store = _store(monkeypatch, postgresql_test_target)
    session_id = f"rewind-{uuid.uuid4()}"
    try:
        store.ensure_session(session_id, source="test")
        first = store.append_message_record(session_id, MessageRecord(role="user", content="first"))
        store.append_message_record(session_id, MessageRecord(role="assistant", content="answer"))
        target = store.append_message_record(session_id, MessageRecord(role="user", content="second"))
        store.append_message_record(session_id, MessageRecord(role="assistant", content="failed"))
        active_ids = store.get_active_message_ids(session_id)
        request_id = uuid.uuid4().hex

        receipt = store.rewind_to_message(session_id, target, expected_active_ids=active_ids,
                                          expected_target_content="second", request_id=request_id)
        assert receipt["rewound_count"] == 2
        assert store.get_active_message_ids(session_id) == [first, first + 1]
        durable = store.get_rewind_receipt(request_id)
        assert durable is not None and durable["retired_count"] == 2
        assert durable["active_prefix_ids"] == [first, first + 1]
        # A lost acknowledgement may only be recovered through the same receipt;
        # it never executes a second destructive mutation.
        assert store.rewind_to_message(session_id, target, request_id=request_id)["rewound_count"] == 2
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT rewind_count FROM {store._schema}.sessions WHERE id=%s", (session_id,))
            assert cursor.fetchone()[0] == 1
        with pytest.raises(RuntimeError, match="active transcript changed"):
            store.rewind_to_message(session_id, first, expected_active_ids=active_ids,
                                    expected_target_content="first", request_id=uuid.uuid4().hex)
    finally:
        store.close()


def test_rewind_retains_only_composite_handoff_and_refuses_live_leases(monkeypatch, postgresql_test_target):
    store = _store(monkeypatch, postgresql_test_target)
    session_id = f"rewind-carrier-{uuid.uuid4()}"
    try:
        store.ensure_session(session_id, source="test")
        target = store.append_message_record(session_id, MessageRecord(role="user", content=_carrier("ask")))
        store.append_message_record(session_id, MessageRecord(role="assistant", content="failed"))
        ids = store.get_active_message_ids(session_id)
        assert store.try_acquire_session_turn_lease(session_id, "foreign", ttl_seconds=30)
        with pytest.raises(RuntimeError, match="active turn lease"):
            store.rewind_to_message(session_id, target, expected_active_ids=ids, expected_target_content="ask")
        assert store.get_active_message_ids(session_id) == ids
        store.release_session_turn_lease(session_id, "foreign")
        result = store.rewind_to_message(session_id, target, preserve_compaction_handoff=True,
                                         expected_active_ids=ids, expected_target_content="ask")
        assert result["replacement_message_id"] is not None
        assert store.get_active_message_ids(session_id) == [result["replacement_message_id"]]
        records = store.get_message_records(session_id)
        assert records[-1]["display_kind"] == "hidden"
        assert "\n\nask" not in records[-1]["content"]
    finally:
        store.close()


def test_facade_rewind_recovers_lost_acknowledgement_from_its_request_receipt(monkeypatch, postgresql_test_target):
    store = _store(monkeypatch, postgresql_test_target)
    session_id = f"rewind-facade-{uuid.uuid4()}"
    try:
        store.ensure_session(session_id, source="test")
        store.append_message_record(session_id, MessageRecord(role="user", content="keep"))
        store.append_message_record(session_id, MessageRecord(role="assistant", content="kept"))
        store.append_message_record(session_id, MessageRecord(role="user", content="retry"))
        store.append_message_record(session_id, MessageRecord(role="assistant", content="retired"))
        facade = PostgreSQLCLISessionStore(store)
        original = facade.rewind_to_message
        acknowledged = False

        def lose_first_ack(*args, **kwargs):
            nonlocal acknowledged
            result = original(*args, **kwargs)
            if not acknowledged:
                acknowledged = True
                raise ConnectionError("simulated acknowledgement loss")
            return result

        monkeypatch.setattr(facade, "rewind_to_message", lose_first_ack)
        outcome = rewind_user_turn(facade, session_id, -1, require_retryable=True)
        assert outcome.request_id
        assert outcome.live_text == "retry"
        assert outcome.rewound_count == 2
        assert store.get_rewind_receipt(outcome.request_id) is not None
        assert len(store.get_active_message_ids(session_id)) == 2
    finally:
        store.close()
