"""Real PostgreSQL contract for the gateway transcript slice (v25), parity-tested
against a temp SQLite SessionDB oracle: replace_messages (both active_only modes
and lease-guard rejection), conversation projection equality, latest-role /
latest-row-id probes, the gateway input-owner marker, api-content sidecar
setters, and platform_message_id unique-partial-index duplicate rejection."""
from __future__ import annotations

import pytest

from hermes_state import SessionDB
from hermes_state_errors import SessionTurnLeaseLostError
from state_store import MessageRecord, PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore

_SETTINGS = PostgreSQLStateStoreConfig(
    dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)


def _pg_store(target):
    return PostgreSQLStateStore(_SETTINGS, target.dsn, schema=target.schema)


@pytest.fixture()
def oracle(tmp_path):
    db = SessionDB(db_path=tmp_path / "oracle.db")
    db.create_session("sess", source="gateway")
    try:
        yield db
    finally:
        db.close()


def _seed(db) -> None:
    db.append_message("sess", "user", "hello gateway", platform_message_id="plat-1")
    db.append_message("sess", "assistant", "hi there", finish_reason="stop",
                      reasoning="thinking", tool_name=None)


def _seed_pg(store) -> None:
    store.ensure_session("sess", source="gateway")
    store.append_message_record("sess", MessageRecord(role="user", content="hello gateway", platform_message_id="plat-1"))
    store.append_message_record("sess", MessageRecord(role="assistant", content="hi there", finish_reason="stop", reasoning="thinking"))


def _project(msgs):
    """Compare the fields the gateway transcript read path consumes."""
    out = []
    for msg in msgs:
        row = {"role": msg["role"], "content": msg["content"]}
        for key in ("message_id", "observed", "tool_call_id", "tool_name", "tool_calls",
                    "finish_reason", "reasoning", "reasoning_content", "api_content",
                    "display_kind", "display_metadata", "effect_disposition"):
            if msg.get(key) is not None:
                row[key] = msg[key]
        out.append(row)
    return out


def test_postgresql_replace_messages_parity_both_active_only_modes(postgresql_test_target, oracle):
    _seed(oracle)
    store = _pg_store(postgresql_test_target)
    try:
        _seed_pg(store)
        # Soft-archive one live row on both stores (simulates in-place compaction).
        store.replace_messages("sess", [], active_only=False, archive_dropped=True)
        oracle.replace_messages("sess", [], active_only=False, archive_dropped=True)
        replacement = [{"role": "user", "content": "rewritten"}]
        # active_only=True path: archived rows must survive the destructive drop.
        store.replace_messages("sess", replacement, active_only=True)
        oracle.replace_messages("sess", replacement, active_only=True)
        assert _project(store.get_messages_as_conversation("sess", include_inactive=True)) == \
            _project(oracle.get_messages_as_conversation("sess", include_inactive=True))
        # Destructive full replace.
        store.replace_messages("sess", [{"role": "user", "content": "final"}])
        oracle.replace_messages("sess", [{"role": "user", "content": "final"}])
        assert _project(store.get_messages_as_conversation("sess", include_inactive=True)) == \
            _project(oracle.get_messages_as_conversation("sess", include_inactive=True))
    finally:
        store.close()


def test_postgresql_replace_messages_rejects_active_turn_lease(postgresql_test_target, oracle):
    _seed(oracle)
    assert oracle.try_acquire_session_turn_lease("sess", "turn-owner")
    replacement = [{"role": "user", "content": "must not land"}]
    with pytest.raises(SessionTurnLeaseLostError):
        oracle.replace_messages("sess", replacement, reject_active_turn_lease=True)
    store = _pg_store(postgresql_test_target)
    try:
        _seed_pg(store)
        assert store.try_acquire_session_turn_lease("sess", "turn-owner")
        with pytest.raises(SessionTurnLeaseLostError):
            store.replace_messages("sess", replacement, reject_active_turn_lease=True)
        # The refusal preceded mutation: the live transcript is intact.
        assert store.latest_conversation_role("sess") == "assistant"
        # After release the same rewrite commits.
        store.release_session_turn_lease("sess", "turn-owner")
        store.replace_messages("sess", replacement, reject_active_turn_lease=True)
        assert store.latest_conversation_role("sess") == "user"
    finally:
        store.close()


def test_postgresql_conversation_projection_and_latest_probes_parity(postgresql_test_target, oracle):
    _seed(oracle)
    store = _pg_store(postgresql_test_target)
    try:
        _seed_pg(store)
        assert _project(store.get_messages_as_conversation("sess")) == \
            _project(oracle.get_messages_as_conversation("sess"))
        with_ids = store.get_messages_as_conversation("sess", include_row_ids=True)
        assert [msg["_row_id"] for msg in with_ids] == [msg["_row_id"] for msg in oracle.get_messages_as_conversation("sess", include_row_ids=True)]
        assert store.latest_conversation_role("sess") == oracle.latest_conversation_role("sess") == "assistant"
        assert store.latest_message_row_id("sess") is not None
        assert store.latest_message_row_id("sess") == oracle.latest_message_row_id("sess")
        assert store.latest_message_row_id("sess", role="assistant") == oracle.latest_message_row_id("sess", role="assistant")
        assert store.latest_message_row_id("sess", offset=1) == oracle.latest_message_row_id("sess", offset=1)
        assert store.latest_message_row_id("missing") is None and oracle.latest_message_row_id("missing") is None
    finally:
        store.close()


def test_postgresql_has_gateway_input_owner(postgresql_test_target, oracle):
    _seed(oracle)
    store = _pg_store(postgresql_test_target)
    try:
        _seed_pg(store)
        # SQLite oracle needs the marker written through display_metadata.
        marker = {"gateway_input_owner": "gateway-input-1"}
        marked_row = oracle.append_message("sess", "user", "accepted", observed=False, display_metadata=marker)
        store.append_message_record("sess", MessageRecord(role="user", content="accepted", observed=False, display_metadata=marker))
        assert oracle.has_gateway_input_owner("sess", "gateway-input-1") is True
        assert store.has_gateway_input_owner("sess", "gateway-input-1") is True
        assert store.has_gateway_input_owner("sess", "someone-else") is False
        assert store.has_gateway_input_owner("missing", "gateway-input-1") is False
    finally:
        store.close()


def test_postgresql_api_content_setters_round_trip(postgresql_test_target, oracle):
    _seed(oracle)
    store = _pg_store(postgresql_test_target)
    try:
        _seed_pg(store)
        pg_rows = store.get_messages_as_conversation("sess", include_row_ids=True)
        row_id = next(msg["_row_id"] for msg in pg_rows if msg["role"] == "user")
        assert store.set_message_api_content("sess", row_id, "hello gateway", "exact bytes") == 1
        assert oracle.set_latest_user_api_content("sess", "hello gateway", "exact bytes") == 1
        assert [msg.get("api_content") for msg in store.get_messages_as_conversation("sess")] == \
            [msg.get("api_content") for msg in oracle.get_messages_as_conversation("sess")]
        # Content mismatch guard: a neighbouring text must not be overwritten.
        assert store.set_message_api_content("sess", row_id, "different text", "wrong") == 0
        assert store.set_message_api_content("sess", row_id + 999, "hello gateway", "wrong") == 0
        assert store.set_latest_user_api_content("sess", "hello gateway", "exact bytes 2") == 1
    finally:
        store.close()


def test_postgresql_platform_message_id_unique_partial_index(postgresql_test_target):
    store = _pg_store(postgresql_test_target)
    try:
        store.ensure_session("sess", source="gateway")
        store.ensure_session("sess-b", source="gateway")
        store.append_message_record("sess", MessageRecord(role="user", content="one", platform_message_id="plat-dup"))
        with pytest.raises(Exception):
            store.append_message_record("sess-b", MessageRecord(role="user", content="two", platform_message_id="plat-dup"))
        # NULL platform ids stay unconstrained.
        store.append_message_record("sess", MessageRecord(role="user", content="null a"))
        store.append_message_record("sess-b", MessageRecord(role="user", content="null b"))
        assert store.has_platform_message_id("sess", "plat-dup") is True
        assert store.has_platform_message_id("sess", "plat-missing") is False
    finally:
        store.close()
