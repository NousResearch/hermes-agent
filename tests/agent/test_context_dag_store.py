import sqlite3

import pytest

from agent.context_dag_store import ContextDAGStore
from hermes_state import SCHEMA_VERSION, SessionDB


DAG_TABLES = {
    "context_message_parts",
    "context_summary_nodes",
    "context_summary_edges",
    "context_summary_sources",
    "context_projection",
    "context_checkpoints",
    "context_mutation_log",
}


def make_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("sess-1", "test", model="test-model")
    return db


def table_names(db):
    with db._lock:
        rows = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {row[0] for row in rows}


def test_empty_db_init_creates_additive_dag_schema(tmp_path):
    db = make_db(tmp_path)
    try:
        assert DAG_TABLES.issubset(table_names(db))
        with db._lock:
            version = db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
            indexes = {
                row[0]
                for row in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
        assert version == SCHEMA_VERSION
        assert "idx_context_summary_nodes_session_status_created" in indexes
        assert "idx_context_mutation_log_session_operation_status" in indexes
    finally:
        db.close()


def test_repeated_init_is_idempotent(tmp_path):
    db_path = tmp_path / "state.db"
    db1 = SessionDB(db_path)
    db1.create_session("sess-1", "test")
    db1.close()

    db2 = SessionDB(db_path)
    db2.close()

    db3 = SessionDB(db_path)
    try:
        assert DAG_TABLES.issubset(table_names(db3))
        with db3._lock:
            schema_rows = db3._conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
        assert schema_rows == 1
    finally:
        db3.close()


def test_old_db_upgrade_creates_dag_tables_and_bumps_version(tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()
    conn = sqlite3.connect(db_path)
    for table in DAG_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute("UPDATE schema_version SET version = 11")
    conn.commit()
    conn.close()

    db = SessionDB(db_path)
    try:
        assert DAG_TABLES.issubset(table_names(db))
        with db._lock:
            version = db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == SCHEMA_VERSION
    finally:
        db.close()


def test_summary_source_projection_checkpoint_and_mutation_roundtrip(tmp_path):
    db = make_db(tmp_path)
    try:
        msg1 = db.append_message("sess-1", "user", "hello")
        msg2 = db.append_message(
            "sess-1",
            "assistant",
            "hi",
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "noop"}}],
            reasoning="brief reasoning",
        )
        store = ContextDAGStore(db)
        source_hash = store.deterministic_source_hash(
            [{"start_message_id": msg1, "end_message_id": msg2}]
        )

        node = store.create_summary_node(
            session_id="sess-1",
            kind="leaf",
            summary_text="User greeted assistant; assistant replied.",
            source_hash=source_hash,
            prompt_version="p1",
            summary_model="unit-test",
            token_estimate=8,
            metadata={"window": 1},
        )
        assert node.session_id == "sess-1"
        assert node.source_hash == source_hash
        assert node.metadata == {"window": 1}

        source = store.link_summary_source(
            session_id="sess-1",
            summary_id=node.id,
            source_type="message_span",
            start_message_id=msg1,
            end_message_id=msg2,
            metadata={"ordinal_start": 0, "ordinal_end": 1},
        )
        assert source.start_message_id == msg1
        assert source.end_message_id == msg2
        source_again = store.link_summary_source(
            session_id="sess-1",
            summary_id=node.id,
            source_type="message_span",
            start_message_id=msg1,
            end_message_id=msg2,
            metadata={"ordinal_start": 0, "ordinal_end": 1},
        )
        assert source_again.id == source.id
        assert store.get_summary_sources("sess-1", node.id) == [source_again]
        assert store.get_summary_node("other-session", node.id) is None

        projection = store.write_active_projection(
            session_id="sess-1",
            engine_version="dag-v1",
            projection=[{"role": "system", "content": "summary ref"}],
            fresh_tail_start_message_id=msg2,
            latest_raw_message_id=msg2,
            token_estimate=42,
            metadata={"fresh": True},
        )
        assert projection.projection == [{"role": "system", "content": "summary ref"}]
        assert projection.metadata == {"fresh": True}
        assert store.read_active_projection("sess-1", "dag-v1") == projection

        checkpoint = store.write_checkpoint(
            session_id="sess-1",
            last_ingested_message_id=msg2,
            last_projection_message_id=msg2,
            last_anchor_message_id=msg1,
            anchor_hash=store.deterministic_message_hash({"role": "user", "content": "hello"}),
            metadata={"reason": "test"},
        )
        assert checkpoint.last_ingested_message_id == msg2
        assert checkpoint.metadata == {"reason": "test"}
        assert store.read_checkpoint("sess-1") == checkpoint

        mutation = store.append_mutation_log(
            session_id="sess-1",
            operation="compact",
            status="ok",
            idempotency_key="sess-1:compact:1",
            payload={"summary_id": node.id},
        )
        assert mutation.operation == "compact"
        assert mutation.payload == {"summary_id": node.id}
    finally:
        db.close()


def test_summary_insert_update_is_idempotent(tmp_path):
    db = make_db(tmp_path)
    try:
        store = ContextDAGStore(db)
        first = store.create_summary_node(
            session_id="sess-1",
            summary_text="old summary",
            source_hash="source-hash",
            prompt_version="p1",
            metadata={"n": 1},
        )
        second = store.create_summary_node(
            session_id="sess-1",
            summary_text="new summary",
            source_hash="source-hash",
            prompt_version="p1",
            metadata={"n": 2},
        )
        assert second.id == first.id
        assert second.summary_text == "new summary"
        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at
        assert second.metadata == {"n": 2}

        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_summary_nodes WHERE session_id = ?",
                ("sess-1",),
            ).fetchone()[0]
        assert count == 1
    finally:
        db.close()


def test_message_hash_is_deterministic_and_covers_required_fields():
    left = {
        "content": "hello",
        "role": "assistant",
        "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}],
        "tool_call_id": None,
        "tool_name": "tool-x",
        "reasoning": "why",
    }
    reordered = {
        "reasoning": "why",
        "tool_name": "tool-x",
        "tool_call_id": None,
        "tool_calls": [{"function": {"arguments": "{}", "name": "x"}, "id": "1"}],
        "role": "assistant",
        "content": "hello",
    }
    changed_reasoning = dict(reordered, reasoning="different")

    assert ContextDAGStore.deterministic_message_hash(left) == ContextDAGStore.deterministic_message_hash(reordered)
    assert ContextDAGStore.deterministic_message_hash(left) != ContextDAGStore.deterministic_message_hash(changed_reasoning)


def test_summary_source_without_message_ids_is_idempotent(tmp_path):
    db = make_db(tmp_path)
    try:
        store = ContextDAGStore(db)
        node = store.create_summary_node(
            session_id="sess-1",
            summary_text="memory-only summary",
            source_hash="no-message-source",
            prompt_version="p1",
        )

        first = store.link_summary_source(
            session_id="sess-1",
            summary_id=node.id,
            source_type="external_memory",
            source_id="mem-1",
            metadata={"n": 1},
        )
        second = store.link_summary_source(
            session_id="sess-1",
            summary_id=node.id,
            source_type="external_memory",
            source_id="mem-1",
            metadata={"n": 2},
        )

        assert second.id == first.id
        assert second.start_message_id is None
        assert second.end_message_id is None
        assert second.metadata == {"n": 2}
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_summary_sources WHERE summary_id = ?",
                (node.id,),
            ).fetchone()[0]
        assert count == 1
    finally:
        db.close()


def test_summary_source_rejects_cross_session_message_ids(tmp_path):
    db = make_db(tmp_path)
    try:
        db.create_session("sess-2", "test", model="test-model")
        sess1_msg = db.append_message("sess-1", "user", "private transcript")
        store = ContextDAGStore(db)
        node = store.create_summary_node(
            session_id="sess-2",
            summary_text="sess-2 summary",
            source_hash="sess-2-source",
            prompt_version="p1",
        )

        with pytest.raises(ValueError, match="message ids must exist and belong"):
            store.link_summary_source(
                session_id="sess-2",
                summary_id=node.id,
                source_type="message_span",
                start_message_id=sess1_msg,
            )

        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_summary_sources WHERE summary_id = ?",
                (node.id,),
            ).fetchone()[0]
        assert count == 0
    finally:
        db.close()


def test_projection_rejects_cross_session_message_ids_without_persisting(tmp_path):
    db = make_db(tmp_path)
    try:
        db.create_session("sess-2", "test", model="test-model")
        sess1_msg = db.append_message("sess-1", "user", "private transcript")
        store = ContextDAGStore(db)

        with pytest.raises(ValueError, match="message ids must exist and belong"):
            store.write_active_projection(
                session_id="sess-2",
                engine_version="dag-v1",
                projection=[{"role": "system", "content": "invalid cursor"}],
                fresh_tail_start_message_id=sess1_msg,
            )

        assert store.read_active_projection("sess-2", "dag-v1") is None
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_projection WHERE session_id = ?",
                ("sess-2",),
            ).fetchone()[0]
        assert count == 0
    finally:
        db.close()


def test_checkpoint_rejects_cross_session_message_ids_without_persisting(tmp_path):
    db = make_db(tmp_path)
    try:
        db.create_session("sess-2", "test", model="test-model")
        sess1_msg = db.append_message("sess-1", "user", "private transcript")
        store = ContextDAGStore(db)

        with pytest.raises(ValueError, match="message ids must exist and belong"):
            store.write_checkpoint(
                session_id="sess-2",
                last_ingested_message_id=sess1_msg,
            )

        assert store.read_checkpoint("sess-2") is None
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_checkpoints WHERE session_id = ?",
                ("sess-2",),
            ).fetchone()[0]
        assert count == 0
    finally:
        db.close()


def test_mutation_log_idempotency_key_is_session_scoped(tmp_path):
    db = make_db(tmp_path)
    try:
        db.create_session("sess-2", "test")
        store = ContextDAGStore(db)

        first = store.append_mutation_log(
            session_id="sess-1",
            operation="compact",
            status="ok",
            idempotency_key="same-key",
            payload={"session": 1},
        )
        second = store.append_mutation_log(
            session_id="sess-2",
            operation="compact",
            status="ok",
            idempotency_key="same-key",
            payload={"session": 2},
        )

        assert first.id != second.id
        assert first.session_id == "sess-1"
        assert second.session_id == "sess-2"
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_mutation_log WHERE idempotency_key = ?",
                ("same-key",),
            ).fetchone()[0]
        assert count == 2
    finally:
        db.close()


def test_summary_idempotency_tuple_wins_over_different_supplied_node_id(tmp_path):
    db = make_db(tmp_path)
    try:
        store = ContextDAGStore(db)
        first = store.create_summary_node(
            session_id="sess-1",
            node_id="custom-node-1",
            summary_text="old",
            kind="leaf",
            source_hash="same-source",
            prompt_version="p1",
        )
        second = store.create_summary_node(
            session_id="sess-1",
            node_id="custom-node-2",
            summary_text="new",
            kind="leaf",
            source_hash="same-source",
            prompt_version="p1",
        )

        assert second.id == first.id == "custom-node-1"
        assert second.summary_text == "new"
        with db._lock:
            count = db._conn.execute(
                "SELECT COUNT(*) FROM context_summary_nodes WHERE session_id = ?",
                ("sess-1",),
            ).fetchone()[0]
        assert count == 1
    finally:
        db.close()


def test_public_compute_summary_id_matches_created_source_hash_node_id(tmp_path):
    db = make_db(tmp_path)
    try:
        store = ContextDAGStore(db)
        expected_id = store.compute_summary_id("sess-1", "leaf", "source-hash", "p1")

        node = store.create_summary_node(
            session_id="sess-1",
            summary_text="summary",
            kind="leaf",
            source_hash="source-hash",
            prompt_version="p1",
        )

        assert node.id == expected_id
    finally:
        db.close()


def test_replace_messages_detaches_dag_message_refs_without_integrity_error(tmp_path):
    db = make_db(tmp_path)
    try:
        msg1 = db.append_message("sess-1", "user", "before")
        msg2 = db.append_message("sess-1", "assistant", "before reply")
        store = ContextDAGStore(db)
        node = store.create_summary_node(
            session_id="sess-1",
            summary_text="old transcript summary",
            source_hash="replace-source",
            prompt_version="p1",
        )
        source = store.link_summary_source(
            session_id="sess-1",
            summary_id=node.id,
            source_type="message_span",
            start_message_id=msg1,
            end_message_id=msg2,
        )
        store.write_active_projection(
            session_id="sess-1",
            engine_version="dag-v1",
            projection=[{"role": "system", "content": "summary"}],
            fresh_tail_start_message_id=msg1,
            latest_raw_message_id=msg2,
        )
        store.write_checkpoint(
            session_id="sess-1",
            last_ingested_message_id=msg2,
            last_projection_message_id=msg2,
            last_anchor_message_id=msg1,
        )
        db._execute_write(
            lambda conn: conn.execute(
                """
                INSERT INTO context_message_parts (
                    message_id, part_index, part_type, content_inline, created_at
                ) VALUES (?, 0, 'text', 'before', 1.0)
                """,
                (msg1,),
            )
        )

        db.replace_messages("sess-1", [{"role": "user", "content": "after"}])

        messages = db.get_messages("sess-1")
        assert [m["content"] for m in messages] == ["after"]
        assert db._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        updated_source = store.get_summary_sources("sess-1", node.id)[0]
        assert updated_source.id == source.id
        assert updated_source.start_message_id is None
        assert updated_source.end_message_id is None
        projection = store.read_active_projection("sess-1", "dag-v1")
        assert projection is not None
        assert projection.fresh_tail_start_message_id is None
        assert projection.latest_raw_message_id is None
        checkpoint = store.read_checkpoint("sess-1")
        assert checkpoint is not None
        assert checkpoint.last_ingested_message_id is None
        assert checkpoint.last_projection_message_id is None
        assert checkpoint.last_anchor_message_id is None
        with db._lock:
            part_count = db._conn.execute("SELECT COUNT(*) FROM context_message_parts").fetchone()[0]
        assert part_count == 0
    finally:
        db.close()
