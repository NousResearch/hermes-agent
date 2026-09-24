"""Differential parity: mid-conversation session branch — SQLite oracle vs PostgreSQL.

The CLI slash-command drives branch publication on both backends (atomic
``branch_session`` on PostgreSQL, create+copy+end on SQLite); the normalized
observations must be identical.  Concurrent-branch rejection and lease gating are
PostgreSQL atomicity guarantees asserted against its focused API.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest

from cli_session_store import PostgreSQLCLISessionStore
from hermes_state import SessionDB
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)

pytestmark = pytest.mark.integration

_FIELDS = ("role", "content", "tool_name", "tool_calls", "tool_call_id", "reasoning",
           "reasoning_details", "codex_reasoning_items", "codex_message_items")


def _normalize(value):
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


def _normalize_rows(rows):
    return [{key: _normalize(row.get(key)) for key in _FIELDS} for row in rows]


def _observe(store, parent, child):
    session = store.get_session(child)
    model_config = session["model_config"]
    if isinstance(model_config, str):
        model_config = json.loads(model_config)
    model_config["_branched_from"] = "PARENT"
    return {
        "parent_end_reason": store.get_session(parent)["end_reason"],
        "child": {
            "parent_session_id": "PARENT",
            "model": session["model"],
            "model_config": model_config,
            "title": session["title"],
        },
        "records": _normalize_rows(store.get_messages_as_conversation(child)),
        "counter": len(store.get_messages_as_conversation(child)),
        "search_visible": child in [row["id"] for row in store.search_sessions(source="cli")],
        "resume": _normalize_rows(store.get_resume_conversations(child)[0]),
    }


def test_branch_mid_conversation_normalized_parity(tmp_path, monkeypatch,
                                                   postgresql_test_target: OwnedPostgreSQLTestTarget):
    from cli import HermesCLI

    sqlite = SessionDB(db_path=tmp_path / "branch-sqlite.db")
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    pg = PostgreSQLCLISessionStore(PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema))
    parent = f"branch-diff-{uuid.uuid4()}"
    history = [
        {"role": "user", "content": "compare input", "timestamp": 100},
        {"role": "assistant", "content": "compare output", "tool_calls": [{"id": "call-1"}],
         "reasoning": "comparison", "reasoning_details": [{"type": "reasoning.text", "text": "detail"}],
         "codex_reasoning_items": [{"id": "rs-1"}], "codex_message_items": [{"id": "msg-1"}], "timestamp": 101},
        {"role": "tool", "content": "compare tool", "tool_name": "local", "tool_call_id": "call-1", "timestamp": 102},
    ]
    monkeypatch.setattr("cli._sync_process_session_id", lambda _sid: None)
    observed, child_ids = {}, {}
    try:
        for name, store in (("sqlite", sqlite), ("postgresql", pg)):
            store.create_session(parent, "cli", model="oracle/model")
            store.set_session_title(parent, "Oracle branch title")
            if name == "postgresql":
                store.append_messages_batch(parent, history)
        for name, store in (("sqlite", sqlite), ("postgresql", pg)):
            shell = HermesCLI.__new__(HermesCLI)
            shell._agent_running = False
            shell._session_db = store
            shell.session_id = parent
            shell.model = "oracle/model"
            shell.max_turns = 17
            shell.reasoning_config = {"effort": "medium"}
            shell.session_start = None
            shell._pending_title = None
            shell._resumed = False
            shell.agent = None
            shell.conversation_history = history
            shell._transfer_session_yolo = MagicMock()
            HermesCLI._handle_branch_command(shell, "/branch Oracle child")
            child_ids[name] = shell.session_id
            observed[name] = _observe(store, parent, shell.session_id)
        # Lease rotation: the freshly branched child accepts a new writer on both backends.
        for name, store in (("sqlite", sqlite), ("postgresql", pg)):
            assert store.try_acquire_session_turn_lease(child_ids[name], "next-writer", ttl_seconds=30) is True
            store.release_session_turn_lease(child_ids[name], "next-writer")
    finally:
        sqlite.close()
        pg.close()
    assert observed["sqlite"] == observed["postgresql"]
    assert observed["postgresql"]["parent_end_reason"] == "branched"
    assert observed["postgresql"]["counter"] == 3


def test_postgresql_branch_rejects_active_lease_and_concurrent_rebranch(
    monkeypatch, postgresql_test_target: OwnedPostgreSQLTestTarget,
):
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    store = PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    parent, child = f"br-{uuid.uuid4()}", f"br-{uuid.uuid4()}"
    try:
        store.ensure_session(parent, source="cli")
        store.append_message(parent, role="user", content="hi")
        assert store.try_acquire_session_turn_lease(parent, "writer", ttl_seconds=30)
        with pytest.raises(RuntimeError, match="active turn lease"):
            store.branch_session(parent_session_id=parent, child_session_id=child, source="cli",
                                 model="m", model_config={"max_iterations": 1}, title="t")
        store.release_session_turn_lease(parent, "writer")
        store.branch_session(parent_session_id=parent, child_session_id=child, source="cli",
                             model="m", model_config={"max_iterations": 1}, title="t")
        assert store.get_session(parent)["end_reason"] == "branched"
        with pytest.raises(ValueError, match="missing or no longer active"):
            store.branch_session(parent_session_id=parent, child_session_id=f"br-{uuid.uuid4()}", source="cli",
                                 model="m", model_config={}, title="t2")
    finally:
        store.close()
