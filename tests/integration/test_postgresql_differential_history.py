"""Differential parity: session list / stats / export — SQLite oracle vs PostgreSQL.

Identical activity on both backends must produce the same listing order, row
counts, token accounting (source preserved), and structurally-equal exports
(after normalizing ids and timestamps).
"""
from __future__ import annotations

import time
import uuid

import pytest

from cli_session_store import PostgreSQLCLISessionStore
from hermes_state import SessionDB
from state_store import PostgreSQLStateStoreConfig
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
        store = PostgreSQLCLISessionStore(
            PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema))
    try:
        yield request.param, store
    finally:
        store.close()


def _seed_activity(store, session_id: str, title: str) -> None:
    store.ensure_session(session_id, source="cli")
    store.append_message(session_id, role="user", content=f"{title}-ask")
    store.append_message(session_id, role="assistant", content=f"{title}-answer")
    store.queue_token_counts(session_id, input_tokens=7, output_tokens=3, model="test/model", source="cli")
    store.flush_token_counts()


def test_history_listing_order_and_row_counts(backend):
    kind, store = backend
    ids = [f"hist-{i}-{uuid.uuid4()}" for i in range(3)]
    for i, sid in enumerate(ids):
        _seed_activity(store, sid, f"session-{i}")
        store.set_session_title(sid, f"session-{i}")
        time.sleep(0.05)

    rows = store.list_sessions_rich(source="cli", limit=10)
    assert [row["id"] for row in rows] == list(reversed(ids))  # newest first, deterministic creation order
    assert len(rows) == 3
    assert store.session_count() == 3
    for row in rows:
        assert row["message_count"] == 2


def test_history_token_accounting_source_preserved(backend):
    kind, store = backend
    sid = f"hist-tok-{uuid.uuid4()}"
    _seed_activity(store, sid, "tok")
    session = store.get_session(sid)
    assert session["source"] == "cli"
    if kind == "sqlite":
        assert store.usage_totals()["tokens"] == 10
    else:
        from agent.insights import InsightsEngine
        report = InsightsEngine(store).generate(days=1, source="cli")
        assert report["overview"]["total_tokens"] == 10


def test_history_export_structural_equality(backend):
    kind, store = backend
    sid = f"hist-exp-{uuid.uuid4()}"
    _seed_activity(store, sid, "export")
    exported = store.export_session(sid)
    assert exported is not None
    assert exported["id"] == sid
    assert exported["message_count"] == 2
    normalized = [{"role": m["role"], "content": m["content"]} for m in exported["messages"]]
    assert normalized == [
        {"role": "user", "content": "export-ask"},
        {"role": "assistant", "content": "export-answer"},
    ]
