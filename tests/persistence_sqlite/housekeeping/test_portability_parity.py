"""The SQLite portability exports on sessions seeded through the storage interface: a session export carries every message visibility, a lineage export spans compression only, and export_all orders by recency and filters by source."""

from __future__ import annotations

import pytest

from persistence.conversation.sessions import SessionLineageKind
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session


def _seed_store(store) -> None:
    """Sessions with live, rewound, compaction-archived and summary rows, a compression lineage, a fork and two
    sources, with fixed clocks so recency orders them, written through the storage interface."""
    seed_session(store, "s1", "cli", model="m", model_config={"x": 1}, system_prompt="sys prompt", cwd="/w", title="One")
    ids = seed_messages(
        store, "s1",
        {"role": "user", "content": "hi", "timestamp": 1000.0},
        {"role": "assistant", "content": "calling", "timestamp": 1001.0, "reasoning": "think", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "c1", "tool_name": "t", "timestamp": 1002.0},
        {"role": "assistant", "content": "done", "timestamp": 1003.0, "finish_reason": "stop",
         "display_metadata": {"k": "v"}},
        {"role": "user", "content": [{"type": "text", "text": "multi"}], "timestamp": 1004.0},
        {"role": "assistant", "content": "summary", "timestamp": 1005.0, "_compressed_summary": True},
        {"role": "user", "content": "again", "timestamp": 1006.0, "observed": True},
    )
    seed_session(store, "root", "cli", model="m")
    seed_messages(store, "root", {"role": "user", "content": "root turn", "timestamp": 900.0})
    store.sessions.end("root", "compression")
    seed_session(store, "child", "cli", parent_session_id="root")
    seed_messages(store, "child", {"role": "user", "content": "child turn", "timestamp": 950.0})
    seed_session(store, "fork", "cli", parent_session_id="s1", lineage_kind=SessionLineageKind.BRANCH)
    seed_messages(store, "fork", {"role": "user", "content": "fork turn", "timestamp": 1100.0})
    seed_session(store, "tg", "telegram", user_id="u1")
    seed_messages(store, "tg", {"role": "user", "content": "from telegram", "timestamp": 1200.0})
    store.sessions.end("tg", "user_exit")

    def visibility_and_clocks(conn):
        conn.execute("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (ids[6],))
        conn.executemany("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", [(ids[0],), (ids[1],)])
        for sid, started in (("s1", 100.0), ("root", 90.0), ("child", 95.0), ("fork", 110.0), ("tg", 120.0)):
            conn.execute("UPDATE sessions SET started_at = ?, last_activity_at = NULL WHERE id = ?", (started, sid))
    raw_holder(store)._execute_write(visibility_and_clocks)


@pytest.fixture
def store(backend, scope_a):
    """The backend handle over the sessions of ``_seed_store``."""
    handle = backend.open(scope_a)
    _seed_store(handle)
    return handle


def test_export_session_rows_carry_every_visibility(store):
    rows = store.maintenance.export_session("s1", include_inactive=True).messages
    assert [(row.as_wire_row()["active"], row.as_wire_row()["compacted"]) for row in rows] == [
        (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0)]
    assert [row.compressed_summary for row in rows] == [False] * 5 + [True, False]
    assert [row.content for row in store.maintenance.export_session("s1").messages] == [
        "result", "done", [{"type": "text", "text": "multi"}], "summary"]


def test_export_session_lineage_spans_compression_only(store):
    assert store.maintenance.export_session_lineage("root").lineage_session_ids == ("root", "child")
    assert store.maintenance.export_session_lineage("child").lineage_session_ids == ("root", "child")
    assert store.maintenance.export_session_lineage("fork").lineage_session_ids == ("fork",)


def test_export_all_orders_by_recency_and_filters_source(store):
    assert [e.session.id for e in store.maintenance.export_all()] == ["tg", "fork", "s1", "child", "root"]
    assert [e.session.id for e in store.maintenance.export_all("cli")] == ["fork", "s1", "child", "root"]
    assert all(e.session.last_active is not None for e in store.maintenance.export_all())
    assert store.maintenance.export_session("s1").session.last_active is None
