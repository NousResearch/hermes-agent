"""Tests for the ``on_kanban_event_appended`` observer hook (#127497).

Every ``task_events`` INSERT (created/commented/linked/assigned/archived/...
including CLI-originated writes no lifecycle hook covers) fires the hook
AFTER the write transaction commits: once per row in ``id`` order for
batched appends, nothing on rollback, short-circuited on ``has_hook()``,
bounded by ``plugins.hook_callback_timeout``, failures swallowed.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_appends():
    mgr = get_plugin_manager()
    events: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("on_kanban_event_appended", []).append(
        lambda **kw: events.append(kw)
    )
    try:
        yield events
    finally:
        mgr._hooks = saved


def _stored_event_row(event_id: int):
    c2 = sqlite3.connect(kb.kanban_db_path())
    try:
        return c2.execute(
            "SELECT id, task_id, run_id, kind, payload, created_at"
            " FROM task_events WHERE id = ?",
            (event_id,),
        ).fetchone()
    finally:
        c2.close()


def test_create_and_comment_fire_appended_post_commit(kanban_home, captured_appends):
    """create + comment fire once each, post-commit, with the full contract kwargs."""
    committed_at_fire: list = []

    def _prove_committed(**kw):
        row = _stored_event_row(kw["event_id"])
        committed_at_fire.append(tuple(row) if row else None)

    mgr = get_plugin_manager()
    mgr._hooks.setdefault("on_kanban_event_appended", []).append(_prove_committed)

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.add_comment(conn, tid, "alice", "hello") is not None
    finally:
        conn.close()

    kinds = [kw["kind"] for kw in captured_appends]
    assert "created" in kinds
    assert "commented" in kinds
    assert len(committed_at_fire) == len(captured_appends)
    for kw, stored in zip(captured_appends, committed_at_fire):
        assert stored is not None, "hook fired before the event row committed"
        event_id, task_id, run_id, kind, payload_text, created_at = stored
        assert kw["event_id"] == event_id
        assert kw["task_id"] == task_id
        assert kw["run_id"] == run_id
        assert kw["kind"] == kind
        assert kw["payload"] == payload_text, "payload must be the stored JSON string"
        assert kw["created_at"] == created_at
        assert kw["board"] == kb.get_current_board()
        assert "profile_name" in kw
        assert kw["origin"] in ("cli", "tool", "dispatcher", "api")


def test_batched_appends_fire_once_per_event_in_id_order(kanban_home, captured_appends):
    """Two appends in one txn fire twice, in ``id`` order."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        captured_appends.clear()  # create-time bookkeeping is not under test
        with kbc.write_txn(conn):
            kb._append_event(conn, tid, "first", {"n": 1})
            kb._append_event(conn, tid, "second", {"n": 2})
    finally:
        conn.close()

    assert [kw["kind"] for kw in captured_appends] == ["first", "second"]
    ids = [kw["event_id"] for kw in captured_appends]
    assert ids == sorted(ids)
    assert all(kw["task_id"] == tid for kw in captured_appends)


def test_rollback_fires_nothing(kanban_home, captured_appends):
    """A rolled-back txn discards staged notifications."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        captured_appends.clear()
        with pytest.raises(RuntimeError, match="boom"):
            with kbc.write_txn(conn):
                kb._append_event(conn, tid, "doomed", {"n": 1})
                raise RuntimeError("boom")
    finally:
        conn.close()

    assert captured_appends == []
    assert kb._PENDING_EVENT_HOOKS == {}


def test_nested_txn_fires_once_on_outer_commit(kanban_home, captured_appends):
    """An inner savepoint RELEASE is not a commit: nothing fires until the outer commits."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        captured_appends.clear()
        with kbc.write_txn(conn):
            with kbc.write_txn(conn, allow_nested=True):
                kb._append_event(conn, tid, "nested", {"n": 1})
            assert captured_appends == [], "inner RELEASE must not fire the hook"
        assert [kw["kind"] for kw in captured_appends] == ["nested"]
    finally:
        conn.close()


def test_no_subscriber_queues_nothing(kanban_home, monkeypatch):
    """With nothing subscribed, appends skip the queue and never reach invoke_hook."""
    calls: list = []
    monkeypatch.setattr(
        "hermes_cli.lifecycle.invoke_hook", lambda *a, **k: calls.append((a, k)) or []
    )
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.add_comment(conn, tid, "alice", "hello") is not None
    finally:
        conn.close()

    assert kb._PENDING_EVENT_HOOKS == {}
    assert not [c for c in calls if c[0][:1] == ("on_kanban_event_appended",)]


def test_raising_callback_does_not_break_mutation(kanban_home):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("on_kanban_event_appended", []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="t")
            assert kb.add_comment(conn, tid, "alice", "hello") is not None
            assert kb.list_events(conn, tid), "mutation must land despite the crash"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_origin_scopes(kanban_home, captured_appends):
    """Entry layers pin origin; the default is the agent/tool path."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert captured_appends[-1]["origin"] == "tool"
        for origin in ("cli", "tool", "dispatcher", "api"):
            captured_appends.clear()
            with kb.scoped_kanban_event_origin(origin):
                kb.add_comment(conn, tid, "alice", f"from {origin}")
            assert captured_appends, f"no hook fired under origin {origin!r}"
            assert captured_appends[-1]["origin"] == origin
            assert captured_appends[-1]["kind"] == "commented"
    finally:
        conn.close()


def test_slow_callback_bounded_by_hook_timeout(kanban_home, monkeypatch):
    """A hung subscriber is abandoned per plugins.hook_callback_timeout, not joined."""
    monkeypatch.setattr(
        "hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.05
    )
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("on_kanban_event_appended", []).append(
        lambda **kw: time.sleep(5)
    )
    try:
        conn = kbc.connect()
        try:
            start = time.monotonic()
            tid = kb.create_task(conn, title="t")
            elapsed = time.monotonic() - start
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

    assert tid
    assert elapsed < 2.0, f"mutation stalled on a slow hook: {elapsed:.2f}s"
