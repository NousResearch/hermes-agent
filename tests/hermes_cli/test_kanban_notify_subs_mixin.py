"""Regression tests for the notify-subs extraction (godfile wave 1, s5 c9).

``kanban_db``'s notification-subscription functions were moved VERBATIM into
``hermes_cli/notify_subs_mixin.py`` (agreement: move=44) and are re-exported
from ``kanban_db`` so the public API is unchanged. These tests pin the moved
bodies' behavior: the pure metadata/profile-filter helpers and the
cursor/claim semantics of the DB-backed subscription functions.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.notify_subs_mixin import (
    _decode_notify_delivery_metadata,
    _encode_notify_delivery_metadata,
    _notify_profile_filter,
    add_notify_sub,
    advance_notify_cursor,
    claim_unseen_events_for_sub,
    list_notify_subs,
    remove_notify_sub,
    rewind_notify_cursor,
    unseen_events_for_sub,
)

# Minimal schema matching the real kanban tables' columns the moved functions
# touch (see SCHEMA_SQL in kanban_db.py).
NOTIFY_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    chat_type     TEXT,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    notifier_profile TEXT,
    delivery_metadata TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);
"""


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.isolation_level = None  # autocommit; write_txn drives explicit BEGIN IMMEDIATE
    c.executescript(NOTIFY_SCHEMA)
    yield c
    c.close()


def _insert_event(conn, task_id, kind="terminal", payload="{}", created_at=1000):
    cur = conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, kind, payload, created_at),
    )
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_encode_delivery_metadata_filters_nonscalars():
    raw = {
        "anchor": 42, "silent": True, "note": "x", "flag": 1.5,
        "bad": {"nested": 1}, "none": None,
    }
    encoded = _encode_notify_delivery_metadata(raw)
    assert json.loads(encoded) == {"anchor": 42, "flag": 1.5, "note": "x", "silent": True}


def test_encode_delivery_metadata_edge_cases():
    assert _encode_notify_delivery_metadata(None) is None
    assert _encode_notify_delivery_metadata({}) is None
    assert _encode_notify_delivery_metadata({"only_none": None}) is None
    assert _encode_notify_delivery_metadata("not-a-mapping") is None


def test_decode_delivery_metadata_roundtrip():
    encoded = _encode_notify_delivery_metadata({"a": 1, "b": "x"})
    assert _decode_notify_delivery_metadata(encoded) == {"a": 1, "b": "x"}
    assert _decode_notify_delivery_metadata(None) == {}
    assert _decode_notify_delivery_metadata("") == {}
    assert _decode_notify_delivery_metadata("not json {") == {}
    assert _decode_notify_delivery_metadata(json.dumps(["a-list"])) == {}
    assert _decode_notify_delivery_metadata({"raw": 1}) == {"raw": 1}
    assert _decode_notify_delivery_metadata(json.dumps({"k": "v", "drop": ["x"]})) == {"k": "v"}


def test_notify_profile_filter():
    assert _notify_profile_filter(None, include_unowned=False) == ("", [])
    assert _notify_profile_filter([], include_unowned=False) == ("0", [])
    where, params = _notify_profile_filter(["b", "a", "a"], include_unowned=False)
    assert where == "(notifier_profile IN (?,?))"
    assert params == ["a", "b"]
    where, params = _notify_profile_filter(["a"], include_unowned=True)
    assert "(notifier_profile IS NULL OR notifier_profile = '')" in where
    assert params == ["a"]


# ---------------------------------------------------------------------------
# DB-backed subscription functions
# ---------------------------------------------------------------------------


def test_add_and_list_notify_subs(conn):
    add_notify_sub(conn, task_id="t_1", platform="telegram", chat_id="c1")
    add_notify_sub(conn, task_id="t_1", platform="telegram", chat_id="c1")  # idempotent
    add_notify_sub(
        conn, task_id="t_1", platform="telegram", chat_id="c2",
        chat_type="group", thread_id="th", notifier_profile="p1",
        delivery_metadata={"anchor": "x"},
    )
    subs = list_notify_subs(conn, "t_1")
    assert len(subs) == 2
    by_chat = {s["chat_id"]: s for s in subs}
    assert by_chat["c1"]["last_event_id"] == 0
    assert by_chat["c2"]["thread_id"] == "th"
    assert by_chat["c2"]["chat_type"] == "group"
    assert by_chat["c2"]["notifier_profile"] == "p1"
    assert by_chat["c2"]["delivery_metadata"] == {"anchor": "x"}


def test_list_notify_subs_profile_filters(conn):
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a", notifier_profile="alice")
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="b")
    assert len(list_notify_subs(conn, "t_1", notifier_profiles=["alice"])) == 1
    assert len(list_notify_subs(conn, "t_1", notifier_profiles=["alice"], include_unowned=True)) == 2
    assert len(list_notify_subs(conn, "t_1", notifier_profiles=["nobody"])) == 0


def test_remove_notify_sub(conn):
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert remove_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a") is True
    assert remove_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a") is False
    assert list_notify_subs(conn, "t_1") == []


def test_new_sub_snaps_cursor_to_current_max_event(conn):
    _insert_event(conn, "t_1", created_at=1000)
    e2 = _insert_event(conn, "t_1", created_at=1001)
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    cursor, events = unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert cursor == e2
    assert events == []
    # a later event becomes visible
    e3 = _insert_event(conn, "t_1", created_at=1002)
    cursor, events = unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert cursor == e3
    assert [ev.id for ev in events] == [e3]
    assert isinstance(events[0], kb.Event)
    assert events[0].kind == "terminal"
    assert events[0].payload == {}


def test_claim_advance_rewind_cursor_cas(conn):
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    e3 = _insert_event(conn, "t_1", created_at=1002)
    old, new, claimed = claim_unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert (old, new) == (0, e3)
    assert [ev.id for ev in claimed] == [e3]
    # cursor advanced -> nothing unseen anymore
    cursor, events = unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert cursor == e3 and events == []
    # rewind restores the old cursor
    assert rewind_notify_cursor(
        conn, task_id="t_1", platform="tg", chat_id="a",
        claimed_cursor=e3, old_cursor=old,
    ) is True
    cursor, events = unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert [ev.id for ev in events] == [e3]
    # CAS guard: a stale claimed cursor (row already advanced past it) is a no-op
    advance_notify_cursor(conn, task_id="t_1", platform="tg", chat_id="a", new_cursor=e3)
    assert rewind_notify_cursor(
        conn, task_id="t_1", platform="tg", chat_id="a",
        claimed_cursor=0, old_cursor=0,
    ) is False
    assert advance_notify_cursor(
        conn, task_id="t_1", platform="tg", chat_id="a", new_cursor=e3,
    ) is None
    cursor, events = unseen_events_for_sub(conn, task_id="t_1", platform="tg", chat_id="a")
    assert events == []


def test_claim_missing_subscription_is_empty(conn):
    assert claim_unseen_events_for_sub(
        conn, task_id="ghost", platform="tg", chat_id="a",
    ) == (0, 0, [])


def test_count_notify_subs_readonly_probe(tmp_path):
    path = tmp_path / "board.db"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(NOTIFY_SCHEMA)
    add_notify_sub(conn, task_id="t_1", platform="Telegram", chat_id="a")
    add_notify_sub(conn, task_id="t_1", platform="tg", chat_id="b", notifier_profile="p1")
    conn.commit()
    conn.close()
    assert kb.count_notify_subs(db_path=path) == 2
    assert kb.count_notify_subs(db_path=path, platform="telegram") == 1  # case-insensitive
    assert kb.count_notify_subs(db_path=path, platform="tg") == 1
    assert kb.count_notify_subs(db_path=path, platform="nope") == 0
    assert kb.count_notify_subs(db_path=path, notifier_profiles=["p1"]) == 1
    assert kb.count_notify_subs(db_path=path, notifier_profiles=["p1"], include_unowned=True) == 2
    assert kb.count_notify_subs(db_path=path, chat_id="zzz") == 0
    assert kb.count_notify_subs(db_path=tmp_path / "missing.db") == 0


def test_public_api_reexported_from_kanban_db():
    for name in (
        "add_notify_sub", "advance_notify_cursor", "claim_unseen_events_for_sub",
        "count_notify_subs", "list_notify_subs", "remove_notify_sub",
        "rewind_notify_cursor", "unseen_events_for_sub",
    ):
        assert callable(getattr(kb, name)), name
    # the re-export must be the very same function objects the mixin defines
    assert kb.add_notify_sub is add_notify_sub
    assert kb.rewind_notify_cursor is rewind_notify_cursor
