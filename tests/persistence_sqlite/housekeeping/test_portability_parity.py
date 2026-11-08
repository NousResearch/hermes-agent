"""The SQLite portability verbs against the legacy store: export, lineage export, export_all, import and foreign import.

Each case seeds one ``state.db`` with the legacy ``SessionDB``, copies it, runs the legacy method on the copy and the
backend verb on the original, and compares the returned values (the contract records projected to the legacy dicts)
and the resulting table contents. The cases that read the backend alone seed the same sessions through the storage
interface.
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import pytest

import agent.message_metadata
from hermes_constants import get_hermes_home
from persistence import Conflict
from persistence.conversation.sessions import SessionLineageKind
from persistence.housekeeping.maintenance import ForeignOrigin, ForeignTurn
from persistence.sqlite import _holder
from persistence.sqlite.conversation._store import rules as store_rules
from persistence.sqlite.housekeeping import _portability
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

TABLES = ("sessions", "messages", "system_prompts", "state_meta")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def _dump(path: Path, skip: dict[str, tuple[str, ...]] | None = None) -> dict[str, list[tuple]]:
    """Every row of the compared tables, ordered by their key, without the ``skip`` columns."""
    out = {}
    with sqlite3.connect(path) as conn:
        for table in TABLES:
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")
                       if row[1] not in (skip or {}).get(table, ())]
            out[table] = conn.execute(f"SELECT {', '.join(columns)} FROM {table} ORDER BY 1, 2").fetchall()
    return out


def _session_dict(export) -> dict:
    return {**export.session.as_dict(), "messages": [m.as_wire_row() for m in export.messages]}


def _legacy(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "timings"}


def _items(value):
    """Dicts as ordered item lists, so key order is compared too."""
    if isinstance(value, dict):
        return [(key, _items(item)) for key, item in value.items()]
    if isinstance(value, list):
        return [_items(item) for item in value]
    return value


def _seed(path: Path) -> None:
    """Sessions with live, rewound, compaction-archived and summary rows, a compression lineage, a fork and two
    sources, with fixed clocks so recency orders them."""
    from hermes_state import SessionDB

    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", "cli", model="m", model_config={"x": 1}, system_prompt="sys prompt", cwd="/w")
        db.set_session_title("s1", "One")
        ids = [
            db.append_message("s1", "user", "hi", timestamp=1000.0),
            db.append_message("s1", "assistant", "calling", timestamp=1001.0, reasoning="think", tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]),
            db.append_message("s1", "tool", "result", tool_call_id="c1", tool_name="t", timestamp=1002.0),
            db.append_message("s1", "assistant", "done", timestamp=1003.0, finish_reason="stop",
                              display_metadata={"k": "v"}),
            db.append_message("s1", "user", [{"type": "text", "text": "multi"}], timestamp=1004.0),
            db.append_message("s1", "assistant", "summary", timestamp=1005.0, _compressed_summary=True),
            db.append_message("s1", "user", "again", timestamp=1006.0, observed=True),
        ]
        db.create_session("root", "cli", model="m")
        db.append_message("root", "user", "root turn", timestamp=900.0)
        db.end_session("root", "compression")
        db.create_session("child", "cli", parent_session_id="root")
        db.append_message("child", "user", "child turn", timestamp=950.0)
        db.create_session("fork", "cli", parent_session_id="s1", model_config={"_branched_from": "s1"})
        db.append_message("fork", "user", "fork turn", timestamp=1100.0)
        db.create_session("tg", "telegram", user_id="u1")
        db.append_message("tg", "user", "from telegram", timestamp=1200.0)
        db.end_session("tg", "user_exit")
    finally:
        db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (ids[6],))
        conn.executemany("UPDATE messages SET active = 0, compacted = 1 WHERE id = ?", [(ids[0],), (ids[1],)])
        for sid, started in (("s1", 100.0), ("root", 90.0), ("child", 95.0), ("fork", 110.0), ("tg", 120.0)):
            conn.execute("UPDATE sessions SET started_at = ?, last_activity_at = NULL WHERE id = ?", (started, sid))


def _seed_store(store) -> None:
    """The sessions and rows of ``_seed``, written through the storage interface."""
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
    """The backend handle over the sessions of ``_seed``, seeded through the storage interface."""
    handle = backend.open(scope_a)
    _seed_store(handle)
    return handle


@pytest.fixture
def stores(backend, scope_a, scope_b, legacy_store):
    """``(store, legacy_path)``: the backend handle over the seeded store and the legacy store's copy of it."""
    _seed(store_for(scope_a))
    _copy(store_for(scope_a), store_for(scope_b))
    return backend.open(scope_a), store_for(scope_b)


@pytest.fixture
def legacy_db(legacy_store):
    from hermes_state import SessionDB

    opened = []

    def open_db(path: Path):
        opened.append(SessionDB(db_path=path))
        return opened[-1]

    yield open_db
    for db in opened:
        db.close()


FLAGS = [(False, False), (True, False), (False, True), (True, True)]


@pytest.mark.parametrize("include_compacted,include_inactive", FLAGS)
@pytest.mark.parametrize("session_id", ["s1", "root", "child", "fork", "tg"])
def test_export_session_matches_legacy(stores, legacy_db, scope_a, session_id, include_compacted, include_inactive):
    store, legacy_path = stores
    expected = legacy_db(legacy_path).export_session(session_id, include_compacted, include_inactive)
    got = store.maintenance.export_session(session_id, include_compacted, include_inactive)
    assert _items(_session_dict(got)) == _items(_legacy(expected))
    # The display read backfills display identities in both stores alike.
    assert _dump(store_for(scope_a)) == _dump(legacy_path)


def test_export_session_rows_carry_every_visibility(store):
    rows = store.maintenance.export_session("s1", include_inactive=True).messages
    assert [(row.as_wire_row()["active"], row.as_wire_row()["compacted"]) for row in rows] == [
        (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0)]
    assert [row.compressed_summary for row in rows] == [False] * 5 + [True, False]
    assert [row.content for row in store.maintenance.export_session("s1").messages] == [
        "result", "done", [{"type": "text", "text": "multi"}], "summary"]


def test_export_session_unknown_id_is_none(stores, legacy_db):
    store, legacy_path = stores
    assert legacy_db(legacy_path).export_session("missing") is None
    assert store.maintenance.export_session("missing") is None
    assert store.maintenance.export_session_lineage("missing") is None


def test_export_session_flushes_queued_usage(stores, legacy_db, monkeypatch):
    store, legacy_path = stores

    def apply(holder, session_id, kwargs):
        holder._write_sql("UPDATE sessions SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ? "
                          "WHERE id = ?", (kwargs.get("input_tokens", 0), kwargs.get("output_tokens", 0), session_id))

    monkeypatch.setattr(_holder, "_token_delta_applier", apply)
    legacy = legacy_db(legacy_path)
    legacy.queue_token_counts("s1", input_tokens=7, output_tokens=3)
    store.holder().queue_token_counts("s1", input_tokens=7, output_tokens=3)
    expected = legacy.export_session("s1")
    got = store.maintenance.export_session("s1")
    assert (expected["input_tokens"], expected["output_tokens"]) == (7, 3)
    assert (got.session.input_tokens, got.session.output_tokens) == (7, 3)


@pytest.mark.parametrize("include_compacted,include_inactive", FLAGS)
@pytest.mark.parametrize("session_id", ["root", "child", "fork", "s1"])
def test_export_session_lineage_matches_legacy(stores, legacy_db, session_id, include_compacted, include_inactive):
    store, legacy_path = stores
    expected = legacy_db(legacy_path).export_session_lineage(session_id, include_compacted, include_inactive)
    got = store.maintenance.export_session_lineage(session_id, include_compacted, include_inactive)
    assert list(got.lineage_session_ids) == expected["lineage_session_ids"]
    assert [_items(_session_dict(seg)) for seg in got.segments] == [
        _items(_legacy(seg)) for seg in expected["segments"]]
    assert got.message_count == expected["message_count"]
    assert [m.as_wire_row() for m in got.messages] == expected["messages"]


def test_export_session_lineage_spans_compression_only(store):
    assert store.maintenance.export_session_lineage("root").lineage_session_ids == ("root", "child")
    assert store.maintenance.export_session_lineage("child").lineage_session_ids == ("root", "child")
    assert store.maintenance.export_session_lineage("fork").lineage_session_ids == ("fork",)


@pytest.mark.parametrize("include_compacted,include_inactive", FLAGS)
@pytest.mark.parametrize("source", [None, "cli", "telegram", "absent"])
def test_export_all_matches_legacy(stores, legacy_db, scope_a, source, include_compacted, include_inactive):
    store, legacy_path = stores
    expected = legacy_db(legacy_path).export_all(source, include_compacted, include_inactive)
    got = store.maintenance.export_all(source, include_compacted, include_inactive)
    assert [_items(_session_dict(export)) for export in got] == [_items(_legacy(payload)) for payload in expected]
    assert _dump(store_for(scope_a)) == _dump(legacy_path)


def test_export_all_orders_by_recency_and_filters_source(store):
    assert [e.session.id for e in store.maintenance.export_all()] == ["tg", "fork", "s1", "child", "root"]
    assert [e.session.id for e in store.maintenance.export_all("cli")] == ["fork", "s1", "child", "root"]
    assert all(e.session.last_active is not None for e in store.maintenance.export_all())
    assert store.maintenance.export_session("s1").session.last_active is None


def _result(result) -> dict:
    return {"ok": result.ok, "imported": result.imported, "skipped": result.skipped, "detached": result.detached,
            "imported_ids": list(result.imported_ids), "skipped_ids": list(result.skipped_ids),
            "errors": list(result.errors)}


def _legacy_result(result: dict) -> dict:
    return {"imported_ids": [], "skipped_ids": [], **result}


@pytest.fixture
def targets(backend, scope_a, scope_b, legacy_store):
    """``(store, legacy_path)``: a store holding only session ``p0`` and the retired id ``gone``, and its copy."""
    from hermes_state import SessionDB
    from hermes_state_mutation_retirement import RETIRED_PREFIX

    path = store_for(scope_a)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=path)
    try:
        db.create_session("p0", "cli")
    finally:
        db.close()
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, '1')", (RETIRED_PREFIX + "gone",))
    _copy(path, store_for(scope_b))
    return backend.open(scope_a), store_for(scope_b)


@pytest.fixture
def source_payload(legacy_db):
    """The seeded sessions as the legacy ``export_all(include_inactive=True)`` backup emits them."""
    path = get_hermes_home() / "source" / "state.db"
    _seed(path)
    return legacy_db(path).export_all(include_inactive=True)


def _import_both(store, legacy, payload):
    expected = legacy.import_sessions(json.loads(json.dumps(payload)))
    got = store.maintenance.import_sessions(json.loads(json.dumps(payload)))
    return got, expected


def test_import_round_trip_matches_legacy(targets, legacy_db, source_payload, scope_a):
    store, legacy_path = targets
    legacy = legacy_db(legacy_path)
    got, expected = _import_both(store, legacy, source_payload)
    assert _result(got) == _legacy_result(expected)
    assert got.imported == 5 and got.detached == 0
    assert _dump(store_for(scope_a)) == _dump(legacy_path)
    reexport = store.maintenance.export_all(include_inactive=True)
    assert [_items(_session_dict(e)) for e in reexport] == [
        _items(_legacy(p)) for p in legacy.export_all(include_inactive=True)]
    # Archived rows come back archived with their flags; the live rows alone are counted.
    s1 = store.maintenance.export_session("s1", include_inactive=True)
    assert [(m.as_wire_row()["active"], m.as_wire_row()["compacted"]) for m in s1.messages] == [
        (0, 1), (0, 1), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0)]
    assert (s1.session.message_count, s1.session.tool_call_count) == (4, 0)
    assert [m.message_uid for m in s1.messages] == [m["message_uid"] for m in source_payload[2]["messages"]]


def test_import_skips_existing_ids_and_resets_runtime_state(targets, legacy_db, scope_a):
    store, legacy_path = targets
    payload = [{"id": "p0", "title": "ignored"},
               {"id": "x", "auto_archived": 1, "archived": 0, "pinned": "1", "hidden": None, "session_key": "k",
                "last_activity_at": 5.0, "rewind_count": 3, "handoff_state": "pending", "ended_at": "inf",
                "input_tokens": "12x", "output_tokens": "4", "started_at": 50.0},
               {"id": "y", "archived": True, "auto_archived": True, "started_at": 60.0}]
    got, expected = _import_both(store, legacy_db(legacy_path), payload)
    assert _result(got) == _legacy_result(expected)
    assert got.skipped_ids == ("p0",) and got.imported_ids == ("x", "y")
    assert _dump(store_for(scope_a)) == _dump(legacy_path)


def test_import_parents_attach_detach_and_break_cycles(targets, legacy_db, scope_a):
    store, legacy_path = targets
    payload = [{"id": "a", "parent_session_id": "b"}, {"id": "b", "parent_session_id": "a"},
               {"id": "c", "parent_session_id": "nowhere"}, {"id": "d", "parent_session_id": "p0"},
               {"id": "e", "parent_session_id": "d"}]
    payload = [{**entry, "started_at": 10.0 + index} for index, entry in enumerate(payload)]
    got, expected = _import_both(store, legacy_db(legacy_path), payload)
    assert _result(got) == _legacy_result(expected)
    assert got.detached == 2
    assert _dump(store_for(scope_a)) == _dump(legacy_path)


BIG = "x" * (5 * 1024 * 1024)
REFUSED = {
    "not_object": ["x"],
    "no_id": [{"messages": []}],
    "blank_id": [{"id": "  "}],
    "duplicate": [{"id": "a"}, {"id": "a"}],
    "messages_not_list": [{"id": "a", "messages": "x"}],
    "message_not_object": [{"id": "a", "messages": [1]}],
    "too_many_messages": [{"id": "a", "messages": [{"role": "user"}] * 10_001}],
    "session_too_big": [{"id": "a", "messages": [{"role": "user", "content": BIG}]}],
    "total_too_big": [{"id": f"t{n}", "messages": [{"role": "user", "content": BIG[:4_500_000]}]} for n in range(6)],
    "total_messages": [{"id": f"m{n}", "messages": [{"role": "user"}] * 9_000} for n in range(6)],
    "empty_role": [{"id": "a", "messages": [{"role": ""}]}],
    "text_field": [{"id": "a", "title": 5}],
    "message_text_field": [{"id": "a", "messages": [{"role": "user", "tool_name": 3}]}],
    "model_config_json": [{"id": "a", "model_config": "{bad"}],
    "model_config_list": [{"id": "a", "model_config": "[1]"}],
    "model_config_non_finite": [{"id": "a", "model_config": {"x": float("inf")}}],
    "non_finite_column": [{"id": "a", "ended_at": float("inf")}],
    "int_range": [{"id": "a", "input_tokens": 1 << 64}],
    "bad_started_at": [{"id": "a", "started_at": "yesterday"}],
    "bad_timestamp": [{"id": "a", "messages": [{"role": "user", "timestamp": "soon"}]}],
    "token_count": [{"id": "a", "messages": [{"role": "user", "token_count": "x"}]}],
    "content_bytes": [{"id": "a", "messages": [{"role": "user", "content": 1 << 64}]}],
    "sidecar_non_finite": [{"id": "a", "messages": [{"role": "assistant", "reasoning_details": "[1e999]"}]}],
    "one_bad_entry": [{"id": "fine"}, {"id": "bad", "title": 1}],
}


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_import_refused_payload_matches_legacy_and_writes_nothing(targets, legacy_db, scope_a, case):
    store, legacy_path = targets
    before = _dump(store_for(scope_a))
    legacy = legacy_db(legacy_path)
    expected = legacy.import_sessions(list(REFUSED[case]))
    got = store.maintenance.import_sessions(list(REFUSED[case]))
    assert _result(got) == _legacy_result(expected)
    assert got.ok is False and got.errors
    assert _dump(store_for(scope_a)) == before == _dump(legacy_path)


def test_import_refuses_a_deleted_id_by_name(targets, legacy_db, scope_a):
    store, legacy_path = targets
    before = _dump(store_for(scope_a))
    got, expected = _import_both(store, legacy_db(legacy_path), [{"id": "new"}, {"id": "gone"}])
    assert _result(got) == _legacy_result(expected)
    assert got.errors == ("session gone was deleted here; a deleted session is not restored",)
    assert _dump(store_for(scope_a)) == before == _dump(legacy_path)


@pytest.mark.parametrize("payload", [[{"id": f"s{n}"} for n in range(501)], {"id": "a"}])
def test_import_over_the_limit_or_not_a_list_raises(targets, legacy_db, payload):
    store, legacy_path = targets
    with pytest.raises(ValueError) as legacy_error:
        legacy_db(legacy_path).import_sessions(payload)
    with pytest.raises(ValueError) as error:
        store.maintenance.import_sessions(payload)
    assert str(error.value) == str(legacy_error.value)


# The clock stamps of a foreign import (no payload timestamps) and the display identity that hashes them.
VOLATILE = {"sessions": ("started_at",), "messages": ("timestamp", "display_identity")}
TURNS = [ForeignTurn("user", "hello"), ForeignTurn("assistant", "hi there")]
CODEX = ForeignOrigin(tool="codex", path="/h/codex.jsonl", foreign_session_id="f-1")
CLAUDE = ForeignOrigin(tool="claude", path="/h/claude.jsonl")


def _mint(monkeypatch, session_id: str) -> None:
    """Both stores mint ``session_id`` next, and message uids from one fresh counter."""
    import hermes_state_portability

    counter = itertools.count()
    monkeypatch.setattr(agent.message_metadata, "mint_uid", lambda: f"uid-{next(counter)}")
    monkeypatch.setattr(store_rules, "mint_uid", lambda: f"uid-{next(counter)}")
    monkeypatch.setattr(hermes_state_portability, "new_session_id", lambda **_: session_id)
    monkeypatch.setattr(_portability, "new_session_id", lambda **_: session_id)


def _foreign_both(monkeypatch, store, legacy, session_id, origin, title, cwd=None, profile="work"):
    _mint(monkeypatch, session_id)
    expected = legacy.import_foreign_history(origin.as_dict(), [{"role": t.role, "content": t.content} for t in TURNS],
                                             title=title, cwd=cwd, profile=profile)
    _mint(monkeypatch, session_id)
    got = store.maintenance.import_foreign_history(origin, TURNS, title=title, cwd=cwd, profile=profile)
    assert {"session_id": got.session_id, "already_imported": got.already_imported} == expected
    return got


def test_foreign_import_mints_suffixes_and_adopts_like_legacy(targets, legacy_db, scope_a, monkeypatch):
    store, legacy_path = targets
    for path in (store_for(scope_a), legacy_path):
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE sessions SET title = 'Taken' WHERE id = 'p0'")
    legacy = legacy_db(legacy_path)
    first = _foreign_both(monkeypatch, store, legacy, "20260101_000000_aaaaaaaaaaaa", CODEX, "Fresh", cwd="/w")
    second = _foreign_both(monkeypatch, store, legacy, "20260101_000001_bbbbbbbbbbbb", CLAUDE, "Taken")
    again = _foreign_both(monkeypatch, store, legacy, "20260101_000002_cccccccccccc", CODEX, "Other")
    assert (first.already_imported, second.already_imported, again.already_imported) == (False, False, True)
    assert again.session_id == first.session_id
    assert _dump(store_for(scope_a), VOLATILE) == _dump(legacy_path, VOLATILE)
    row = store.maintenance.export_session(second.session_id)
    assert (row.session.title, row.session.source, row.session.profile_name) == (
        "Taken (bbbbbbbbbbbb)", "claude", "work")
    assert json.loads(row.session.origin_json) == {"imported_from": CLAUDE.as_dict()}
    assert [(m.role, m.content) for m in row.messages] == [("user", "hello"), ("assistant", "hi there")]
    assert (row.session.message_count, row.session.tool_call_count) == (2, 0)


@pytest.mark.parametrize("probe", [
    CODEX, ForeignOrigin(tool="codex", path="/elsewhere", foreign_session_id="f-1"),
    ForeignOrigin(tool="codex", path="/h/codex.jsonl", foreign_session_id="f-2"), CLAUDE,
    ForeignOrigin(tool="claude", path="/h/other.jsonl"), ForeignOrigin(tool="other", path="/h/claude.jsonl"),
])
def test_find_foreign_import_matches_legacy(targets, legacy_db, monkeypatch, probe):
    store, legacy_path = targets
    legacy = legacy_db(legacy_path)
    _foreign_both(monkeypatch, store, legacy, "20260101_000000_aaaaaaaaaaaa", CODEX, "One")
    _foreign_both(monkeypatch, store, legacy, "20260101_000001_bbbbbbbbbbbb", CLAUDE, "Two")
    assert store.maintenance.find_foreign_import(probe) == legacy.find_foreign_import(probe.as_dict())


def test_foreign_import_of_a_taken_minted_id_is_a_conflict(targets, legacy_db, scope_a, monkeypatch):
    store, legacy_path = targets
    before = _dump(store_for(scope_a))
    _mint(monkeypatch, "p0")
    with pytest.raises(sqlite3.IntegrityError):
        legacy_db(legacy_path).import_foreign_history(CODEX.as_dict(), [], title="T", cwd=None, profile="work")
    with pytest.raises(Conflict) as error:
        store.maintenance.import_foreign_history(CODEX, [], title="T", cwd=None, profile="work")
    assert (error.value.entity, error.value.entity_id) == ("session", "p0")
    assert _dump(store_for(scope_a)) == before == _dump(legacy_path)


def test_foreign_import_refused_payload_raises_before_any_write(targets, legacy_db, scope_a, monkeypatch):
    store, legacy_path = targets
    before = _dump(store_for(scope_a))
    _mint(monkeypatch, "20260101_000000_aaaaaaaaaaaa")
    turns = [ForeignTurn("user", BIG)]
    with pytest.raises(ValueError) as legacy_error:
        legacy_db(legacy_path).import_foreign_history(CODEX.as_dict(), [{"role": "user", "content": BIG}], title="T",
                                                      cwd=None, profile="work")
    with pytest.raises(ValueError) as error:
        store.maintenance.import_foreign_history(CODEX, turns, title="T", cwd=None, profile="work")
    assert str(error.value) == str(legacy_error.value) == "session exceeds the import size limit"
    assert _dump(store_for(scope_a)) == before == _dump(legacy_path)
