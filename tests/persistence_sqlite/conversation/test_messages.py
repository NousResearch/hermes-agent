"""The transcript projections of the SQLite backend, byte for byte against the legacy store.

Each case replays one seeding script onto the backend and onto a legacy ``SessionDB`` on its own file. The replay projection must ``json.dumps`` identical to ``get_messages_as_conversation`` (the prompt-cache byte-stability requirement in ``agent/AGENTS.md``) and every transcript page identical to ``get_messages``, key order and values included. Message and tool-call uids are minted at random per store, so both sides name each uid by its order of first appearance before the comparison.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterable

import pytest

from persistence.conversation.messages import ConversationRequest, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession
from persistence.sqlite import SQLiteBackend
from tests.contracts.conversation.test_messages import (
    CHILD, SESSION, Script, dumps, lineage_script, new, projection_script, ts,
)
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

_MINTED_UID = re.compile(r"\b[0-9a-f]{32}\b")


def same_uids(text: str) -> str:
    """``text`` with each minted uid replaced by its order of first appearance."""
    names: dict[str, str] = {}
    return _MINTED_UID.sub(lambda match: names.setdefault(match[0], f"uid-{len(names)}"), text)


def _legacy_session(db, ids: list[int], sid, source, parent, branch, key) -> None:
    db.create_session(sid, source, parent_session_id=parent,
                      model_config={"_branched_from": parent} if branch else None, session_key=key)


def _legacy_msg(db, ids: list[int], sid, role, content, i, kwargs) -> None:
    options = {k: list(v) if k == "tool_calls" else v for k, v in kwargs.items()}
    ids.append(db.append_message(sid, role, content=content, timestamp=ts(i), **options))


def _legacy_rewind(db, ids: list[int], sid, ordinal) -> None:
    for row_id in db.get_active_message_ids(sid):
        if row_id >= ids[ordinal]:
            db.deactivate_message(sid, row_id)


def _legacy_compact(db, ids: list[int], sid, replacement) -> None:
    db.archive_and_compact(sid, [
        {k: v for k, v in {"role": m.role, "content": m.content, "timestamp": m.timestamp,
                           "_compressed_summary": m.compressed_summary or None}.items() if v is not None}
        for m in replacement])


def _legacy_end(db, ids: list[int], sid, reason) -> None:
    db.end_session(sid, reason)


_LEGACY_OPS = {"session": _legacy_session, "msg": _legacy_msg, "rewind": _legacy_rewind,
               "compact": _legacy_compact, "end": _legacy_end}


def apply_legacy(script: Script, db) -> list[int]:
    """Replay ``script`` onto a legacy store through its own methods; returns the appended row ids."""
    ids: list[int] = []
    for kind, *args in script.ops:
        _LEGACY_OPS[kind](db, ids, *args)
    return ids


@pytest.fixture
def legacy(tmp_path, legacy_store):
    """Replay a script into a fresh legacy store and return what one of its read methods answers."""
    from hermes_state import SessionDB

    def run(script: Script, read: str, sid: str, **options: Any) -> list[dict]:
        db = SessionDB(db_path=tmp_path / f"oracle-{len(list(tmp_path.iterdir()))}.db")
        try:
            apply_legacy(script, db)
            return getattr(db, read)(sid, **options)
        finally:
            db.close()

    return run


PROJECTIONS = [
    pytest.param(ConversationRequest(), dict(), id="active"),
    pytest.param(ConversationRequest(repair_alternation=True), dict(repair_alternation=True), id="repaired"),
    pytest.param(ConversationRequest(view=TranscriptView.ALL, include_row_ids=True), dict(include_inactive=True, include_row_ids=True), id="all-with-ids"),
    pytest.param(ConversationRequest(view=TranscriptView.DISPLAY, include_lineage=True, include_row_ids=True), dict(include_compacted=True, include_ancestors=True, include_row_ids=True), id="display-lineage"),
]


@pytest.mark.parametrize("request_, options", PROJECTIONS)
def test_read_conversation_is_byte_identical_to_the_legacy_projection(backend, scope_a, legacy, request_, options):
    script = projection_script()
    script.apply(backend, scope_a)
    ours = backend.open(scope_a).messages.read_conversation(SESSION, request_)
    assert same_uids(dumps(ours)) == same_uids(dumps(legacy(script, "get_messages_as_conversation", SESSION, **options)))


@pytest.mark.parametrize("request_, options", PROJECTIONS)
def test_lineage_projection_is_byte_identical_to_the_legacy_projection(backend, scope_a, legacy, request_, options):
    script = lineage_script()
    script.apply(backend, scope_a)
    ours = backend.open(scope_a).messages.read_conversation(CHILD, request_)
    assert same_uids(dumps(ours)) == same_uids(dumps(legacy(script, "get_messages_as_conversation", CHILD, **options)))


PAGES = [
    pytest.param(dict(), id="whole"),
    pytest.param(dict(limit=3), id="limit"),
    pytest.param(dict(limit=2, offset=1), id="limit-offset"),
    pytest.param(dict(offset=2), id="offset-only"),
    pytest.param(dict(limit=3, latest=True), id="latest"),
    pytest.param(dict(limit=2, offset=1, latest=True), id="latest-offset"),
    pytest.param(dict(limit=0), id="empty-limit"),
    pytest.param(dict(limit=5, offset=50), id="past-the-end"),
]


def _read(view: TranscriptView, options: dict) -> TranscriptRead:
    return TranscriptRead(view=view, limit=options.get("limit"), offset=options.get("offset", 0), latest=options.get("latest", False))


def _wire(rows: Iterable[Any]) -> str:
    return json.dumps([row.as_wire_row() for row in rows], ensure_ascii=False)


@pytest.mark.parametrize("options", PAGES)
@pytest.mark.parametrize("view, script_of, sid", [
    pytest.param(TranscriptView.ACTIVE, projection_script, SESSION, id="active-rewound-tail"),
    pytest.param(TranscriptView.DISPLAY, projection_script, SESSION, id="display-rewound-tail"),
    pytest.param(TranscriptView.ACTIVE, lineage_script, CHILD, id="active-compacted"),
    pytest.param(TranscriptView.DISPLAY, lineage_script, CHILD, id="display-compacted"),
])
def test_read_transcript_page_equals_the_legacy_page_byte_for_byte(backend, scope_a, legacy, view, script_of, sid, options):
    script = script_of()
    script.apply(backend, scope_a)
    ours = backend.open(scope_a).messages.read_transcript(sid, _read(view, options))
    expected = legacy(script, "get_messages", sid, include_compacted=view is TranscriptView.DISPLAY, **options)
    assert same_uids(_wire(ours)) == same_uids(json.dumps(expected, ensure_ascii=False))


@pytest.mark.parametrize("options", PAGES)
def test_read_only_display_page_on_a_store_without_display_order_equals_legacy(backend, scope_a, tmp_path, legacy_store,
                                                                               options):
    """A read-only store whose display identities were never persisted (a pre-index store; the dashboard reads
    read-only) pages through the legacy scan and serves the legacy read-only page, writing nothing."""
    from hermes_state import SessionDB

    lineage_script().apply(backend, scope_a)
    path = store_for(scope_a)
    with sqlite3.connect(path) as raw:
        raw.execute("UPDATE messages SET display_order = NULL, display_identity = NULL")
    copy = tmp_path / "oracle.db"
    with sqlite3.connect(path) as source, sqlite3.connect(copy) as target:
        source.backup(target)
    reader = SQLiteBackend({scope_a.hermes_home_key: path}, read_only=True)
    try:
        ours = reader.open(scope_a).messages.read_transcript(CHILD, _read(TranscriptView.DISPLAY, options))
    finally:
        reader.close()
    oracle = SessionDB(db_path=copy, read_only=True)
    try:
        expected = oracle.get_messages(CHILD, include_compacted=True, **options)
    finally:
        oracle.close()
    assert _wire(ours) == json.dumps(expected, ensure_ascii=False)
    with sqlite3.connect(path) as raw:
        assert raw.execute("SELECT COUNT(*) FROM messages WHERE display_order IS NOT NULL").fetchone()[0] == 0


def test_single_append_stores_only_the_fields_the_legacy_append_message_takes(backend, scope_a, tmp_path, legacy_store):
    """``append`` drops ``data_use``, which main's ``append_message`` has no parameter for; ``append_batch`` folds it
    into ``display_metadata`` as main's batch does. Both read back as the legacy store's projection."""
    from hermes_state import SessionDB

    store = backend.open(scope_a)
    store.sessions.create(NewSession(SESSION, "cli"))
    store.messages.append(SESSION, new("assistant", "alone", 1, display_metadata={"m": 1}, data_use=("d",)))
    store.messages.append_batch(SESSION, [new("assistant", "batched", 2, display_metadata={"m": 1}, data_use=("d",))])
    db = SessionDB(db_path=tmp_path / "oracle.db")
    try:
        db.create_session(SESSION, "cli")
        db.append_message(SESSION, "assistant", content="alone", timestamp=ts(1), display_metadata={"m": 1})
        db.append_messages_batch(SESSION, [{"role": "assistant", "content": "batched", "timestamp": ts(2),
                                            "display_metadata": {"m": 1}, "data_use": ["d"]}])
        expected = db.get_messages_as_conversation(SESSION)
    finally:
        db.close()
    ours = store.messages.read_conversation(SESSION)
    assert [(m.display_metadata, m.data_use) for m in ours] == [({"m": 1}, ()), ({"m": 1, "data_use": ["d"]}, ("d",))]
    assert same_uids(dumps(ours)) == same_uids(dumps(expected))
