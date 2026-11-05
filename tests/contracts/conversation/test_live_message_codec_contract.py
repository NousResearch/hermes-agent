"""Contract cases for the live message codec: what ``NewMessage.from_live`` and ``stamp_live`` around ``replace_transcript`` and ``append_batch`` leave on a caller's live dicts.

Each expected dict is literal: the keys left on that dict, with the values the store assigned read back through ``read_transcript`` and the write's result, so the minted uids and ownership versions are the backend's own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from persistence import CompressionClosed
from persistence.conversation.messages import NewMessage, ReplaceMode, stamp_live
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from persistence.core.identity import MessageId, SessionId

S = SessionId("codec-s1")
CALL = {"id": "call-1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
T = datetime(2026, 9, 11, 10, 0, 0, tzinfo=UTC).timestamp()


def _live() -> list[dict]:
    """A user turn with a stale canonical row, a tool call and its result, and a dict carrying another row's state."""
    return [
        {"role": "user", "content": "u1", "timestamp": T, "_canonical_row": {"stale": True}},
        {"role": "assistant", "content": "", "tool_calls": [CALL], "timestamp": T + 1},
        {"role": "tool", "content": "out", "tool_call_id": "call-1", "tool_name": "terminal", "timestamp": T + 2},
        {"role": "user", "content": "u2", "message_uid": "uid-given", "_db_row_snapshot": "snap-given", "_row_id": 999},
    ]


def _store(backend, scope):
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=S, source="cli"))
    return store


def _replace(repo, messages: list[dict], mode: ReplaceMode):
    result = repo.replace_transcript(S, [NewMessage.from_live(m) for m in messages], mode)
    stamp_live(messages, result)
    return result, repo.read_transcript(S)


def test_from_live_reads_every_key_the_store_reads():
    """CONTRACT: every key a writer reads off a live dict lands in its field; the born-durable marker and other live-only keys are not read."""
    live = {
        "role": "assistant", "content": [{"type": "text", "text": "hi"}], "timestamp": T, "tool_calls": [CALL], "tool_name": "terminal",
        "tool_call_id": "call-0", "display_kind": "hidden", "_compressed_summary": 1, "api_content": "sent", "display_metadata": {"k": 1},
        "platform_message_id": "pm-1", "observed": 1, "effect_disposition": "applied", "finish_reason": "stop", "reasoning": "r",
        "reasoning_content": "rc", "reasoning_details": [{"d": 1}], "codex_reasoning_items": [{"c": 1}], "codex_message_items": [{"m": 1}],
        "_row_id": 7, "_db_row_snapshot": "snap", "message_uid": "uid", "_absorbed_message_uids": ["a"], "_tool_call_uids": {"call-1": "tu"},
        "_tool_call_uid": "t", "token_count": 3, "data_use": ["memory"], "_db_persisted": True, "_canonical_row": {"x": 1},
    }
    assert NewMessage.from_live(live) == NewMessage(
        role="assistant", content=[{"type": "text", "text": "hi"}], timestamp=T, tool_calls=(CALL,), tool_name="terminal",
        tool_call_id="call-0", display_kind="hidden", compressed_summary=True, api_content="sent", display_metadata={"k": 1},
        platform_message_id="pm-1", observed=True, effect_disposition="applied", finish_reason="stop", reasoning="r",
        reasoning_content="rc", reasoning_details=[{"d": 1}], codex_reasoning_items=[{"c": 1}], codex_message_items=[{"m": 1}],
        repair_target=MessageId(7), db_row_snapshot="snap", message_uid="uid", absorbed_message_uids=("a",),
        tool_call_uids={"call-1": "tu"}, tool_call_uid="t", token_count=3, data_use=("memory",))


def test_from_live_reads_import_column_names_and_json_text():
    """CONTRACT: an import payload's column names and JSON text are read as the store reads them; a live key wins over its column even when unset; a missing role is unknown; message_id stands in for a missing platform_message_id; a bool row id, an empty uid and undecodable JSON text name nothing."""
    imported = {
        "content": "x", "tool_calls": json.dumps([CALL]), "tool_call_uids": json.dumps({"call-1": "tu"}),
        "absorbed_message_uids": json.dumps(["a", "b"]), "tool_call_uid": "t", "message_id": "pm-2", "_row_id": True, "message_uid": "",
    }
    assert NewMessage.from_live(imported) == NewMessage(
        role="unknown", content="x", tool_calls=(CALL,), tool_call_uids={"call-1": "tu"}, absorbed_message_uids=("a", "b"),
        tool_call_uid="t", platform_message_id="pm-2")
    shadowed = {"role": "assistant", "content": "", "_tool_call_uids": None, "tool_call_uids": json.dumps({"call-1": "tu"}), "tool_calls": "not json"}
    assert NewMessage.from_live(shadowed) == NewMessage(role="assistant", content="")


def test_replace_preserving_the_archive_stamps_every_row(backend, scope_a):
    """CONTRACT: every row a replace inserts gets its id, uid (the given one kept), stored timestamp and ownership version, an assistant row its tool-call uids and its result the paired uid; the stale canonical row and the other row's id and version are gone."""
    repo = _store(backend, scope_a).messages
    live = _live()
    result, rows = _replace(repo, live, ReplaceMode.PRESERVE_EXISTING_ARCHIVE)
    snaps = [stamp.db_row_snapshot for stamp in result.rows]
    calls = json.loads(rows[1].tool_call_uids)
    assert live == [
        {"role": "user", "content": "u1", "timestamp": T, "_row_id": rows[0].id, "message_uid": rows[0].message_uid, "_db_row_snapshot": snaps[0]},
        {"role": "assistant", "content": "", "tool_calls": [CALL], "timestamp": T + 1, "_row_id": rows[1].id, "message_uid": rows[1].message_uid,
         "_db_row_snapshot": snaps[1], "_tool_call_uids": calls},
        {"role": "tool", "content": "out", "tool_call_id": "call-1", "tool_name": "terminal", "timestamp": T + 2, "_row_id": rows[2].id,
         "message_uid": rows[2].message_uid, "_db_row_snapshot": snaps[2], "_tool_call_uid": calls["call-1"]},
        {"role": "user", "content": "u2", "message_uid": "uid-given", "_db_row_snapshot": snaps[3], "_row_id": rows[3].id, "timestamp": rows[3].timestamp},
    ]
    assert rows[2].tool_call_uid == calls["call-1"] and rows[3].message_uid == "uid-given"
    assert all(isinstance(s, str) and s and s != "snap-given" for s in snaps)
    assert len({m["message_uid"] for m in live}) == 4 and all(m["message_uid"] for m in live)


def test_replace_archiving_dropped_rows_stamps_a_kept_row_with_its_id_and_uid_only(backend, scope_a):
    """CONTRACT: a row the replace keeps takes only its stored id and uid, keeping its own timestamp (or none) and ownership version; the new tail is stamped as inserted."""
    repo = _store(backend, scope_a).messages
    rows = _replace(repo, _live(), ReplaceMode.PRESERVE_EXISTING_ARCHIVE)[1]
    kept = [
        {"role": "user", "content": "u1", "timestamp": T + 50, "_canonical_row": {"stale": True}},
        {"role": "assistant", "content": "", "tool_calls": [CALL], "_db_row_snapshot": "snap-given"},
        {"role": "user", "content": "u3"},
    ]
    result, active = _replace(repo, kept, ReplaceMode.ARCHIVE_DROPPED)
    assert [row.id for row in active[:2]] == [rows[0].id, rows[1].id]
    assert kept == [
        {"role": "user", "content": "u1", "timestamp": T + 50, "_row_id": rows[0].id, "message_uid": rows[0].message_uid},
        {"role": "assistant", "content": "", "tool_calls": [CALL], "_db_row_snapshot": "snap-given", "_row_id": rows[1].id, "message_uid": rows[1].message_uid},
        {"role": "user", "content": "u3", "_row_id": active[2].id, "message_uid": active[2].message_uid, "timestamp": active[2].timestamp,
         "_db_row_snapshot": result.rows[2].db_row_snapshot},
    ]
    assert result.rows[2].db_row_snapshot and active[2].message_uid


def test_append_batch_stamps_fresh_and_row_addressed_rows(backend, scope_a):
    """CONTRACT: a fresh batch row gets its id, uid, stored timestamp and ownership version; a row-addressed dict whose version another writer moved keeps its live content and takes the stored row's identity, its new version and the stored row as ``_canonical_row``; a stale canonical row on a fresh dict is dropped."""
    repo = _store(backend, scope_a).messages
    first = [{"role": "user", "content": "a", "timestamp": T}, {"role": "assistant", "content": "b", "timestamp": T + 1}]
    done = repo.append_batch(S, [NewMessage.from_live(m) for m in first])
    stamp_live(first, done)
    rows = repo.read_transcript(S)
    assert first == [
        {"role": "user", "content": "a", "timestamp": T, "_row_id": rows[0].id, "message_uid": rows[0].message_uid,
         "_db_row_snapshot": done.outcomes[0].db_row_snapshot},
        {"role": "assistant", "content": "b", "timestamp": T + 1, "_row_id": rows[1].id, "message_uid": rows[1].message_uid,
         "_db_row_snapshot": done.outcomes[1].db_row_snapshot},
    ]
    again = [dict(first[1], content="b edited", _db_row_snapshot="snap-given"),
             {"role": "user", "content": "c", "timestamp": T + 2, "_canonical_row": {"stale": True}}]
    done = repo.append_batch(S, [NewMessage.from_live(m) for m in again])
    stamp_live(again, done)
    rows = repo.read_transcript(S)
    assert [row.content for row in rows] == ["a", "b", "c"]
    assert again == [
        {"role": "assistant", "content": "b edited", "timestamp": T + 1, "_row_id": rows[1].id, "message_uid": rows[1].message_uid,
         "_db_row_snapshot": done.outcomes[0].db_row_snapshot,
         "_canonical_row": {"role": "assistant", "content": "b", "message_uid": rows[1].message_uid}},
        {"role": "user", "content": "c", "timestamp": T + 2, "_row_id": rows[2].id, "message_uid": rows[2].message_uid,
         "_db_row_snapshot": done.outcomes[1].db_row_snapshot},
    ]


def test_a_refused_write_only_drops_the_canonical_row(backend, scope_a):
    """CONTRACT: after a write refused before any insert (a session closed by compression) each dict keeps its keys, and only ``_canonical_row`` is gone."""
    store = _store(backend, scope_a)
    store.sessions.end(S, END_REASON_COMPRESSION)
    live = _live()
    with pytest.raises(CompressionClosed):
        store.messages.replace_transcript(S, [NewMessage.from_live(m) for m in live], ReplaceMode.PRESERVE_EXISTING_ARCHIVE)
    stamp_live(live, None)
    expected = _live()
    del expected[0]["_canonical_row"]
    assert live == expected
