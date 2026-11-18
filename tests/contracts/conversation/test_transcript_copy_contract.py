"""Contract suite for the API fork's transcript copy: ``read_transcript`` of the source's ACTIVE rows replayed onto the destination through ``replace_transcript(ReplaceMode.DESTRUCTIVE)``.

The copy keeps every payload field the store's fork kept (``get_messages`` then ``replace_messages``): the row fields ``TranscriptRow`` serves, per-row ``token_count`` and ``display_metadata`` reactions included, with the reasoning columns kept on assistant rows only. Durable ids (row id, message and tool-call uids) are minted fresh on the destination.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from persistence import CompressionClosed, ProfileScope, RowVisibility
from persistence.core.identity import MessageId, SessionId
from persistence.conversation.messages import NewMessage, ReplaceMode, ReplaceResult, TranscriptRead, TranscriptRow, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from tests.contracts.conversation.test_messages import rewind_from, totals
from tests.contracts._scopes import MEDIA_ROOT

SOURCE = SessionId("copy-src")
DEST = SessionId("copy-dst")

_REASONING = ("reasoning", "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items")
_IDENTITY = ("id", "session_id", "visibility", "active", "compacted", "message_uid", "absorbed_message_uids", "tool_call_uids", "tool_call_uid")


def ts(i: int) -> float:
    return datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp()


def msg(content: Any, i: int = 0, role: str = "user", **kwargs: Any) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)


def _replayed(row: TranscriptRow) -> NewMessage:
    """The stored row as the write the fork replays: every payload field, durable ids left to the store."""
    return NewMessage(
        role=row.role, content=row.content, timestamp=row.timestamp, tool_calls=tuple(row.tool_calls or ()), tool_name=row.tool_name,
        tool_call_id=row.tool_call_id, display_kind=row.display_kind, compressed_summary=row.compressed_summary, api_content=row.api_content,
        display_metadata=row.display_metadata, platform_message_id=row.platform_message_id, observed=row.observed,
        effect_disposition=row.effect_disposition, finish_reason=row.finish_reason, reasoning=row.reasoning, reasoning_content=row.reasoning_content,
        reasoning_details=row.reasoning_details, codex_reasoning_items=row.codex_reasoning_items, codex_message_items=row.codex_message_items,
        token_count=row.token_count)


def fork(backend, scope: ProfileScope, source: SessionId, dest: SessionId) -> ReplaceResult:
    store = backend.open(scope)
    rows = store.messages.read_transcript(source)
    return store.messages.replace_transcript(dest, [_replayed(r) for r in rows], ReplaceMode.DESTRUCTIVE)


def _page(backend, scope: ProfileScope, session_id: SessionId, view: TranscriptView = TranscriptView.ACTIVE):
    store = backend.open(scope)
    return store.messages.read_transcript(session_id, TranscriptRead(view=view))


def _all(backend, scope: ProfileScope, session_id: SessionId):
    return _page(backend, scope, session_id, TranscriptView.ALL)


def _state(backend, scope: ProfileScope, session_id: SessionId) -> tuple:
    """What a refusal or a copy must leave untouched: every row, plus the live and display pages."""
    return (_all(backend, scope, session_id), _page(backend, scope, session_id), _page(backend, scope, session_id, TranscriptView.DISPLAY))


def _payload(row: TranscriptRow) -> dict:
    """Every wire field except row identity, placement and minted uids; the reasoning columns only on assistant rows, as the fork keeps them."""
    wire = row.as_wire_row()
    for key in _IDENTITY:
        wire.pop(key, None)
    if row.role != "assistant":
        for key in _REASONING:
            wire[key] = None
    return wire


def _seed_rich_source(backend, scope: ProfileScope) -> list[MessageId]:
    """A source whose rows carry every copied field, reactions and token counts on several rows, plus archived and rewound rows that must not travel."""
    store = backend.open(scope)
    repo = backend.open(scope).messages
    store.sessions.create(NewSession(session_id=SOURCE, source="cli"))
    call = {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    # History that is not ACTIVE: compaction-archived, then rewound after the compaction.
    repo.append(SOURCE, msg("archived turn", 0))
    repo.compact(SOURCE, [msg("summary", 1, compressed_summary=True, display_kind="compaction_summary")])
    rows = [
        msg("hello", 2, platform_message_id="plat-1", api_content="hello (sent)", display_metadata={"gateway_input_owner": "owner-1"}, token_count=7),
        msg("", 3, role="assistant", tool_calls=(call,), finish_reason="tool_calls", effect_disposition="applied",
            reasoning="look first", reasoning_content="rc", reasoning_details=[{"type": "summary", "text": "t"}],
            codex_reasoning_items=[{"id": "r1"}], codex_message_items=[{"id": "m1"}], token_count=11),
        msg("file body", 4, role="tool", tool_name="read_file", tool_call_id="call-1", reasoning="dropped on non-assistant",
            reasoning_content="rc drop", reasoning_details=[{"type": "summary", "text": "drop"}],
            codex_reasoning_items=[{"id": "r-drop"}], codex_message_items=[{"id": "m-drop"}], display_metadata={"note": "kept"}, token_count=13),
        msg("seen it", 5, observed=True),
        msg([{"type": "text", "text": "multimodal"}], 6, role="assistant", display_kind="chat"),
    ]
    ids = [repo.append(SOURCE, m) for m in rows]
    rewound_at = repo.append(SOURCE, msg("take it back", 7))
    repo.append(SOURCE, msg("gone reply", 8, role="assistant"))
    rewind_from(repo, SOURCE, rewound_at)
    repo.set_reaction(SOURCE, ids[0], "👍", author="user")
    repo.set_reaction(SOURCE, ids[0], "🎉", author="agent")
    repo.set_reaction(SOURCE, ids[3], "❤️", author="user")
    return ids


def _create(backend, scope: ProfileScope, *session_ids: SessionId) -> None:
    store = backend.open(scope)
    for session_id in session_ids:
        store.sessions.create(NewSession(session_id=session_id, source="cli"))


def test_copy_keeps_every_payload_field_in_order(backend, scope_a):
    _seed_rich_source(backend, scope_a)
    _create(backend, scope_a, DEST)
    source_before = _state(backend, scope_a, SOURCE)
    source_active = _page(backend, scope_a, SOURCE)
    # Precondition: reactions and token counts really are stored on the source.
    assert [r.token_count for r in source_active][1:4] == [7, 11, 13]
    assert [e["emoji"] for e in source_active[1].display_metadata["reactions"]] == ["👍", "🎉"]
    assert source_active[1].display_metadata["gateway_input_owner"] == "owner-1"

    fork(backend, scope_a, SOURCE, DEST)

    copied = _page(backend, scope_a, DEST)
    assert [_payload(r) for r in copied] == [_payload(r) for r in source_active]
    assert [r.content for r in copied] == ["summary", "hello", "", "file body", "seen it", [{"type": "text", "text": "multimodal"}]]
    # The non-assistant row loses all five reasoning columns; the assistant row keeps all five.
    assert all(getattr(source_active[3], key) is not None for key in _REASONING)
    assert [getattr(copied[3], key) for key in _REASONING] == [None] * len(_REASONING)
    assert [getattr(copied[2], key) for key in _REASONING] == [getattr(source_active[2], key) for key in _REASONING]
    assert all(getattr(copied[2], key) is not None for key in _REASONING)
    # Durable ids are minted fresh; the source is unchanged, every visibility.
    assert not {r.message_uid for r in copied} & {r.message_uid for r in source_active}
    assert _state(backend, scope_a, SOURCE) == source_before


def test_copy_mints_fresh_destination_rows_and_only_active_ones(backend, scope_a):
    _seed_rich_source(backend, scope_a)
    _create(backend, scope_a, DEST)
    source_ids = {r.id for r in _all(backend, scope_a, SOURCE)}

    result = fork(backend, scope_a, SOURCE, DEST)

    everything = _all(backend, scope_a, DEST)
    live = _page(backend, scope_a, DEST)
    assert result.survivor_ids == tuple(r.id for r in everything) == tuple(r.id for r in live)
    assert [(row.timestamp, row.message_uid) for row in result.rows] == [(r.timestamp, r.message_uid) for r in live]
    assert not source_ids & set(result.survivor_ids)
    assert {r.session_id for r in live} == {DEST}
    assert {r.visibility for r in everything} == {RowVisibility.ACTIVE}
    assert "archived turn" not in [r.content for r in everything]
    assert "take it back" not in [r.content for r in everything]
    counted = totals(backend, scope_a, DEST)
    assert (counted.message_count, counted.tool_call_count) == (6, 1)


def test_copy_replaces_every_prior_destination_row(backend, scope_a):
    repo = backend.open(scope_a).messages
    _create(backend, scope_a, SOURCE, DEST)
    repo.append(SOURCE, msg("new", 0))
    repo.append(DEST, msg("old archived", 0))
    repo.compact(DEST, [msg("old live", 1, role="assistant", tool_calls=({"id": "c", "type": "function", "function": {"name": "x", "arguments": "{}"}},))])

    fork(backend, scope_a, SOURCE, DEST)

    assert [r.content for r in _all(backend, scope_a, DEST)] == ["new"]
    counted = totals(backend, scope_a, DEST)
    assert (counted.message_count, counted.tool_call_count) == (1, 0)


@pytest.mark.parametrize("seed_source", [True, False], ids=["with-rows", "missing-source"])
def test_copy_recounts_totals_a_deactivation_left_on_the_destination(backend, scope_a, seed_source):
    repo = backend.open(scope_a).messages
    if seed_source:
        _create(backend, scope_a, SOURCE)
        repo.append(SOURCE, msg("carry", 0))
    _create(backend, scope_a, DEST)
    repo.append(DEST, msg("ask", 0))
    called = repo.append(DEST, msg("draft", 1, role="assistant", tool_calls=({"id": "c", "type": "function", "function": {"name": "x", "arguments": "{}"}},)))
    assert repo.deactivate_message(DEST, called)
    # Deactivation writes no counter, so the destination totals still hold the deactivated row and its call.
    before = totals(backend, scope_a, DEST)
    assert (before.message_count, before.tool_call_count) == (2, 1)

    fork(backend, scope_a, SOURCE, DEST)

    counted = totals(backend, scope_a, DEST)
    assert (counted.message_count, counted.tool_call_count) == (len(_all(backend, scope_a, DEST)), 0)


@pytest.mark.parametrize("seed_source", [False, True], ids=["missing", "empty"])
def test_missing_or_empty_source_empties_the_destination(backend, scope_a, seed_source):
    repo = backend.open(scope_a).messages
    if seed_source:
        _create(backend, scope_a, SOURCE)
    _create(backend, scope_a, DEST)
    repo.append(DEST, msg("old", 0))

    result = fork(backend, scope_a, SOURCE, DEST)

    assert result.survivor_ids == ()
    assert _all(backend, scope_a, DEST) == ()
    counted = totals(backend, scope_a, DEST)
    assert (counted.message_count, counted.tool_call_count) == (0, 0)


@pytest.mark.parametrize("deleted", [False, True], ids=["never-created", "deleted"])
def test_absent_destination_refuses_and_stays_absent(backend, scope_a, deleted):
    store_a = backend.open(scope_a)
    from persistence import SessionRowMissing

    _create(backend, scope_a, SOURCE)
    store_a.messages.append(SOURCE, msg("carry", 0))
    if deleted:
        _create(backend, scope_a, DEST)
        assert store_a.sessions.delete(DEST, media_root=MEDIA_ROOT) == (DEST,)
    source_before = _state(backend, scope_a, SOURCE)

    with pytest.raises(SessionRowMissing):
        fork(backend, scope_a, SOURCE, DEST)

    assert store_a.sessions.get(DEST) is None
    assert _all(backend, scope_a, DEST) == ()
    assert _state(backend, scope_a, SOURCE) == source_before


def test_compression_closed_destination_refuses_and_changes_nothing(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    _create(backend, scope_a, SOURCE, DEST)
    repo.append(SOURCE, msg("carry", 0))
    repo.append(DEST, msg("closed history", 0))
    sessions.end(DEST, END_REASON_COMPRESSION)
    tip = SessionId("copy-dst-tip")
    sessions.create(NewSession(session_id=tip, source="cli", parent_session_id=DEST))
    dest_before = _state(backend, scope_a, DEST)
    source_before = _state(backend, scope_a, SOURCE)

    with pytest.raises(CompressionClosed) as refused:
        fork(backend, scope_a, SOURCE, DEST)

    assert refused.value.tip == tip
    assert _state(backend, scope_a, DEST) == dest_before
    assert _state(backend, scope_a, SOURCE) == source_before
    assert totals(backend, scope_a, DEST).message_count == 1


def test_copy_onto_itself_reinserts_the_active_rows_and_drops_history(backend, scope_a):
    repo = backend.open(scope_a).messages
    _create(backend, scope_a, SOURCE)
    repo.append(SOURCE, msg("archived", 0))
    repo.compact(SOURCE, [msg("live", 1)])
    repo.append(SOURCE, msg("reply", 2, role="assistant", token_count=5))
    live = _page(backend, scope_a, SOURCE)
    repo.set_reaction(SOURCE, live[0].id, "👍", author="user")
    active_before = _page(backend, scope_a, SOURCE)

    result = fork(backend, scope_a, SOURCE, SOURCE)

    after = _page(backend, scope_a, SOURCE)
    assert [_payload(r) for r in after] == [_payload(r) for r in active_before]
    assert [r.token_count for r in after] == [None, 5]
    assert result.survivor_ids == tuple(r.id for r in after) == tuple(r.id for r in _all(backend, scope_a, SOURCE))
    assert not {r.id for r in active_before} & set(result.survivor_ids)


def test_copy_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_b).messages
    _create(backend, scope_b, SOURCE, DEST)
    _create(backend, scope_a, DEST)
    repo.append(SOURCE, msg("other profile source", 0))
    repo.append(DEST, msg("other profile dest", 0))
    store_a.messages.append(DEST, msg("mine", 0))

    # The source exists only in scope_b: in scope_a it reads empty, so the copy empties scope_a's destination only.
    fork(backend, scope_a, SOURCE, DEST)

    assert _all(backend, scope_a, DEST) == ()
    assert [r.content for r in _page(backend, scope_b, SOURCE)] == ["other profile source"]
    assert [r.content for r in _page(backend, scope_b, DEST)] == ["other profile dest"]
    # And a copy in scope_b never sees scope_a rows.
    _create(backend, scope_a, SOURCE)
    store_a.messages.append(SOURCE, msg("scope a source", 0))
    fork(backend, scope_b, SOURCE, DEST)
    assert [r.content for r in _page(backend, scope_b, DEST)] == ["other profile source"]
    assert _all(backend, scope_a, DEST) == ()
