"""Contract cases for a failed ``append_batch``: the error carries what committed and every row's uids, so a caller can stamp the committed rows on its dicts and a retry repairs them instead of appending them twice.

The expected end state is a chunked batch's: each chunk commits in its own write; a later chunk's failure leaves the earlier chunks' rows durable with their dicts stamped, restores only the failed chunk's row id, ownership version and timestamp, and keeps the uids it stamped, including the tool-call uids stamped over the whole batch before chunking. No input fails a chunk on both backends alike, so the failure is injected into the row insert of the chunk named.
"""

from __future__ import annotations

import pytest

from persistence.conversation.messages import (
    BatchAppendProgress, NewMessage, RowDisposition, RowUids, TranscriptRead, TranscriptView, batch_progress, stamp_live,
)
from persistence.conversation.sessions import NewSession


class _Injected(RuntimeError):
    """The failure injected into a chunk's write."""


def _fail_chunk(backend_name: str, monkeypatch, chunk: int) -> None:
    """Make the row insert of the ``chunk``-th write (1-based) raise ``_Injected``: on SQLite after its rows went in, inside the write transaction, so the store's rollback and restore run; on the reference before it stores a row, its chunk having no rollback."""
    if backend_name == "sqlite":
        from persistence.sqlite.conversation._store import messages as module

        name = "_insert_message_rows"
    else:
        from tests.contracts._reference.conversation import messages_writes as module

        name = "insert_rows"
    real, calls = getattr(module, name), []

    def insert(*args, **kwargs):
        calls.append(None)
        if len(calls) != chunk:
            return real(*args, **kwargs)
        if backend_name == "sqlite":
            real(*args, **kwargs)
        raise _Injected("injected write failure")
    monkeypatch.setattr(module, name, insert)


def _live() -> list[dict]:
    return [
        {"role": "user", "content": "u1", "timestamp": 1790553600.0},
        {"role": "assistant", "content": "", "timestamp": 1790553601.0,
         "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "content": "r1", "timestamp": 1790553602.0, "tool_call_id": "call-1", "tool_name": "t"},
        {"role": "user", "content": "u2", "timestamp": 1790553603.0},
    ]


def _active(store, sid: str) -> list:
    return list(store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.ACTIVE)))


def _failed(store, sid: str, live: list[dict], **kwargs) -> BatchAppendProgress:
    with pytest.raises(_Injected) as caught:
        store.messages.append_batch(sid, [NewMessage.from_live(m) for m in live], **kwargs)
    progress = batch_progress(caught.value)
    assert progress is not None
    return progress


def test_a_later_chunk_failing_returns_the_committed_chunk_with_the_error(backend, backend_name, scope_a, monkeypatch):
    """CONTRACT: the first chunk stays committed and its outcomes come back on the error; every row's uids come back, the committed ones equal to their outcomes."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("bp-chunk", "cli"))
    _fail_chunk(backend_name, monkeypatch, 2)
    progress = _failed(store, "bp-chunk", _live(), chunk_rows=2)
    assert [o.disposition for o in progress.outcomes] == [RowDisposition.APPENDED, RowDisposition.APPENDED]
    assert len(progress.uids) == 4
    assert [row.id for row in _active(store, "bp-chunk")] == [o.id for o in progress.outcomes]
    for outcome, uids in zip(progress.outcomes, progress.uids):
        assert outcome.message_uid
        assert uids == RowUids(outcome.message_uid, outcome.tool_call_uids, outcome.tool_call_uid)


def test_the_tool_call_uids_stamped_before_chunking_come_back_on_the_failed_chunk(backend, backend_name, scope_a, monkeypatch):
    """CONTRACT: a result in the failed chunk carries the uid its call in the committed chunk was stored with."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("bp-pair", "cli"))
    _fail_chunk(backend_name, monkeypatch, 2)
    progress = _failed(store, "bp-pair", _live(), chunk_rows=2)
    call_uid = progress.outcomes[1].tool_call_uids["call-1"]
    assert call_uid and progress.uids[2].tool_call_uid == call_uid


def test_a_retry_after_stamping_the_progress_appends_nothing_twice(backend, backend_name, scope_a, monkeypatch):
    """CONTRACT: the caller stamps the progress on its live dicts; the same rows sent again repair the committed rows in place and append only the rest, the result keeping its call's uid."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("bp-retry", "cli"))
    live = _live()
    _fail_chunk(backend_name, monkeypatch, 2)
    progress = _failed(store, "bp-retry", live, chunk_rows=2)
    stamp_live(live, progress)
    assert [m.get("_row_id") for m in live] == [int(progress.outcomes[0].id), int(progress.outcomes[1].id), None, None]
    assert live[2]["_tool_call_uid"] == progress.outcomes[1].tool_call_uids["call-1"]
    result = store.messages.append_batch("bp-retry", [NewMessage.from_live(m) for m in live], chunk_rows=2)
    assert [o.disposition for o in result.outcomes] == [RowDisposition.REPAIRED, RowDisposition.REPAIRED,
                                                        RowDisposition.APPENDED, RowDisposition.APPENDED]
    assert [o.id for o in result.outcomes[:2]] == [o.id for o in progress.outcomes]
    rows = _active(store, "bp-retry")
    assert [row.content for row in rows] == ["u1", "", "r1", "u2"]
    assert rows[2].tool_call_uid == progress.outcomes[1].tool_call_uids["call-1"]


def test_a_single_transaction_failing_commits_nothing(backend, backend_name, scope_a, monkeypatch):
    """CONTRACT: without chunks nothing commits; the progress has no outcomes and one uid entry per row, and stamping it sets no row id."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("bp-one", "cli"))
    live = _live()
    _fail_chunk(backend_name, monkeypatch, 1)
    progress = _failed(store, "bp-one", live)
    assert progress.outcomes == () and len(progress.uids) == 4
    assert _active(store, "bp-one") == []
    stamp_live(live, progress)
    assert all("_row_id" not in m for m in live)


def test_other_errors_carry_no_progress():
    """CONTRACT: ``batch_progress`` reads ``None`` for an error that did not come out of ``append_batch``."""
    assert batch_progress(ValueError("x")) is None
