"""Contract suite for compaction's held-row identity and coverage proof.

Backend cases run on every registered backend through ``MessageRepository.compact`` and read the outcome back through the public transcript views, so a backend that resolves carried identity differently, proves coverage differently, or drops the unseen rows fails on the backend that did it. The record-validation case checks held records and their wire payloads without a backend. The worker composite forwards the same two arguments; its transcript and receipt-replay case lives in ``tests/contracts/workers/test_worker_compaction_contract.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from persistence import CompressionInProgress, RowVisibility
from persistence.core.identity import MessageId, SessionId
from persistence.conversation.messages import DB_PERSISTED_MARKER_KEY, HeldCoverage, HeldMessage, NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession
from persistence.workers.compaction import WorkerCompaction
from tests.contracts.conftest import require_capability
from tests.contracts.conversation.test_messages import totals

SESSION = SessionId("held-compaction-session")


@pytest.fixture
def backend(backend, scope_a, scope_b):
    for scope in (scope_a, scope_b):
        backend.open(scope).sessions.create(NewSession(session_id=SESSION, source="cli"))
    return backend


def _ts(i: int) -> float:
    return datetime(2026, 9, 28, 10, 0, i, 250_000, tzinfo=UTC).timestamp()


def _msg(content: str, i: int, role: str = "user", **kwargs: object) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=_ts(i), **kwargs)  # type: ignore[arg-type]


def _held(record, *, row_id: object = True, timestamp: bool = True, persisted: bool = True, **overrides) -> HeldMessage:
    """The held copy of a stored row, as a compactor that loaded it would name it (``row_id``: True = the row's own id, False = none, else that id)."""
    fields = dict(
        role=record.role, content=record.content, tool_call_id=record.tool_call_id, tool_calls=tuple(record.tool_calls or ()),
        timestamp=record.timestamp if timestamp else None,
        row_id=record.id if row_id is True else (None if row_id is False else row_id), persisted=persisted,
    )
    fields.update(overrides)
    return HeldMessage(**fields)  # type: ignore[arg-type]


def _seed(repo, *rows: NewMessage):
    for row in rows:
        repo.append(SESSION, row)
    return list(repo.read_transcript(SESSION))


def _states(repo) -> dict:
    return {r.id: r.visibility for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))}


def _live(repo) -> list:
    return [(r.role, r.content) for r in repo.read_transcript(SESSION)]


def test_held_records_validate_and_round_trip_the_wire():
    with pytest.raises(ValueError):
        HeldMessage(role="", content="x")
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", row_id=MessageId(0))
    with pytest.raises(ValueError):
        HeldMessage(role="user", content="x", timestamp="yesterday")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeldCoverage(covered_ids=(MessageId(-1),))
    with pytest.raises(ValueError):
        HeldCoverage(unresolved=({"role": "user"},))  # type: ignore[arg-type]
    # The worker record carries held rows as the live dicts the compaction reads, key for key.
    held = {"role": "assistant", "content": "c", "tool_call_id": None, "tool_calls": [{"id": "t"}], "timestamp": 1790614856.9382791, "_row_id": 4, DB_PERSISTED_MARKER_KEY: True}
    payload = WorkerCompaction(messages=[], carried_messages=[held], covered_ids=[2], unresolved_held=[held]).payload()
    assert (payload["carried_messages"], payload["covered_ids"], payload["unresolved_held"]) == ([held], [2], [held])
    bare = WorkerCompaction(messages=[]).payload()
    assert (bare["carried_messages"], bare["covered_ids"], bare["unresolved_held"]) == ([], None, None)


def test_carried_noncontiguous_originals_are_rewound_and_the_rest_archived(backend, scope_a):
    repo = backend.open(scope_a).messages
    prefix, call, result, suffix = _seed(
        repo, _msg("prefix", 0), _msg("calling", 1, "assistant", tool_calls=({"id": "c1", "type": "function"},)),
        _msg("output", 2, "tool", tool_call_id="c1"), _msg("suffix", 3))
    replacement = [_msg("prefix", 0), _msg("SUMMARY", 4, "assistant"), _msg("suffix", 3)]
    assert (repo.compact(SESSION, replacement, carried=(_held(prefix), _held(suffix)))).active_count == 3
    states = _states(repo)
    assert (states[prefix.id], states[suffix.id]) == (RowVisibility.REWOUND, RowVisibility.REWOUND)
    assert (states[call.id], states[result.id]) == (RowVisibility.COMPACTION_ARCHIVED, RowVisibility.COMPACTION_ARCHIVED)
    display = [r.content for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.DISPLAY))]
    assert display.count("prefix") == 1 and display.count("suffix") == 1


@pytest.mark.parametrize("case", ["id_without_timestamp", "stale_id_unique_timestamp", "no_id_no_timestamp", "duplicate_timestamp", "changed_tool_call_id"])
def test_carried_identity_resolution_matrix(backend, scope_a, case):
    repo = backend.open(scope_a).messages
    rows = _seed(repo, _msg("twin", 0), _msg("twin", 0), _msg("solo", 1), _msg("tool out", 2, "tool", tool_call_id="c9"))
    twin_a, twin_b, solo, tool = rows
    held, expect_rewound = {
        # A matching row id wins on its own.
        "id_without_timestamp": (_held(solo, timestamp=False), {solo.id}),
        # An id naming a row of another identity falls back to the unique identity-plus-timestamp match.
        "stale_id_unique_timestamp": (_held(solo, row_id=twin_b.id), {solo.id}),
        # Nothing to go on names nothing; no refusal.
        "no_id_no_timestamp": (_held(solo, row_id=False, timestamp=False), set()),
        # Identity plus timestamp matching two rows is ambiguous and names nothing.
        "duplicate_timestamp": (_held(twin_a, row_id=False), set()),
        # The id's row no longer has this identity, and the fallback finds no row either.
        "changed_tool_call_id": (_held(tool, tool_call_id="c10"), set()),
    }[case]
    repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant")], carried=(held,))
    states = _states(repo)
    assert {i for i, v in states.items() if v is RowVisibility.REWOUND} == expect_rewound
    assert all(states[r.id] is RowVisibility.COMPACTION_ARCHIVED for r in rows if r.id not in expect_rewound)


def test_a_holding_proof_archives_only_what_it_covers_and_clones_the_rest(backend, scope_a):
    repo = backend.open(scope_a).messages
    a, gap, c = _seed(repo, _msg("held a", 0), _msg("unseen gap", 1, "assistant"), _msg("held c", 2))
    watermark = repo.active_watermark(SESSION)
    foreign = (_seed(repo, _msg("foreign tail", 3, "assistant")))[-1]
    # The proof overrides the watermark: the gap row below it survives too; a marker-less miss keeps the proof.
    coverage = HeldCoverage(covered_ids=(a.id, c.id), unresolved=(HeldMessage(role="user", content="never persisted"),))
    assert (repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant")], watermark=watermark, coverage=coverage)).active_count == 3
    assert _live(repo) == [("assistant", "SUMMARY"), ("assistant", "unseen gap"), ("assistant", "foreign tail")]
    live_ids = {r.id for r in repo.read_transcript(SESSION)}
    assert live_ids.isdisjoint({gap.id, foreign.id})
    states = _states(repo)
    assert (states[a.id], states[c.id]) == (RowVisibility.COMPACTION_ARCHIVED, RowVisibility.COMPACTION_ARCHIVED)
    assert (states[gap.id], states[foreign.id]) == (RowVisibility.REWOUND, RowVisibility.REWOUND)
    assert (totals(backend, scope_a, SESSION)).message_count == 3


def test_compaction_returns_the_ids_its_replacement_rows_received(backend, scope_a):
    repo = backend.open(scope_a).messages
    before = {r.id for r in _seed(repo, _msg("one", 0), _msg("two", 1, "assistant"))}
    watermark = repo.active_watermark(SESSION)
    _seed(repo, _msg("late", 2))
    result = repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant"), _msg("kept", 6)], watermark=watermark)
    live = repo.read_transcript(SESSION)
    assert result.active_count == len(live) == 3
    # The replacement rows lead the live transcript, in order, under the ids the result names; none is a reused id.
    assert result.row_ids == tuple(r.id for r in live[:2])
    assert before.isdisjoint(result.row_ids)


def test_an_empty_proof_preserves_every_active_row(backend, scope_a):
    repo = backend.open(scope_a).messages
    _seed(repo, _msg("one", 0), _msg("two", 1, "assistant"))
    assert (repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant")], coverage=HeldCoverage())).active_count == 3
    assert _live(repo) == [("assistant", "SUMMARY"), ("user", "one"), ("assistant", "two")]


@pytest.mark.parametrize("case", ["unique_match", "unpersisted_miss", "persisted_miss", "ambiguous", "non_string_persisted"])
def test_unresolved_held_decides_between_the_proof_and_the_watermark_path(backend, scope_a, case):
    repo = backend.open(scope_a).messages
    first, dup_a, dup_b = _seed(repo, _msg("first", 0), _msg("dup", 1), _msg("dup", 2))
    unresolved = {
        "unique_match": HeldMessage(role="user", content="first", persisted=True),
        "unpersisted_miss": HeldMessage(role="user", content="nowhere"),
        "persisted_miss": HeldMessage(role="user", content="nowhere", persisted=True),
        "ambiguous": HeldMessage(role="user", content="dup"),
        "non_string_persisted": HeldMessage(role="user", content=["first"], persisted=True),
    }[case]
    repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant")], coverage=HeldCoverage(unresolved=(unresolved,)))
    live = _live(repo)
    if case == "unique_match":
        # The proof holds and covers "first"; the dup rows were never covered and survive as clones.
        assert live == [("assistant", "SUMMARY"), ("user", "dup"), ("user", "dup")]
    elif case == "unpersisted_miss":
        # The proof holds and covers nothing.
        assert live == [("assistant", "SUMMARY"), ("user", "first"), ("user", "dup"), ("user", "dup")]
    else:
        # The proof is abandoned: coverage=None semantics, the watermark path archives everything.
        assert live == [("assistant", "SUMMARY")]
        states = _states(repo)
        assert all(states[r.id] is RowVisibility.COMPACTION_ARCHIVED for r in (first, dup_a, dup_b))


def test_a_refused_held_compaction_changes_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    require_capability(backend, "transactional_claims")
    rows = _seed(repo, _msg("one", 0), _msg("two", 1, "assistant"))
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(SESSION)), holder="owner", ttl_seconds=300.0) is True
    before = [(r.id, r.visibility, r.content) for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))]
    with pytest.raises(CompressionInProgress):
        repo.compact(SESSION, [_msg("SUMMARY", 5, "assistant")],
                           lock_holder="intruder",
                           carried=(_held(rows[1]),), coverage=HeldCoverage(covered_ids=(rows[0].id,)),
                           model_config_patch={"_usage_anchor": 9})
    assert [(r.id, r.visibility, r.content) for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))] == before
    record = store_a.sessions.get(SESSION)
    assert record is None or "_usage_anchor" not in record.model_config
