"""Contract suite for the message (transcript) capability.

Runs every registered backend through the ``MessageRepository`` contract; the documented exclusions (reopen durability, independent handles, carrier projection, JSON-string tool_calls, the no-op-rewind QUIRK path) live in the ``persistence.conversation.messages`` module docstring. Compaction lease fencing is tested here; ``model_config_patch`` is part of the contract rather than an exclusion.

Validity: backend cases run a real fixture backend end to end through public protocol operations only; record validation constructs records without a backend. Whole-transcript reads are asserted by exact sequence equality, which proves inclusion and exclusion in one assertion; refused operations are proven non-mutating against re-read state; the two-profile case seeds colliding domain ids with distinct sentinels and proves cross-scope invisibility and mutation impotence; the concurrency test records every attempt's outcome individually under a bounded timeout and asserts the full partition.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from persistence import CompressionInProgress, Conflict, ProfileScope, RowVisibility
from persistence.core.identity import MessageId, SessionId
from persistence.conversation.messages import (
    CooldownSnapshot,
    NewMessage,
    Reaction,
    ReplaceMode,
    TranscriptRead,
    TranscriptView,
    Watermark,
)
from persistence.conversation.sessions import NewSession
from tests.contracts.conftest import require_capability
from tests.contracts._concurrency import run_concurrently
from tests.contracts._waiting import wait_past
from tests.contracts.conversation.test_messages import Totals, rewind_from, totals

SESSION = SessionId("msg-contract-session-1")
OTHER_SESSION = SessionId("msg-contract-session-2")

NOW = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC).timestamp()
FUTURE = datetime(2100, 1, 1, tzinfo=UTC).timestamp()
PAST = datetime(2001, 9, 9, tzinfo=UTC).timestamp()


@pytest.fixture
def backend(backend, scope_a, scope_b):
    for scope in (scope_a, scope_b):
        for session_id in (SESSION, OTHER_SESSION):
            backend.open(scope).sessions.create(NewSession(session_id=session_id, source="cli"))
    return backend


def ts(i: int) -> float:
    return datetime(2026, 9, 11, 10, 0, i, tzinfo=UTC).timestamp()


def msg(content: str, i: int = 0, role: str = "user", **kwargs: object) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)  # type: ignore[arg-type]


def tool_calls(n: int, tag: str) -> tuple[dict, ...]:
    return tuple(
        {"id": f"call-{tag}-{i}", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}
        for i in range(n)
    )


def seed_turns(repo, session_id: SessionId = SESSION, turns: int = 3) -> list[MessageId]:
    """Seed *turns* user/assistant pairs through the public append; returns the ids in order."""
    ids: list[MessageId] = []
    for turn in range(turns):
        ids.append(repo.append(session_id, msg(f"user turn {turn}", i=turn * 2)))
        ids.append(repo.append(session_id, msg(f"assistant turn {turn}", i=turn * 2 + 1, role="assistant")))
    return ids


def contents(repo, session_id=SESSION, view=TranscriptView.ACTIVE) -> list[str]:
    return [str(r.content) for r in repo.read_transcript(session_id, TranscriptRead(view=view))]


def test_compact_hands_out_ownership_versions_a_later_edit_rewrites_with(backend, scope_a):
    """``row_snapshots`` name each replacement row's ownership version; handing one back with the row's id rewrites that row in place, so an edit to a compacted row is saved."""
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("m0", i=0))
    returned = repo.compact(SESSION, [msg("partial", i=10, role="assistant")])
    assert len(returned.row_snapshots) == 1 and returned.row_snapshots[0]
    completed = msg("completed answer", i=10, role="assistant", repair_target=returned.row_ids[0], db_row_snapshot=returned.row_snapshots[0])
    repo.append_batch(SESSION, [completed])
    assert contents(repo) == ["completed answer"]


def test_compact_stores_the_exact_epoch_timestamp_a_replacement_row_carries(backend, scope_a):
    """A row's epoch ``timestamp`` is stored exactly, so a row written with sub-microsecond seconds keeps its identity when a compaction re-inserts it and the display transcript shows it once."""
    store_a = backend.open(scope_a)
    exact = 1791234649.8550882
    turn = NewMessage(role="user", content="one logical turn", timestamp=exact)
    store_a.messages.append(SESSION, turn)
    store_a.messages.compact(SESSION, [turn])
    display = store_a.messages.read_transcript(SESSION, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [row.timestamp for row in display] == [exact]


def active_ids(repo, session_id=SESSION) -> list[MessageId]:
    return [r.id for r in repo.read_transcript(session_id)]


# ── record validation (shared conventions on the capability records) ──────────


def test_new_message_rejects_an_empty_role():
    with pytest.raises(ValueError):
        NewMessage(role="", content="x", timestamp=NOW)
    with pytest.raises(ValueError):
        CooldownSnapshot(present=False, until=FUTURE, error=None)


# ── append: ids, order, fields, totals ────────────────────────────────────────


def test_append_returns_distinct_ids_identifying_rows_in_insertion_order(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = [repo.append(SESSION, msg(f"msg-{i}", i=i, role="user" if i % 2 == 0 else "assistant")) for i in range(6)]
    assert len(set(ids)) == len(ids)
    records = repo.read_transcript(SESSION)
    assert [r.id for r in records] == ids
    assert [r.content for r in records] == [f"msg-{i}" for i in range(6)]


def test_readback_is_insertion_order_with_timestamps_stored_as_given(backend, scope_a):
    repo = backend.open(scope_a).messages
    stamps = [ts(50), ts(10), ts(30), ts(20)]
    for i, stamp in enumerate(stamps):
        repo.append(SESSION, NewMessage(role="user", content=f"ordered-{i}", timestamp=stamp))
    records = repo.read_transcript(SESSION)
    assert [r.content for r in records] == [f"ordered-{i}" for i in range(4)]
    assert [r.timestamp for r in records] == stamps


def test_append_field_round_trip(backend, scope_a):
    repo = backend.open(scope_a).messages
    calls = tool_calls(2, "rt")
    repo.append(SESSION, msg("calling tools", i=0, role="assistant", tool_calls=calls))
    repo.append(SESSION, msg("tool output", i=1, role="tool", tool_name="terminal", tool_call_id="call-rt-0"))
    assistant, tool = repo.read_transcript(SESSION)
    assert (assistant.role, assistant.content, assistant.tool_calls) == ("assistant", "calling tools", list(calls))
    assert (tool.role, tool.tool_name, tool.tool_call_id) == ("tool", "terminal", "call-rt-0")
    assert assistant.visibility is RowVisibility.ACTIVE
    assert (assistant.tool_name, tool.tool_calls) == (None, None)


def nested_message() -> NewMessage:
    calls = ({"id": "call-n-0", "type": "function", "function": {"name": "terminal", "arguments": "{}"}},)
    return NewMessage(role="assistant", content={"original": 1}, timestamp=ts(0), tool_calls=calls, display_metadata={"nested": {"k": 1}})


def change_nested(content: Any, calls: Any, metadata: Any) -> None:
    content["original"] = 99
    calls[0]["function"]["name"] = "changed"
    metadata["nested"]["k"] = 2


def stored_nested(repo, read: str = "read_transcript") -> tuple:
    row = getattr(repo, read)(SESSION)[0]
    return row.content, list(row.tool_calls), row.display_metadata


AS_WRITTEN = ({"original": 1}, [{"id": "call-n-0", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}], {"nested": {"k": 1}})


@pytest.mark.parametrize("verb", ["append", "append_batch", "replace_transcript"])
def test_changing_a_written_message_afterwards_changes_nothing_stored(backend, scope_a, verb):
    """CONTRACT: the store keeps its own copy of what a write took: changing a nested value of the written content, tool calls or display metadata afterwards leaves the stored row as written."""
    repo = backend.open(scope_a).messages
    message = nested_message()
    if verb == "append":
        repo.append(SESSION, message)
    elif verb == "append_batch":
        repo.append_batch(SESSION, [message])
    else:
        repo.replace_transcript(SESSION, [message], ReplaceMode.DESTRUCTIVE)
    change_nested(message.content, message.tool_calls, message.display_metadata)
    assert stored_nested(repo) == AS_WRITTEN


@pytest.mark.parametrize("read", ["read_transcript", "read_conversation"])
def test_changing_a_read_row_changes_nothing_stored(backend, scope_a, read):
    """CONTRACT: a read hands out the caller's own copy: changing a nested value of a row's content, tool calls or display metadata leaves the next read as stored."""
    repo = backend.open(scope_a).messages
    repo.append(SESSION, nested_message())
    change_nested(*stored_nested(repo, read))
    assert stored_nested(repo, read) == AS_WRITTEN


def test_changing_display_metadata_after_restamping_it_changes_nothing_stored(backend, scope_a):
    """CONTRACT: ``set_latest_matching_message_display_kind`` stores its own copy of the metadata it takes."""
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("plain", i=0))
    metadata = {"nested": {"k": 1}}
    assert repo.set_latest_matching_message_display_kind(SESSION, role="user", content="plain", display_kind="note", display_metadata=metadata)
    metadata["nested"]["k"] = 2
    assert repo.read_transcript(SESSION)[0].display_metadata == {"nested": {"k": 1}}


def test_totals_count_rows_and_tool_call_entries(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("plain 1", i=0))
    repo.append(SESSION, msg("two calls", i=1, role="assistant", tool_calls=tool_calls(2, "m1")))
    repo.append(SESSION, msg("out", i=2, role="tool", tool_name="terminal", tool_call_id="call-m1-0"))
    repo.append(SESSION, msg("three calls", i=3, role="assistant", tool_calls=tool_calls(3, "m2")))
    repo.append(SESSION, msg("plain 2", i=4))
    assert totals(backend, scope_a, SESSION) == Totals(5, 5, 0)


def test_deactivation_leaves_totals_until_a_recounting_write(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("u0", i=0))
    called = repo.append(SESSION, msg("a1", i=1, role="assistant", tool_calls=tool_calls(2, "d")))
    repo.append(SESSION, msg("u2", i=2))
    assert repo.deactivate_message(SESSION, called)
    # CONTRACT: deactivate_message writes no counter, so the totals keep the deactivated row and its calls; a later insert bumps them from there.
    assert totals(backend, scope_a, SESSION) == Totals(3, 2, 0)
    repo.append(SESSION, msg("u3", i=3))
    assert totals(backend, scope_a, SESSION) == Totals(4, 2, 0)
    # CONTRACT: compaction recounts from what it leaves active (summary plus the cloned unseen tail u0, u2, u3).
    repo.compact(SESSION, [msg("summary", i=9, role="assistant")], watermark=Watermark(None))
    assert totals(backend, scope_a, SESSION) == Totals(4, 0, 0)


def test_owned_repair_leaves_totals_until_a_recounting_write(backend, scope_a):
    repo = backend.open(scope_a).messages
    first = repo.append_batch(SESSION, [msg("ask", i=0), msg("draft", i=1, role="assistant")])
    mine = first.outcomes[1]
    calls = tool_calls(1, "r")
    repaired = repo.append_batch(SESSION, [msg("draft", i=1, role="assistant", tool_calls=calls, repair_target=mine.id, db_row_snapshot=mine.db_row_snapshot)])
    assert repaired.inserted == 0
    # CONTRACT: a snapshot-matched rewrite writes no counter: the rewrite gains a call but the totals stay as the inserts left them.
    assert totals(backend, scope_a, SESSION) == Totals(2, 0, 0)
    # CONTRACT: an archive-dropped replace recounts the rows it keeps, so the stored call is counted from then on.
    repo.replace_transcript(SESSION, [msg("ask", i=0), msg("draft", i=1, role="assistant", tool_calls=calls)], ReplaceMode.ARCHIVE_DROPPED)
    assert contents(repo) == ["ask", "draft"]
    assert totals(backend, scope_a, SESSION) == Totals(2, 1, 0)


def test_unknown_session_reads_are_typed_empties_not_errors(backend, scope_a):
    repo = backend.open(scope_a).messages
    ghost = SessionId("never-created")
    assert repo.read_transcript(ghost) == ()
    assert repo.read_transcript(ghost, TranscriptRead(view=TranscriptView.ALL)) == ()
    assert totals(backend, scope_a, ghost) == Totals(0, 0, 0)
    assert repo.active_watermark(ghost) == Watermark(last_active=None)


# ── append_batch ──────────────────────────────────────────────────────────────


def test_batch_lands_after_prior_rows_in_batch_order_and_bumps_totals(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("pre-batch", i=0))
    batch = [
        msg("b-0", i=1),
        msg("b-1", i=2, role="assistant", tool_calls=tool_calls(2, "b")),
        msg("b-2", i=3, role="tool", tool_name="terminal", tool_call_id="call-b-0"),
    ]
    assert (repo.append_batch(SESSION, batch)).inserted == 3
    records = repo.read_transcript(SESSION)
    assert [r.content for r in records] == ["pre-batch", "b-0", "b-1", "b-2"]
    assert records[2].tool_calls == list(tool_calls(2, "b"))
    assert totals(backend, scope_a, SESSION) == Totals(4, 2, 0)


def test_empty_batch_is_a_no_op(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("only", i=0))
    assert (repo.append_batch(SESSION, [])).inserted == 0
    assert contents(repo) == ["only"]
    assert totals(backend, scope_a, SESSION) == Totals(1, 0, 0)


# ── replace_transcript: the three typed intents ───────────────────────────────


def seed_replaceable(repo, session_id=SESSION) -> None:
    repo.append(session_id, msg("old-user", i=0))
    repo.append(session_id, msg("old-assistant", i=1, role="assistant", tool_calls=tool_calls(1, "old")))


def test_replace_destructive_leaves_no_trace_and_resets_totals(backend, scope_a):
    repo = backend.open(scope_a).messages
    seed_replaceable(repo)
    repo.replace_transcript(SESSION, [msg("new-user", i=10)], ReplaceMode.DESTRUCTIVE)
    assert contents(repo) == ["new-user"]
    assert contents(repo, view=TranscriptView.ALL) == ["new-user"]
    assert totals(backend, scope_a, SESSION) == Totals(1, 0, 0)


def test_replace_archive_dropped_keeps_old_rows_restorable(backend, scope_a):
    repo = backend.open(scope_a).messages
    seed_replaceable(repo)
    replacement = [msg("new-user", i=10), msg("new-assistant", i=11, role="assistant", tool_calls=tool_calls(2, "new"))]
    repo.replace_transcript(SESSION, replacement, ReplaceMode.ARCHIVE_DROPPED)
    assert contents(repo) == ["new-user", "new-assistant"]
    everything = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    assert [r.content for r in everything] == ["old-user", "old-assistant", "new-user", "new-assistant"]
    assert [r.visibility for r in everything[:2]] == [RowVisibility.REWOUND, RowVisibility.REWOUND]
    assert totals(backend, scope_a, SESSION) == Totals(2, 2, 0)


def test_replace_preserve_existing_archive_spares_archived_rows(backend, scope_a):
    repo = backend.open(scope_a).messages
    seed_replaceable(repo)
    repo.replace_transcript(SESSION, [msg("gen-2", i=10)], ReplaceMode.ARCHIVE_DROPPED)
    repo.replace_transcript(SESSION, [msg("gen-3", i=20)], ReplaceMode.PRESERVE_EXISTING_ARCHIVE)
    assert contents(repo, view=TranscriptView.ALL) == ["old-user", "old-assistant", "gen-3"]
    repo.replace_transcript(SESSION, [msg("gen-4", i=30)], ReplaceMode.DESTRUCTIVE)
    assert contents(repo, view=TranscriptView.ALL) == ["gen-4"]


@pytest.mark.parametrize("mode", list(ReplaceMode))
def test_a_replace_with_a_row_that_cannot_be_stored_changes_nothing(backend, scope_a, mode):
    """CONTRACT: a replacement row whose content cannot be serialized fails the whole replace: the rows, their visibility and the totals stay as they were."""
    repo = backend.open(scope_a).messages
    seed_replaceable(repo)
    before = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    replacement = [msg("new-user", i=10), NewMessage(role="assistant", content={"number": 10**5000}, timestamp=ts(11))]
    with pytest.raises(ValueError):
        repo.replace_transcript(SESSION, replacement, mode)
    assert repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL)) == before
    assert totals(backend, scope_a, SESSION) == Totals(2, 1, 0)


# ── compact ───────────────────────────────────────────────────────────────────


def test_a_compaction_with_a_row_that_cannot_be_stored_changes_nothing(backend, scope_a):
    """CONTRACT: a replacement row whose content cannot be serialized fails the whole compaction: the rows, their visibility and the totals stay as they were."""
    repo = backend.open(scope_a).messages
    seed_replaceable(repo)
    before = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    with pytest.raises(ValueError):
        repo.compact(SESSION, [msg("summary", i=10, role="assistant"), NewMessage(role="user", content={"number": 10**5000}, timestamp=ts(11))])
    assert repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL)) == before
    assert totals(backend, scope_a, SESSION) == Totals(2, 1, 0)


def test_compact_archives_active_rows_and_inserts_replacement(backend, scope_a):
    repo = backend.open(scope_a).messages
    for i, text in enumerate(["m0", "m1", "m2"]):
        repo.append(SESSION, msg(text, i=i))
    returned = repo.compact(SESSION, [msg("SUMMARY", i=10, role="assistant")])
    assert returned.active_count == 1

    live = repo.read_transcript(SESSION)
    assert [(r.role, r.content) for r in live] == [("assistant", "SUMMARY")]
    assert all(r.visibility is RowVisibility.ACTIVE for r in live)

    everything = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    archived = [r for r in everything if r.visibility is not RowVisibility.ACTIVE]
    assert [(r.content, r.visibility) for r in archived] == [
        ("m0", RowVisibility.COMPACTION_ARCHIVED),
        ("m1", RowVisibility.COMPACTION_ARCHIVED),
        ("m2", RowVisibility.COMPACTION_ARCHIVED),
    ]
    assert len(everything) == 4

    display = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [(r.content, r.visibility is RowVisibility.ACTIVE) for r in display] == [
        ("m0", False), ("m1", False), ("m2", False), ("SUMMARY", True)]
    assert totals(backend, scope_a, SESSION) == Totals(1, 0, 0)


def test_compact_totals_recomputed_from_replacement_set(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.append(SESSION, msg("m0", i=0, tool_calls=tool_calls(3, "pre")))
    replacement = [
        msg("SUMMARY", i=10, role="assistant"),
        msg("calls", i=11, role="assistant", tool_calls=tool_calls(1, "post")),
        msg("out", i=12, role="tool", tool_call_id="call-post-0"),
    ]
    assert (repo.compact(SESSION, replacement)).active_count == 3
    assert totals(backend, scope_a, SESSION) == Totals(3, 1, 0)


def test_compact_watermark_preserves_concurrent_tail_as_fresh_rows(backend, scope_a):
    repo = backend.open(scope_a).messages
    seeded = [repo.append(SESSION, msg(t, i=i)) for i, t in enumerate(["m0", "m1", "m2"])]
    watermark = repo.active_watermark(SESSION)
    tail_a = repo.append(SESSION, msg("tail-a", i=10))
    tail_b = repo.append(SESSION, msg("tail-b", i=11, role="assistant"))

    returned = repo.compact(SESSION, [msg("SUMMARY", i=20, role="assistant")], watermark=watermark)
    assert returned.active_count == 3
    live = repo.read_transcript(SESSION)
    assert [(r.role, r.content) for r in live] == [("assistant", "SUMMARY"), ("user", "tail-a"), ("assistant", "tail-b")]
    assert {tail_a, tail_b}.isdisjoint({r.id for r in live})

    by_id = {r.id: r for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))}
    assert by_id[tail_a].visibility is RowVisibility.REWOUND
    assert by_id[tail_b].visibility is RowVisibility.REWOUND
    assert all(by_id[i].visibility is RowVisibility.COMPACTION_ARCHIVED for i in seeded)

    display = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [r.content for r in display] == ["m0", "m1", "m2", "SUMMARY", "tail-a", "tail-b"]


def test_compact_carried_tail_originals_get_rewind_visibility(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = [repo.append(SESSION, msg(t, i=i)) for i, t in enumerate(["m0", "m1", "m2", "m3"])]
    replacement = [msg("SUMMARY", i=10, role="assistant"), msg("m3", i=3)]
    assert (repo.compact(SESSION, replacement, carried_tail_count=1)).active_count == 2

    by_id = {r.id: r for r in repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))}
    assert by_id[ids[3]].visibility is RowVisibility.REWOUND
    for archived_id in ids[:3]:
        assert by_id[archived_id].visibility is RowVisibility.COMPACTION_ARCHIVED

    display = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.DISPLAY))
    assert [r.content for r in display] == ["m0", "m1", "m2", "SUMMARY", "m3"]
    assert [r.content for r in display].count("m3") == 1


def test_watermark_captured_on_empty_transcript_preserves_everything_after(backend, scope_a):
    repo = backend.open(scope_a).messages
    watermark = repo.active_watermark(SESSION)
    assert watermark == Watermark(last_active=None)
    repo.append(SESSION, msg("appeared later", i=0))
    assert (repo.compact(SESSION, [msg("SUMMARY", i=10, role="assistant")], watermark=watermark)).active_count == 2
    assert contents(repo) == ["SUMMARY", "appeared later"]


# ── compaction lease fencing ─────────────────────────────────────
# The fence reads the compression lock the coordination capability
# holds on the session, so these cases require the
# coordination lease surface; transactional_claims is that repository's earned
# flag on every registered backend (repository registration keeps them in step).


def test_compact_with_live_lease_token_commits(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    require_capability(backend, "transactional_claims")
    seed_turns(repo, turns=1)
    token = "compressor-a" if store_a.coordination.try_acquire_compression_lock(
        session_id=SESSION, holder="compressor-a", ttl_seconds=300.0) else None
    assert token is not None
    # CONTRACT: the live lease owner passes the in-transition fence.
    assert (repo.compact(
        SESSION, [msg("SUMMARY", i=10, role="assistant")], lock_holder=token)).active_count == 1
    assert contents(repo) == ["SUMMARY"]


def test_compact_with_foreign_holder_token_refuses_without_publishing(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    require_capability(backend, "transactional_claims")
    seed_turns(repo, turns=1)
    assert store_a.coordination.try_acquire_compression_lock(
        session_id=SESSION, holder="owner", ttl_seconds=300.0) is True
    # CONTRACT: a holder that does not own the live lease is refused before
    # publishing and the transcript is untouched; only the live lease row can
    # vouch for a holder.
    forged = "intruder"
    before = contents(repo)
    with pytest.raises(CompressionInProgress):
        repo.compact(SESSION, [msg("stale", i=10, role="assistant")], lock_holder=forged)
    assert contents(repo) == before
    assert (totals(backend, scope_a, SESSION)).message_count == len(before)


def test_compact_with_expired_lease_token_refuses_without_publishing(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    require_capability(backend, "transactional_claims")
    seed_turns(repo, turns=1)
    # Reach expiry through a short REAL ttl (the negative-ttl route is QUIRK).
    token = "owner" if store_a.coordination.try_acquire_compression_lock(
        session_id=SESSION, holder="owner", ttl_seconds=0.01) else None
    assert token is not None
    wait_past(time.time() + 0.01)
    before = contents(repo)
    # CONTRACT: an expired lease fails the in-transition check; the stale
    # compaction is not published.
    with pytest.raises(CompressionInProgress):
        repo.compact(SESSION, [msg("stale", i=10, role="assistant")], lock_holder=token)
    assert contents(repo) == before
    # After a fresh acquire the same holder commits (the ABA window a reused
    # holder string opens stays QUIRK(lease-aba): documented
    # in the module docstrings, never asserted against backends).
    fresh = "owner" if store_a.coordination.try_acquire_compression_lock(
        session_id=SESSION, holder="owner", ttl_seconds=300.0) else None
    assert fresh is not None
    assert (repo.compact(
        SESSION, [msg("SUMMARY", i=11, role="assistant")], lock_holder=fresh)).active_count == 1


def test_compact_token_for_foreign_resource_refuses(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    require_capability(backend, "transactional_claims")
    seed_turns(repo, turns=1)
    # A legitimately held lease on a DIFFERENT session does not fence this
    # session: the fence reads only this session's compression lock.
    token = "owner" if store_a.coordination.try_acquire_compression_lock(
        session_id=OTHER_SESSION, holder="owner", ttl_seconds=300.0) else None
    assert token is not None
    before = contents(repo)
    with pytest.raises(CompressionInProgress):
        repo.compact(SESSION, [msg("stale", i=10, role="assistant")], lock_holder=token)
    assert contents(repo) == before
    # CONTRACT: lock_holder=None compacts unfenced.
    assert (repo.compact(SESSION, [msg("SUMMARY", i=11, role="assistant")])).active_count == 1


# ── compression-failure cooldown ──────────────────────────────────────────────


def test_cooldown_record_and_active_read(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.record_compression_cooldown(SESSION, FUTURE, "provider timeout")
    active = repo.read_compression_cooldown(SESSION, at=NOW)
    assert active is not None
    assert (active.until, active.error) == (FUTURE, "provider timeout")
    snapshot = repo.read_compression_cooldown_snapshot(SESSION)
    assert snapshot == CooldownSnapshot(present=True, until=FUTURE, error="provider timeout")


def test_cooldown_expired_reads_none_but_snapshot_survives(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.record_compression_cooldown(SESSION, PAST, "old failure")
    assert repo.read_compression_cooldown(SESSION, at=NOW) is None
    assert repo.read_compression_cooldown_snapshot(SESSION) == CooldownSnapshot(
        present=True, until=PAST, error="old failure")


def test_cooldown_merge_max_keeps_longer_live_deadline_and_latest_error(backend, scope_a):
    repo = backend.open(scope_a).messages
    earlier = datetime(2099, 12, 1, tzinfo=UTC).timestamp()
    later = datetime(2100, 2, 1, tzinfo=UTC).timestamp()
    repo.record_compression_cooldown(SESSION, FUTURE, "first")
    repo.record_compression_cooldown(SESSION, earlier, "second")
    snapshot = repo.read_compression_cooldown_snapshot(SESSION)
    assert (snapshot.until, snapshot.error) == (FUTURE, "second")
    repo.record_compression_cooldown(SESSION, later, "third")
    assert (repo.read_compression_cooldown_snapshot(SESSION)).until == later


@pytest.mark.quirk
def test_quirk_expired_stored_deadline_still_wins_merge_max(backend, backend_name, scope_a):
    # QUIRK(merge-max-vs-expired, test_cooldown_expired_deadline_still_wins_merge_max): merge-max compares
    # raw stored deadlines, not liveness. Documented on the reference backend only, never gating others:
    # a conforming backend may legitimately let a later write replace an expired deadline.
    if backend_name != "in_memory":
        pytest.skip("non-promoted QUIRK: documented on the reference backend only, never gates other backends")
    repo = backend.open(scope_a).messages
    slightly_later_past = datetime(2001, 9, 9, 0, 0, 5, tzinfo=UTC).timestamp()
    repo.record_compression_cooldown(SESSION, slightly_later_past, "expired")
    repo.record_compression_cooldown(SESSION, PAST, "smaller expired")
    snapshot = repo.read_compression_cooldown_snapshot(SESSION)
    assert (snapshot.until, snapshot.error) == (slightly_later_past, "smaller expired")


def test_cooldown_clear_nulls_fields_but_record_stays_present(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.record_compression_cooldown(SESSION, FUTURE, "boom")
    repo.clear_compression_cooldown(SESSION)
    assert repo.read_compression_cooldown(SESSION, at=NOW) is None
    assert repo.read_compression_cooldown_snapshot(SESSION) == CooldownSnapshot(
        present=True, until=None, error=None)


def test_cooldown_restore_round_trips_exact_snapshot(backend, scope_a):
    repo = backend.open(scope_a).messages
    repo.record_compression_cooldown(SESSION, FUTURE, "original")
    snapshot = repo.read_compression_cooldown_snapshot(SESSION)
    repo.clear_compression_cooldown(SESSION)
    repo.restore_compression_cooldown(SESSION, snapshot)
    assert repo.read_compression_cooldown_snapshot(SESSION) == snapshot


def test_cooldown_absent_record_reads_and_restore_edges(backend, scope_a):
    repo = backend.open(scope_a).messages
    ghost = SessionId("cooldown-ghost")
    assert repo.read_compression_cooldown(ghost, at=NOW) is None
    absent = repo.read_compression_cooldown_snapshot(ghost)
    assert absent == CooldownSnapshot(present=False, until=None, error=None)
    repo.clear_compression_cooldown(ghost)
    repo.restore_compression_cooldown(ghost, absent)
    assert repo.read_compression_cooldown_snapshot(ghost) == absent
    repo.record_compression_cooldown(SESSION, FUTURE, "live")
    with pytest.raises(Conflict):
        repo.restore_compression_cooldown(SESSION, absent)
    assert repo.read_compression_cooldown_snapshot(SESSION) == CooldownSnapshot(
        present=True, until=FUTURE, error="live")


# ── two-profile isolation ──────────────────────────────────────────────────────


def test_two_scopes_colliding_session_ids_stay_isolated(backend, backend_name, scope_a, scope_b):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_b).messages
    store_a.messages.append(SESSION, msg("sentinel-a", i=0))
    id_a2 = store_a.messages.append(SESSION, msg("sentinel-a-2", i=1))
    repo.append(SESSION, msg("sentinel-b", i=0))
    store_a.messages.record_compression_cooldown(SESSION, FUTURE, "a-only")

    assert contents(backend.open(scope_a).messages) == ["sentinel-a", "sentinel-a-2"]
    assert contents(repo) == ["sentinel-b"]
    assert totals(backend, scope_b, SESSION) == Totals(1, 0, 0)
    # The other scope keeps its present session row without a cooldown.
    snapshot_b = repo.read_compression_cooldown_snapshot(SESSION)
    assert (snapshot_b.until, snapshot_b.error) == (None, None)
    if backend_name == "in_memory":
        assert snapshot_b == CooldownSnapshot(present=True, until=None, error=None)

    # Cross-scope mutation impotence: ids are scope-local anchors, so scope A's
    # second row (an id scope B never minted) is unaddressable from scope B, and
    # the refused deactivation leaves both scopes untouched.
    assert repo.deactivate_message(SESSION, id_a2) is False
    assert contents(backend.open(scope_a).messages) == ["sentinel-a", "sentinel-a-2"]
    assert contents(repo) == ["sentinel-b"]
    assert (totals(backend, scope_a, SESSION)).rewind_count == 0
    assert (totals(backend, scope_b, SESSION)).rewind_count == 0

    # Cross-scope replace touches only the caller's scope.
    repo.replace_transcript(SESSION, [msg("b-replaced", i=10)], ReplaceMode.DESTRUCTIVE)
    assert contents(backend.open(scope_a).messages) == ["sentinel-a", "sentinel-a-2"]
    assert contents(repo) == ["b-replaced"]


# ── concurrency: append storm through the public surface ─────────────────────


def test_concurrent_append_storm_exact_totals_and_distinct_ids(backend, scope_a):
    repo = backend.open(scope_a).messages
    n_actors, n_rounds = 8, 20
    outcomes: list[MessageId] = []
    failures: list[BaseException] = []

    def actor(a: int) -> None:
        # Record every attempt's result or exception individually so the final
        # partition (successes + failures = attempts) is fully observable.
        for r in range(n_rounds):
            try:
                outcomes.append(
                    repo.append(SESSION, msg(f"storm-{a}-{r}", i=0, tool_calls=tool_calls(2, f"{a}-{r}"))))
            except BaseException as exc:
                failures.append(exc)

    run_concurrently(*[lambda a=a: actor(a) for a in range(n_actors)])
    assert failures == []
    assert len(outcomes) == n_actors * n_rounds
    assert len(set(outcomes)) == n_actors * n_rounds
    counted = totals(backend, scope_a, SESSION)
    assert (counted.message_count, counted.tool_call_count) == (n_actors * n_rounds, n_actors * n_rounds * 2)
    records = repo.read_transcript(SESSION)
    assert sorted(str(r.content) for r in records) == sorted(
        f"storm-{a}-{r}" for a in range(n_actors) for r in range(n_rounds))


# ── latest_dialogue_message ────────────


def test_latest_dialogue_message_skips_non_dialogue_and_blank_rows(backend, scope_a):
    repo = backend.open(scope_a).messages
    # Distinguishable seed: dialogue rows, then newer rows the preview must skip
    # (tool result, system note, spaces-only streaming placeholder). The blank
    # arm uses spaces exactly: the predicate is SQLite-style TRIM, and other
    # whitespace kinds are the documented QUIRK region, deliberately unseeded.
    repo.append(SESSION, msg("hello there", i=0))
    repo.append(SESSION, msg("the real preview", i=1, role="assistant"))
    repo.append(SESSION, msg("tool output", i=2, role="tool", tool_name="terminal", tool_call_id="call-p-0"))
    repo.append(SESSION, msg("system note", i=3, role="system"))
    repo.append(SESSION, msg("    ", i=4, role="assistant"))
    # CONTRACT: the content of the newest ACTIVE user/assistant row with non-blank content,
    # not merely the newest row.
    assert repo.latest_dialogue_message(SESSION) == "the real preview"


def test_latest_dialogue_message_typed_none_arms(backend, scope_a):
    repo = backend.open(scope_a).messages
    # CONTRACT: an unknown session reads None, never an exception.
    assert repo.latest_dialogue_message(SessionId("msg-nope")) is None
    # A transcript with no qualifying row reads None the same way.
    repo.append(OTHER_SESSION, msg("tool only", i=0, role="tool", tool_name="terminal", tool_call_id="call-p-1"))
    repo.append(OTHER_SESSION, msg("   ", i=1))
    assert repo.latest_dialogue_message(OTHER_SESSION) is None


def test_latest_dialogue_message_ignores_rewound_history(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=2)
    # Rewind from the second user turn: the newest surviving dialogue row is turn 0's
    # assistant reply, and the rewound rows must not surface even though they are newer.
    rewind_from(repo, SESSION, ids[2])
    assert repo.latest_dialogue_message(SESSION) == "assistant turn 0"


# ── message_storage_state ─────────────────


def test_message_storage_state_resolves_owner_and_visibility_by_id_alone(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=2)
    other = repo.append(OTHER_SESSION, msg("elsewhere", i=9))
    # CONTRACT: the anchor alone names its owning session; no session argument is needed or consulted.
    state = repo.message_storage_state(ids[0])
    assert state is not None
    assert (state.session_id, state.visibility) == (SESSION, RowVisibility.ACTIVE)
    assert (repo.message_storage_state(other)).session_id == OTHER_SESSION
    # CONTRACT: rows of every visibility resolve, each reporting its own partition.
    rewind_from(repo, SESSION, ids[2])
    rewound = repo.message_storage_state(ids[3])
    assert (rewound.session_id, rewound.visibility) == (SESSION, RowVisibility.REWOUND)
    repo.compact(SESSION, [msg("summary", i=20, role="assistant")])
    archived = repo.message_storage_state(ids[0])
    assert (archived.session_id, archived.visibility) == (SESSION, RowVisibility.COMPACTION_ARCHIVED)
    # The consumer's predicate (tools/session_search_tool._is_compacted_state) expressed on the record.
    assert archived.visibility is RowVisibility.COMPACTION_ARCHIVED and rewound.visibility is not RowVisibility.COMPACTION_ARCHIVED
    live = repo.message_storage_state((repo.read_transcript(SESSION))[0].id)
    assert live.visibility is RowVisibility.ACTIVE


def test_message_storage_state_unknown_id_is_typed_none(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=1)
    # CONTRACT: an anchor no stored row carries reads None, never NotFound; an untouched store reads None too.
    assert repo.message_storage_state(MessageId(max(int(i) for i in ids) + 1000)) is None
    assert repo.message_storage_state(MessageId(0)) is None
    # CONTRACT: a destructive replace removes the rows, so their anchors stop resolving.
    repo.replace_transcript(SESSION, [msg("fresh", i=5)], ReplaceMode.DESTRUCTIVE)
    assert repo.message_storage_state(ids[0]) is None
    fresh = (repo.read_transcript(SESSION))[0].id
    assert (repo.message_storage_state(fresh)).session_id == SESSION


def test_message_storage_state_scope_isolation(backend, scope_a, scope_b):
    store_b = backend.open(scope_b)
    repo = backend.open(scope_a).messages
    a_id = repo.append(SESSION, msg("in a", i=1))
    b_id = store_b.messages.append(OTHER_SESSION, msg("in b", i=1))
    # CONTRACT: a lookup never resolves to a row stored in another scope: the answer in each scope is that scope's own row or None.
    in_a = repo.message_storage_state(b_id)
    assert in_a is None or in_a.session_id == SESSION
    in_b = store_b.messages.message_storage_state(a_id)
    assert in_b is None or in_b.session_id == OTHER_SESSION
    assert (repo.message_storage_state(a_id)).session_id == SESSION
    assert (store_b.messages.message_storage_state(b_id)).session_id == OTHER_SESSION


# ── reactions family ───────────────────────


def _emojis(reactions) -> list[tuple[str, str]]:
    return [(r.author, r.emoji) for r in reactions]


def test_latest_row_id_targets_newest_active_dialogue_row_with_offset(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=3)
    # CONTRACT: offset 0 is the newest active row of the role; offsets step back one row of that role at a time.
    assert repo.latest_row_id(SESSION, role="user") == ids[4]
    assert repo.latest_row_id(SESSION, role="user", offset=1) == ids[2]
    assert repo.latest_row_id(SESSION, role="user", offset=2) == ids[0]
    assert repo.latest_row_id(SESSION, role="assistant") == ids[5]
    # CONTRACT: stepping past the oldest row of the role reads None, never an error.
    assert repo.latest_row_id(SESSION, role="user", offset=3) is None
    # CONTRACT: newer rows of OTHER roles never shadow the target role.
    repo.append(SESSION, msg("tool output", i=30, role="tool", tool_name="terminal", tool_call_id="call-r-0"))
    assert repo.latest_row_id(SESSION, role="user") == ids[4]
    # CONTRACT: an unknown session reads None.
    assert repo.latest_row_id(SessionId("msg-nope"), role="user") is None


def test_latest_row_id_skips_invisible_bubbles_unless_asked(backend, scope_a):
    repo = backend.open(scope_a).messages
    real = repo.append(SESSION, msg("real words", i=0))
    blank = repo.append(SESSION, msg("    ", i=1))
    # CONTRACT: with require_text (the default) a spaces-only row is skipped so the newest visible message is the target; the predicate strips spaces exactly (other whitespace kinds are the documented QUIRK region, deliberately unseeded).
    assert repo.latest_row_id(SESSION, role="user") == real
    # CONTRACT: require_text=False resolves the newest row regardless of content.
    assert repo.latest_row_id(SESSION, role="user", require_text=False) == blank


def test_latest_row_id_ignores_inactive_rows(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=2)
    rewind_from(repo, SESSION, ids[2])
    # CONTRACT: rewound rows are not targets; the newest surviving active user row is turn 0's.
    assert repo.latest_row_id(SESSION, role="user") == ids[0]


def test_latest_row_id_out_of_range_arguments_read_none(backend, scope_a):
    repo = backend.open(scope_a).messages
    seed_turns(repo, turns=1)
    # CONTRACT: a non-dialogue role and a negative offset read None, never an error.
    assert repo.latest_row_id(SESSION, role="tool") is None
    assert repo.latest_row_id(SESSION, role="user", offset=-1) is None
    # CONTRACT: the role defaults to user.
    assert repo.latest_row_id(SESSION) == repo.latest_row_id(SESSION, role="user")


def test_message_role_resolves_only_active_rows_of_the_named_session(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=2)
    other = repo.append(OTHER_SESSION, msg("elsewhere", i=9, role="assistant"))
    # CONTRACT: the row's role inside its own session.
    assert repo.message_role(SESSION, ids[0]) == "user"
    assert repo.message_role(SESSION, ids[1]) == "assistant"
    # CONTRACT: another session's row, an unknown id, and an inactive row all read None.
    assert repo.message_role(SESSION, other) is None
    assert repo.message_role(SESSION, MessageId(max(int(i) for i in ids) + 1000)) is None
    rewind_from(repo, SESSION, ids[2])
    assert repo.message_role(SESSION, ids[2]) is None


def test_set_reaction_tapback_semantics_one_per_author(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=1)
    target = ids[0]
    first = repo.set_reaction(SESSION, target, "👍", author="agent")
    # CONTRACT: the wholesale list after the write; the record carries emoji, author, and an epoch timestamp.
    assert first is not None and _emojis(first) == [("agent", "👍")]
    assert all(isinstance(r, Reaction) and isinstance(r.at, float) and r.at > 0 for r in first)
    # CONTRACT: a different emoji by the same author REPLACES (still one per author).
    assert _emojis(repo.set_reaction(SESSION, target, "❤️", author="agent")) == [("agent", "❤️")]
    # CONTRACT: another author's reaction coexists; the writer's fresh reaction lists last.
    both = repo.set_reaction(SESSION, target, "😂", author="user")
    assert _emojis(both) == [("agent", "❤️"), ("user", "😂")]
    # CONTRACT: the same emoji again by its author CLEARS it (toggle); the other author's stays.
    assert _emojis(repo.set_reaction(SESSION, target, "❤️", author="agent")) == [("user", "😂")]
    # CONTRACT: None clears; clearing an absent reaction is a no-op that still returns the list.
    assert _emojis(repo.set_reaction(SESSION, target, None, author="user")) == []
    assert _emojis(repo.set_reaction(SESSION, target, None, author="user")) == []


def test_set_reaction_never_touches_the_transcript(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=2)
    before = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    totals_before = totals(backend, scope_a, SESSION)
    # The reactions must actually land (a stubbed writer would make the invariance below vacuous).
    assert _emojis(repo.set_reaction(SESSION, ids[0], "👍", author="agent")) == [("agent", "👍")]
    assert _emojis(repo.set_reaction(SESSION, ids[3], "🎉", author="user")) == [("user", "🎉")]
    # CONTRACT (agent/AGENTS.md cache invariant): reacting adds no row, edits no content, role, timestamp, or visibility, and moves no counter; the stored row read differs only by the reactions its display metadata carries.
    after = repo.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    assert [replace(r, display_metadata=None) for r in after] == [replace(r, display_metadata=None) for r in before]
    assert [[(x["author"], x["emoji"]) for x in (r.display_metadata or {}).get("reactions", [])] for r in after] == [
        [("agent", "👍")], [], [], [("user", "🎉")]]
    assert totals(backend, scope_a, SESSION) == totals_before


def test_set_reaction_on_a_foreign_row_is_typed_none_and_writes_nothing(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=1)
    other = repo.append(OTHER_SESSION, msg("elsewhere", i=9))
    # CONTRACT: a row outside the named session's visible resume lineage (an unrelated session's row, an unknown id) reads None.
    assert repo.set_reaction(SESSION, other, "👍", author="agent") is None
    assert repo.set_reaction(SESSION, MessageId(max(int(other), *map(int, ids)) + 1000), "👍", author="agent") is None
    # ...and nothing landed on the foreign row: its own session sees an empty list on a no-op clear.
    assert _emojis(repo.set_reaction(OTHER_SESSION, other, None, author="agent")) == []
    # CONTRACT: a rewound row is no longer on screen, so it is not addressable either: typed None, nothing written.
    rewind_from(repo, SESSION, ids[0])
    assert repo.set_reaction(SESSION, ids[0], "👀", author="agent") is None


def test_set_reaction_stores_the_author_as_given(backend, scope_a):
    repo = backend.open(scope_a).messages
    ids = seed_turns(repo, turns=1)
    # CONTRACT: the author is not validated; an empty author is one more author.
    assert _emojis(repo.set_reaction(SESSION, ids[0], "👍", author="")) == [("", "👍")]
    assert _emojis(repo.set_reaction(SESSION, ids[0], "❤️", author="agent")) == [("", "👍"), ("agent", "❤️")]


def test_reactions_are_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_b).messages
    a_id = store_a.messages.append(SESSION, msg("in a", i=1))
    b_id = repo.append(SESSION, msg("in b", i=1))
    assert _emojis(store_a.messages.set_reaction(SESSION, a_id, "👍", author="agent")) == [("agent", "👍")]
    # CONTRACT: a reaction set in one scope is invisible in the other. Observe B through a DIFFERENT author's no-op clear (so the probe cannot erase what it is looking for), then prove A still holds its reaction the same way.
    assert _emojis(repo.set_reaction(SESSION, b_id, None, author="probe")) == []
    foreign = repo.set_reaction(SESSION, a_id, None, author="probe")
    assert foreign is None or _emojis(foreign) == []
    assert _emojis(store_a.messages.set_reaction(SESSION, a_id, None, author="probe")) == [("agent", "👍")]
    assert repo.latest_row_id(SESSION, role="user") == b_id
