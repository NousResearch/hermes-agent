"""Passive conversation-history commits: one atomic append per client event.

Every case runs against a real temporary ``state.db`` with two independent writer
handles, because the contract under test is a cross-writer one: the receipt, the
canonical rows and the external-history watermark either all land together or none
of them do, and a second writer replaying the same client event must observe the
first commit rather than copy it.
"""

from __future__ import annotations

import os
import threading

import pytest

from hermes_state import SessionDB
from hermes_state_passive_history import (
    PassiveHistoryBusyError,
    PassiveHistoryConflictError,
    PassiveHistoryRetiredError,
    PassiveHistoryTargetError,
    SessionPassiveHistoryMixin,
)

USER = {"role": "user", "content": "book the 9am flight"}
ASSISTANT = {"role": "assistant", "content": "booked, confirmation ABC123"}
IDENTITY = {"producer": "talk.voice", "event_id": "evt-1", "origin_turn_id": "origin-1"}


def _append(db, session_id, *, messages, **overrides):
    kwargs = {**IDENTITY, **overrides}
    return db.append_passive_messages(session_id, messages=messages, **kwargs)


def _receipt_rows(db):
    with db._read_ctx() as conn:
        return conn.execute(
            "SELECT producer, event_id, origin_turn_id, conversation_id, session_id, message_ids_json "
            "FROM passive_history_commits ORDER BY id"
        ).fetchall()


def _rows(db, session_id):
    """Every stored row for a session, live or archived (deletion is never inferred from ``active``)."""
    return [
        (m["role"], m["content"], m.get("display_kind"))
        for m in db.get_messages(session_id, include_inactive=True, include_compacted=True)
    ]


@pytest.fixture
def store(tmp_path):
    """A live conversation plus two independent process-shaped writers on one file."""
    path = tmp_path / "state.db"
    writer, peer = SessionDB(path), SessionDB(path)
    writer.create_session("conv", source="test")
    try:
        yield writer, peer
    finally:
        writer.close()
        peer.close()


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([USER], id="single-user"),
        pytest.param([ASSISTANT], id="single-assistant"),
        pytest.param([USER, ASSISTANT], id="ordered-pair"),
    ],
)
def test_accepted_shapes_commit_once_with_bounded_provenance(store, messages):
    writer, peer = store

    receipt = _append(writer, "conv", messages=messages)

    assert receipt.replayed is False
    assert receipt.conversation_id == "conv" and receipt.session_id == "conv"
    assert (receipt.producer, receipt.event_id, receipt.origin_turn_id) == (
        IDENTITY["producer"], IDENTITY["event_id"], IDENTITY["origin_turn_id"])
    assert len(receipt.message_ids) == len(messages)
    # Canonical commit order matches submission order and row ids only move forward.
    assert list(receipt.message_ids) == sorted(receipt.message_ids)
    assert _rows(peer, "conv") == [
        (m["role"], m["content"], "passive_conversation") for m in messages]
    # Counters reconcile through the canonical bump, and no tool-call count is invented.
    session = peer.get_session("conv")
    assert (session["message_count"], session["tool_call_count"]) == (len(messages), 0)
    # Metadata carries provenance and the row index only: no transcript, no caller row ids.
    stored = peer.get_messages("conv")
    for index, row in enumerate(stored):
        assert row["display_metadata"] == {
            "producer": IDENTITY["producer"], "event_id": IDENTITY["event_id"],
            "origin_turn_id": IDENTITY["origin_turn_id"], "index": index,
        }
    # Timestamps are host-assigned and monotonic within the committed pair.
    assert [row["timestamp"] for row in stored] == sorted(row["timestamp"] for row in stored)
    watermark = peer.get_passive_history_watermark("conv")
    assert (watermark.conversation_id, watermark.revision) == ("conv", receipt.revision)
    assert receipt.revision > 0


def test_finalized_user_turn_is_durable_without_any_assistant_reply(store, tmp_path):
    """A disconnect right after transcription must not cost the operator their turn."""
    writer, _peer = store
    receipt = _append(writer, "conv", messages=[USER])
    writer.end_session("conv", "cli_close")
    writer.close()

    reopened = SessionDB(tmp_path / "state.db")
    try:
        assert _rows(reopened, "conv") == [("user", USER["content"], "passive_conversation")]
        assert reopened.get_passive_history_watermark("conv").revision == receipt.revision
    finally:
        reopened.close()


def test_concurrent_equal_retry_commits_exactly_once(store):
    """Two writers racing the same client event: one insert, one acknowledged replay."""
    writer, peer = store
    ready = threading.Barrier(2)
    results: dict[str, object] = {}

    def submit(name, db):
        ready.wait(timeout=5)
        try:
            results[name] = _append(db, "conv", messages=[USER, ASSISTANT])
        except BaseException as exc:  # recorded, then re-asserted on the main thread
            results[name] = exc

    threads = [threading.Thread(target=submit, args=(name, db))
               for name, db in (("a", writer), ("b", peer))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    first, second = results["a"], results["b"]
    assert not isinstance(first, BaseException) and not isinstance(second, BaseException)
    assert first.message_ids == second.message_ids
    assert first.revision == second.revision
    assert sorted([first.replayed, second.replayed]) == [False, True]
    assert len(_receipt_rows(peer)) == 1
    assert len(_rows(peer, "conv")) == 2
    assert peer.get_session("conv")["message_count"] == 2
    # Exactly one FTS row per committed message: the replay must not double-index.
    assert len(peer.search_messages("ABC123")) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"messages": [{"role": "user", "content": "different words"}]}, id="changed-content"),
        pytest.param({"messages": [USER, ASSISTANT]}, id="changed-shape"),
        pytest.param({"origin_turn_id": "origin-2"}, id="changed-origin-identity"),
    ],
)
def test_equal_id_different_payload_conflicts_without_copying(store, overrides):
    writer, peer = store
    original = _append(writer, "conv", messages=[USER])
    messages = overrides.pop("messages", [USER])

    with pytest.raises(PassiveHistoryConflictError):
        _append(peer, "conv", messages=messages, **overrides)

    assert len(_receipt_rows(peer)) == 1
    assert _rows(peer, "conv") == [("user", USER["content"], "passive_conversation")]
    assert peer.get_passive_history_watermark("conv").revision == original.revision


def test_active_turn_lease_returns_retryable_busy_and_writes_nothing(store):
    """A voice commit is fenced by the live turn, never by a call-duration lease of its own."""
    writer, peer = store
    # This process is provably alive, so the guard cannot reclaim the row as a dead holder.
    holder = f"pid={os.getpid()}:turn=live"
    assert peer.try_acquire_session_turn_lease("conv", holder, ttl_seconds=300)

    with pytest.raises(PassiveHistoryBusyError) as excinfo:
        _append(writer, "conv", messages=[USER])
    assert excinfo.value.retryable is True
    assert _rows(writer, "conv") == []
    assert _receipt_rows(writer) == []
    assert writer.get_passive_history_watermark("conv").revision == 0

    peer.release_session_turn_lease("conv", holder)
    receipt = _append(writer, "conv", messages=[USER])
    assert receipt.replayed is False
    assert len(_rows(peer, "conv")) == 1


def test_stale_lease_holder_is_reclaimed_rather_than_blocking(store):
    """The guard's own reclamation rule is preserved: a dead holder must not wedge history."""
    writer, peer = store
    # pid=1 is not a live local Hermes process: reclaimable by the same rule as acquisition,
    # so this asserts the fence without depending on a TTL elapsing.
    assert peer.try_acquire_session_turn_lease("conv", "pid=1:turn=stale", ttl_seconds=300)

    receipt = _append(writer, "conv", messages=[USER])

    assert receipt.replayed is False
    assert len(_rows(peer, "conv")) == 1
    with writer._read_ctx() as conn:
        assert conn.execute(
            "SELECT 1 FROM session_turn_leases WHERE conversation_id = 'conv'").fetchone() is None


def test_receipt_insert_failure_rolls_back_rows_counters_and_watermark(store, monkeypatch):
    writer, peer = store

    def explode(_self, _conn, **_kwargs):
        raise RuntimeError("injected receipt-insert failure")

    monkeypatch.setattr(
        SessionPassiveHistoryMixin, "_insert_passive_receipt", explode, raising=True)

    with pytest.raises(RuntimeError, match="injected receipt-insert failure"):
        _append(writer, "conv", messages=[USER, ASSISTANT])

    assert _rows(peer, "conv") == []
    assert _receipt_rows(peer) == []
    assert peer.get_session("conv")["message_count"] == 0
    assert peer.get_passive_history_watermark("conv").revision == 0
    assert peer.search_messages("ABC123") == []


def test_compression_successor_replay_returns_the_original_receipt(store):
    """Continuation lineage is the same conversation; the committed segment stays original."""
    writer, peer = store
    original = _append(writer, "conv", messages=[USER])
    writer.end_session("conv", "compression")
    writer.create_session("conv-2", source="test", parent_session_id="conv")

    through_original = _append(peer, "conv", messages=[USER])
    through_successor = _append(peer, "conv-2", messages=[USER])

    for replay in (through_original, through_successor):
        assert replay.replayed is True
        assert replay.message_ids == original.message_ids
        assert (replay.session_id, replay.conversation_id) == ("conv", "conv")
    # A genuinely new event on the same conversation lands on the live tip.
    fresh = _append(writer, "conv", messages=[ASSISTANT], event_id="evt-2")
    assert fresh.session_id == "conv-2" and fresh.conversation_id == "conv"
    assert fresh.revision > original.revision
    assert peer.get_passive_history_watermark("conv-2").revision == fresh.revision


def test_in_place_compaction_retry_keeps_the_original_identity(store):
    """Compaction clones rows and mints new ids; the receipt still names the original commit."""
    writer, peer = store
    original = _append(writer, "conv", messages=[USER, ASSISTANT])
    writer.archive_and_compact("conv", [{"role": "user", "content": "summary of the exchange"}])

    replay = _append(peer, "conv", messages=[USER, ASSISTANT])

    assert replay.replayed is True and replay.message_ids == original.message_ids
    assert replay.revision == original.revision
    assert len(_receipt_rows(peer)) == 1
    # The compacted display generation is additive: original rows are retained, not deleted.
    assert ("user", USER["content"], "passive_conversation") in _rows(peer, "conv")


def test_explicit_branch_is_a_separate_owner(store):
    """A branch child never inherits or reuses the parent's dedupe authority."""
    writer, peer = store
    original = _append(writer, "conv", messages=[USER])
    writer.create_session(
        "branch", source="test", parent_session_id="conv",
        model_config={"_branched_from": "conv"})

    with pytest.raises(PassiveHistoryConflictError):
        _append(peer, "branch", messages=[USER])

    separate = _append(peer, "branch", messages=[USER], event_id="evt-branch")
    assert separate.conversation_id == "branch" and separate.session_id == "branch"
    assert separate.message_ids != original.message_ids
    assert peer.get_passive_history_watermark("conv").revision == original.revision
    assert peer.get_passive_history_watermark("branch").revision == separate.revision


@pytest.mark.parametrize(
    "destroy",
    [
        pytest.param(lambda db: db.clear_messages("conv"), id="messages-deleted"),
        pytest.param(lambda db: db.delete_session("conv"), id="session-deleted"),
    ],
)
def test_deleted_content_retires_the_identity_instead_of_replaying_it(store, destroy):
    writer, peer = store
    _append(writer, "conv", messages=[USER])

    destroy(writer)
    if peer.get_session("conv") is None:  # recreating the id must not revive deleted rows
        peer.create_session("conv", source="test")

    with pytest.raises(PassiveHistoryRetiredError):
        _append(peer, "conv", messages=[USER])

    assert _rows(peer, "conv") == []
    # The content-free tombstone survives so the identity can never be reused.
    assert len(_receipt_rows(peer)) == 1


@pytest.mark.parametrize(
    "session_id, prepare",
    [
        pytest.param("missing", lambda db: None, id="unknown-session"),
        pytest.param("conv", lambda db: db.end_session("conv", "cli_close"), id="explicitly-closed-tip"),
        pytest.param(
            "conv",
            lambda db: (db.end_session("conv", "compression"),
                        db.create_session("c-a", source="test", parent_session_id="conv"),
                        db.create_session("c-b", source="test", parent_session_id="conv")),
            id="ambiguous-continuation",
        ),
        pytest.param(
            "conv", lambda db: db.end_session("conv", "compression"), id="missing-continuation"),
    ],
)
def test_unresolvable_targets_fail_closed(store, session_id, prepare):
    writer, peer = store
    prepare(writer)

    with pytest.raises(PassiveHistoryTargetError):
        _append(writer, session_id, messages=[USER])

    assert _receipt_rows(peer) == []
    assert _rows(peer, "conv") == []


def test_watermark_reads_the_conversation_lineage_and_never_invents_a_session(store):
    writer, peer = store
    writer.end_session("conv", "compression")
    writer.create_session("conv-2", source="test", parent_session_id="conv")
    receipt = _append(writer, "conv-2", messages=[USER])

    for session_id in ("conv", "conv-2"):
        watermark = peer.get_passive_history_watermark(session_id)
        assert (watermark.conversation_id, watermark.revision) == ("conv", receipt.revision)

    with pytest.raises(PassiveHistoryTargetError):
        peer.get_passive_history_watermark("nope")


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"messages": [{"role": "system", "content": "be nice"}]}, id="system-role"),
        pytest.param({"messages": [{"role": "tool", "content": "{}"}]}, id="tool-role"),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi", "tool_calls": []}]}, id="tool-fields"),
        pytest.param(
            {"messages": [{"role": "assistant", "content": "hi", "reasoning": "why"}]}, id="reasoning"),
        pytest.param({"messages": [{"role": "user"}]}, id="missing-content"),
        pytest.param({"messages": [{"role": "user", "content": ""}]}, id="empty-content"),
        pytest.param({"messages": [{"role": "user", "content": "   \n"}]}, id="whitespace-content"),
        pytest.param({"messages": [{"role": "user", "content": "a" * (64 * 1024 + 1)}]}, id="oversized"),
        pytest.param({"messages": [{"role": "user", "content": "bad \ud800"}]}, id="lone-surrogate"),
        pytest.param(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
            id="multimodal"),
        pytest.param({"messages": [ASSISTANT, USER]}, id="wrong-pair-order"),
        pytest.param({"messages": [USER, USER]}, id="duplicate-role-pair"),
        pytest.param({"messages": [USER, ASSISTANT, USER]}, id="too-many-rows"),
        pytest.param({"messages": []}, id="no-rows"),
        pytest.param({"messages": USER}, id="not-a-list"),
        pytest.param({"messages": [USER], "producer": ""}, id="empty-producer"),
        pytest.param({"messages": [USER], "producer": "talk voice"}, id="producer-space"),
        pytest.param({"messages": [USER], "producer": "a" * 65}, id="producer-too-long"),
        pytest.param({"messages": [USER], "event_id": ""}, id="empty-event-id"),
        pytest.param({"messages": [USER], "event_id": "a" * 129}, id="event-id-too-long"),
        pytest.param({"messages": [USER], "event_id": "evt/1"}, id="event-id-charset"),
        pytest.param({"messages": [USER], "origin_turn_id": ""}, id="empty-origin"),
        pytest.param({"messages": [USER], "origin_turn_id": None}, id="origin-not-a-string"),
    ],
)
def test_invalid_submissions_are_rejected_before_any_write(store, kwargs):
    writer, peer = store

    with pytest.raises(ValueError) as excinfo:
        _append(writer, "conv", **kwargs)
    # Conflict/target errors are ValueErrors with their own meaning; a malformed payload is neither.
    assert type(excinfo.value) is ValueError

    assert _rows(peer, "conv") == []
    assert _receipt_rows(peer) == []
    assert peer.get_passive_history_watermark("conv").revision == 0


def test_repeated_same_role_finalized_turns_both_persist(store):
    """Canonical raw history keeps both finalized events; replay repair is the reader's job."""
    writer, peer = store
    first = _append(writer, "conv", messages=[USER])
    second = _append(
        writer, "conv", messages=[{"role": "user", "content": "actually make it 10am"}],
        event_id="evt-2", origin_turn_id="origin-2")

    assert second.revision > first.revision
    assert [role for role, _content, _kind in _rows(peer, "conv")] == ["user", "user"]
    assert peer.get_session("conv")["message_count"] == 2
    # Model replay reuses the host's existing repair: both texts survive in one user turn,
    # and no assistant answer is fabricated for the unanswered first turn.
    replayed = peer.get_messages_as_conversation("conv", repair_alternation=True)
    assert [m["role"] for m in replayed] == ["user"]
    assert USER["content"] in replayed[0]["content"]
    assert "actually make it 10am" in replayed[0]["content"]
