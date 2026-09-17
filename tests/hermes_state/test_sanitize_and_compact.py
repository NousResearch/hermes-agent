"""Durable sanitation commit invariants."""

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionCompressionInProgressError, SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    handle = SessionDB(tmp_path / "state.db")
    handle.create_session("sess1", source="test")
    return handle


def test_sanitation_purges_rewind_inactive_rows_and_fts(db: SessionDB) -> None:
    for content in ("keep", "reply", "secret-token-alpha", "secret-token-beta"):
        db.append_message("sess1", role="user", content=content)
    secret_row = db.get_messages("sess1")[2]["id"]
    db.rewind_to_message("sess1", secret_row)
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "clean"}],
        watermark=watermark,
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1", include_inactive=True)] == [
        "clean"
    ]
    assert db.search_messages("secret-token", include_inactive=True) == []


def test_sanitation_refuses_a_vanished_session(db: SessionDB) -> None:
    db.append_message("sess1", role="user", content="original")
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    assert db.delete_session("sess1")

    with pytest.raises(ValueError, match="Session not found"):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "orphan"}],
            watermark=watermark,
            lock_holder="sanitizer",
        )

    assert db.message_count("sess1") == 0


def test_sanitation_transaction_failure_restores_rows_and_fts(
    db: SessionDB, monkeypatch
) -> None:
    db.append_message("sess1", role="user", content="secret-token")
    watermark = db.get_active_message_watermark("sess1")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    monkeypatch.setattr(
        db,
        "_insert_message_rows",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected")),
    )

    with pytest.raises(RuntimeError, match="injected"):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "clean"}],
            watermark=watermark,
            lock_holder="sanitizer",
        )

    assert [row["content"] for row in db.get_messages("sess1")] == [
        "secret-token"
    ]
    assert len(db.search_messages("secret-token", include_inactive=True)) == 1


def test_sanitation_lost_lease_cannot_mutate_transcript(db: SessionDB) -> None:
    db.append_message("sess1", role="user", content="original")
    watermark = db.get_active_message_watermark("sess1")

    with pytest.raises(SessionCompressionInProgressError):
        db.sanitize_and_compact(
            "sess1",
            [{"role": "user", "content": "clean"}],
            watermark=watermark,
            lock_holder="not-held",
        )

    assert [row["content"] for row in db.get_messages("sess1")] == ["original"]


def test_sanitation_preserves_active_rows_absent_from_noncontiguous_snapshot(
    db: SessionDB,
) -> None:
    first = db.append_message("sess1", role="user", content="represented-first")
    db.append_message("sess1", role="assistant", content="concurrent-middle")
    last = db.append_message("sess1", role="user", content="represented-last")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "clean-first"},
            {"role": "user", "content": "clean-last"},
        ],
        watermark=last,
        represented_row_ids=(first, last),
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1")] == [
        "clean-first",
        "concurrent-middle",
        "clean-last",
    ]
    assert [row["role"] for row in db.get_messages("sess1")] == [
        "user",
        "assistant",
        "user",
    ]


def test_sanitation_accepts_large_represented_row_id_sets(db: SessionDB) -> None:
    represented = [
        db.append_message("sess1", role="user", content=f"represented-{idx}")
        for idx in range(1_005)
    ]
    db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 600)
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "clean"}],
        watermark=represented[-1],
        represented_row_ids=tuple(represented),
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1")] == ["clean"]


def test_sanitation_accepts_large_absent_tail_row_sets(db: SessionDB) -> None:
    first = db.append_message("sess1", role="user", content="represented-first")
    concurrent_rows = [
        db.append_message(
            "sess1",
            role="user",
            content=f"concurrent-{index:04d}",
        )
        for index in range(1_005)
    ]
    last = db.append_message("sess1", role="assistant", content="represented-last")
    db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 600)
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "clean-first"},
            {"role": "assistant", "content": "clean-last"},
        ],
        watermark=last,
        represented_row_ids=(first, last),
        lock_holder="sanitizer",
    )

    contents = [row["content"] for row in db.get_messages("sess1")]
    assert contents[0] == "clean-first"
    assert contents[-1] == "clean-last"
    assert contents[1:-1] == [f"concurrent-{index:04d}" for index in range(1_005)]
    assert concurrent_rows


def test_sanitation_does_not_insert_ephemeral_recovery_scaffolding(db: SessionDB) -> None:
    row_id = db.append_message("sess1", role="user", content="durable")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "clean"},
            {
                "role": "assistant",
                "content": "transient retry",
                "_thinking_prefill": True,
            },
        ],
        watermark=row_id,
        represented_row_ids=(row_id,),
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1")] == ["clean"]


def test_sanitation_places_post_snapshot_rows_after_unpersisted_live_tail(
    db: SessionDB,
) -> None:
    """A concurrent append after the snapshot must follow the whole candidate."""
    represented = db.append_message("sess1", role="user", content="represented")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    db.append_message("sess1", role="assistant", content="post-snapshot")

    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "clean-represented"},
            {"role": "user", "content": "live-unpersisted"},
        ],
        watermark=represented,
        represented_row_ids=(represented,),
        lock_holder="sanitizer",
    )

    assert [row["content"] for row in db.get_messages("sess1")] == [
        "clean-represented",
        "live-unpersisted",
        "post-snapshot",
    ]


def test_sanitation_preserves_concurrent_display_metadata_on_represented_rows(
    db: SessionDB,
) -> None:
    """Reactions and display kind written during sanitation must survive rewrite."""
    row_id = db.append_message("sess1", role="user", content="secret-token")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    assert db.set_latest_matching_message_display_kind(
        "sess1",
        role="user",
        content="secret-token",
        display_kind="steer",
        display_metadata={"source": "gateway"},
    )
    assert db.set_message_reaction("sess1", row_id, "👍", author="user")

    db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "clean"}],
        watermark=row_id,
        represented_row_ids=(row_id,),
        lock_holder="sanitizer",
    )

    published = db.get_messages_as_conversation("sess1")
    assert published[0]["content"] == "clean"
    assert published[0]["display_kind"] == "steer"
    assert published[0]["display_metadata"]["source"] == "gateway"
    assert published[0]["display_metadata"]["reactions"][0]["emoji"] == "👍"


def test_sanitation_accepts_empty_durable_prefix_watermark(db: SessionDB) -> None:
    assert db.get_active_message_watermark("sess1") == 0
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    inserted = db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "live-only"}],
        watermark=0,
        represented_row_ids=(),
        lock_holder="sanitizer",
    )

    assert inserted == 1
    assert [row["content"] for row in db.get_messages("sess1")] == ["live-only"]


def test_sanitation_live_cleared_display_fields_replace_stale_candidate(
    db: SessionDB,
) -> None:
    row_id = db.append_message(
        "sess1",
        role="user",
        content="secret-token",
        display_kind="steer",
        display_metadata={"reactions": [{"emoji": "👍"}]},
    )
    candidate = [{
        "role": "user",
        "content": "clean",
        "display_kind": "steer",
        "display_metadata": {"reactions": [{"emoji": "👍"}]},
    }]
    db._write_sql(
        "UPDATE messages SET display_kind = NULL, display_metadata = NULL WHERE id = ?",
        (row_id,),
    )
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        candidate,
        watermark=row_id,
        represented_row_ids=(row_id,),
        lock_holder="sanitizer",
    )

    published = db.get_messages_as_conversation("sess1")[0]
    assert published.get("display_kind") is None
    assert published.get("display_metadata") is None


def test_sanitation_represents_merged_rows_from_durable_membership(db: SessionDB) -> None:
    """Alternation repair merged two durable rows: BOTH must be represented (round-6 finding).

    The live candidate is a stale witness — if membership came from it, a candidate that dropped
    the merged message would leave the second source row unrepresented and its byte-exact
    UNSANITIZED content would be re-cloned (secret intact, FTS re-indexed).
    """
    db.append_message("sess1", role="user", content="first ask")
    db.append_message("sess1", role="assistant", content="reply")
    merged_first = db.append_message(
        "sess1", role="user", content="unanswered turn"
    )
    secret_row = db.append_message(
        "sess1", role="user", content="password=hostpw7"
    )
    db.append_message("sess1", role="assistant", content="next reply")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    # The LIVE candidate dropped the merged message entirely (stale witness shape).
    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "first ask"},
            {"role": "assistant", "content": "reply"},
            {"role": "assistant", "content": "next reply"},
        ],
        watermark=secret_row,
        represented_row_ids=(1, 2, merged_first, secret_row, secret_row + 1),
        member_row_ids=((1,), (2,), (merged_first, secret_row), (secret_row + 1,)),
        lock_holder="sanitizer",
    )

    contents = [row["content"] for row in db.get_messages("sess1")]
    assert "password=hostpw7" not in "".join(contents)
    assert len(db.search_messages("hostpw7", include_inactive=True)) == 0


def test_sanitation_member_groups_fall_back_to_snapshot_ids_per_slot(db: SessionDB) -> None:
    """A group with no still-active ids falls back to the slot's snapshot id (per-slot)."""
    only = db.append_message("sess1", role="user", content="represented")
    concurrent = db.append_message("sess1", role="assistant", content="concurrent")
    assert db.try_acquire_compression_lock("sess1", "sanitizer")

    db.sanitize_and_compact(
        "sess1",
        [{"role": "user", "content": "clean"}],
        watermark=only,
        represented_row_ids=(only,),
        member_row_ids=((999,),),  # stale id no longer active
        lock_holder="sanitizer",
    )

    contents = [row["content"] for row in db.get_messages("sess1")]
    # The stale group must NOT orphan the snapshot row: it is still represented...
    assert "represented" not in contents[0] or contents[0] == "clean"
    # ...the concurrent row is re-cloned after the candidate as usual...
    assert contents[-1] == "concurrent"
    assert db.search_messages("represented", include_inactive=True) == []


def test_repaired_transcript_row_ids_carry_merged_membership(db) -> None:
    """A repair-alternation restore publishes which rows each survivor MERGED (round-6 finding)."""
    db.append_message("sess1", role="user", content="first ask")
    db.append_message("sess1", role="assistant", content="first reply")
    merged_first = db.append_message("sess1", role="user", content="unanswered turn")
    merged_second = db.append_message(
        "sess1", role="user", content="password=hostpw7 and next turn"
    )
    db.append_message("sess1", role="assistant", content="next reply")

    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]
    merged = messages[2]
    assert merged["_row_id"] == merged_first
    assert merged["_merged_row_ids"] == [merged_first, merged_second]


def test_repaired_transcript_row_ids_carry_dropped_leading_membership(db) -> None:
    """A repair that drops a LEADING row must still carry its durable membership (round-7 finding).

    A stray tool result leading a resumed transcript is dropped by the repair; the first
    survivor's ``_row_id`` differs from ``pre_repair_row_ids[0]``, so the position-aligned
    walk bails before any stamping and the dropped durable row looks absent from the
    snapshot. ``sanitize_and_compact`` then clones it byte-for-byte — the secret stays in
    SQLite and stays FTS-indexed. The dropped-leading ids must ride the FIRST survivor's
    membership group.
    """
    db.append_message(
        "sess1",
        role="tool",
        tool_call_id="call-stray-orphan",
        content="password=leadpw1 in stray result",
    )
    db.append_message("sess1", role="assistant", content="first reply")
    secret_row = db.append_message("sess1", role="user", content="password=hostpw7")
    db.append_message("sess1", role="assistant", content="next reply")

    messages = db.get_messages_as_conversation(
        "sess1", repair_alternation=True, include_row_ids=True
    )
    assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
    assert messages[0]["_row_id"] != 1  # the leading row was dropped by the repair
    assert messages[0]["_merged_row_ids"] == [1, messages[0]["_row_id"]]

    # Membership derived from this snapshot covers the dropped leading row: a sanitation
    # commit that sanitizes the surviving assistant message must purge the dropped row's
    # secret too. Call sanitize_and_compact exactly the way the sanitation host does —
    # plan row membership comes from the snapshot, never re-derived per position.
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "password=hostpw7"},
            {"role": "assistant", "content": "next reply"},
        ],
        watermark=secret_row,
        represented_row_ids=(1, 2, secret_row, secret_row + 1),
        member_row_ids=((1, 2), (secret_row,), (secret_row + 1,)),
        lock_holder="sanitizer",
    )

    contents = [row["content"] for row in db.get_messages("sess1")]
    assert len(contents) == 3
    assert "password=leadpw1" not in "".join(contents), (
        "the dropped leading durable row was re-cloned byte-exact instead of "
        "being represented by the first survivor's membership group"
    )
    assert len(db.search_messages("leadpw1", include_inactive=True)) == 0


def test_sanitation_refuses_when_filtering_leaves_represented_set_empty(
    db: SessionDB,
) -> None:
    """Round-2 finding 4041479204: after an in-place compaction rewrites rows
    with NEW ids, a caller still holding the OLD snapshot ids must be refused,
    not re-clone the whole transcript byte-exact. The represented-row filter
    drops every stale id against the current active set; if the filtered set
    comes back EMPTY while active rows exist, the snapshot no longer describes
    this transcript — fail closed like the empty-tuple gate."""
    first = db.append_message("sess1", role="user", content="password=orig-pw-1")
    second = db.append_message("sess1", role="assistant", content="reply-1")

    # First commit: the snapshot ids are the real row ids and the commit lands,
    # rewriting every row with a fresh id (row ids rotate on an in-place
    # rewrite).
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    db.sanitize_and_compact(
        "sess1",
        [
            {"role": "user", "content": "clean-first"},
            {"role": "assistant", "content": "reply-1"},
        ],
        watermark=second,
        represented_row_ids=(first, second),
        lock_holder="sanitizer",
    )
    db.release_compression_lock("sess1", "sanitizer")

    rotated = [row["id"] for row in db.get_messages("sess1", include_inactive=True)]
    assert rotated and set(rotated).isdisjoint({first, second}), (
        "the committed rows must carry fresh ids (in-place rewrite)"
    )

    # Second commit with the STALE pre-rewrite ids: every id filters out
    # against the current active set, so the represented set is empty while
    # active rows exist — refuse instead of re-cloning.
    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    with pytest.raises(ValueError, match="filtered represented set empty"):
        db.sanitize_and_compact(
            "sess1",
            [
                {"role": "user", "content": "clean-again"},
                {"role": "assistant", "content": "reply-1"},
            ],
            watermark=second,
            represented_row_ids=(first, second),
            lock_holder="sanitizer",
        )
    contents = [row["content"] for row in db.get_messages("sess1")]
    assert contents == ["clean-first", "reply-1"], (
        "a refused commit must not delete or duplicate any durable row"
    )


def test_sanitation_represents_concurrently_persisted_snapshot_tail_by_content(
    db: SessionDB,
) -> None:
    """Round-2 finding 4041479212: the current user message persisted by a
    close-flush WHILE turn-start sanitation is in flight lands a durable row
    the snapshot's membership cannot name (the snapshot captured no row id for
    it). Left as-is the row is cloned byte-exact at the END of the transcript —
    the message appears twice and any secret in the concurrent bytes stays in
    SQLite/FTS. The row must be matched against the candidate's own
    unpersisted tail by durable content identity and represented-by-that-slot
    instead."""
    old_row = db.append_message("sess1", role="user", content="password=orig-pw-4")
    db.append_message("sess1", role="assistant", content="old reply")
    watermark = db.get_active_message_watermark("sess1")

    # The sanitation snapshot: history (persisted) + the current turn's user
    # message, which is NOT yet persisted (no row id).
    candidate = [
        {"role": "user", "content": "clean"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "fresh concurrent ask"},
    ]

    assert db.try_acquire_compression_lock("sess1", "sanitizer")
    # The close-flush lands the current ask while sanitation is in flight: a
    # new active row above the watermark the snapshot never named.
    concurrent_row = db.append_message(
        "sess1", role="user", content="fresh concurrent ask"
    )
    assert concurrent_row > watermark

    db.sanitize_and_compact(
        "sess1",
        candidate,
        watermark=watermark,
        represented_row_ids=(old_row, old_row + 1),
        member_row_ids=((old_row,), (old_row + 1,), ()),
        lock_holder="sanitizer",
    )

    contents = [row["content"] for row in db.get_messages("sess1")]
    assert contents == ["clean", "old reply", "fresh concurrent ask"], (
        "the concurrently persisted snapshot-tail message must appear exactly "
        "once (from the candidate slot), not re-cloned after it"
    )
    assert len(db.search_messages("fresh concurrent ask")) == 1
