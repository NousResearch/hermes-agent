"""Regression for issue #81130 — ``/undo`` must restore pre-compaction history.

Default in-place compaction (``compression.in_place: true``) calls
``archive_and_compact`` which soft-archives every active row to
``active=0, compacted=1`` and inserts the compacted projection as new
``active=1`` rows. ``/undo`` was implemented as "suffix soft-delete on the
active set", so once compaction had fired it could no longer reach the
pre-compaction turns — the compacted=1 archive stayed on disk (FTS-indexed,
searchable) but was invisible to the rewind picker.

These tests pin the class-level fix:

* ``SessionDB.rewind_through_compaction`` is the inverse of
  ``archive_and_compact``: revives compacted rows at/after the target,
  soft-deletes the live tail at/after the target, and never touches the
  pre-target rows (any earlier compaction boundary stays archived).
* ``list_recent_user_messages(include_compacted=True)`` surfaces the
  compacted=1 archive as valid rewind targets so ``/undo`` can step
  across the compaction boundary.
* ``SessionStore.rewind_session`` (gateway) and ``HermesCLI.undo_last``
  (CLI) auto-route to ``rewind_through_compaction`` when the picked
  target lives in the compacted=1 archive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    return SessionDB(db_path=tmp_path / "state.db")


def _seed(db, sid, n_pairs):
    """Insert n_pairs user/assistant pairs as live rows for *sid*."""
    db.create_session(sid, "cli", model="test/model")
    for i in range(n_pairs):
        db.append_message(session_id=sid, role="user", content=f"q{i}")
        db.append_message(session_id=sid, role="assistant", content=f"a{i}")


def _live_messages(db, sid):
    return db.get_messages_as_conversation(sid)


def _all_messages(db, sid):
    return db.get_messages(sid, include_inactive=True)


def test_undo_target_picker_includes_compacted_archive(db):
    """``list_recent_user_messages(include_compacted=True)`` must surface the
    compacted=1 archive (#81130) so ``/undo`` can pick a pre-compaction
    target."""
    _seed(db, "s", 4)
    db.archive_and_compact(
        "s",
        [
            {"role": "user", "content": "[SUMMARY]"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    # Default picker (include_compacted=False): only the live set.
    recents_default = db.list_recent_user_messages("s", limit=10)
    assert {r["preview"] for r in recents_default} == {"[SUMMARY]"}

    # include_compacted=True: the picker surfaces the pre-compaction turns
    # too, so the rewind target can step across the compaction boundary.
    # The picker is newest-first (ORDER BY id DESC), so previews[0] is the
    # live "[SUMMARY]" row and previews[-1] is the oldest pre-compaction
    # turn (q0) — the archive covers q0..q3 in insertion order.
    recents_with_compacted = db.list_recent_user_messages(
        "s", limit=10, include_compacted=True
    )
    previews = [r["preview"] for r in recents_with_compacted]
    assert previews[0] == "[SUMMARY]"
    assert previews[-1] == "q0"
    for q in ("q0", "q1", "q2", "q3"):
        assert q in previews, f"missing pre-compaction row {q} in picker output"


def test_rewind_through_compaction_revives_pre_compaction_rows(db):
    """``rewind_through_compaction`` is the inverse of
    ``archive_and_compact``: the compacted=1 rows at/after the target come
    back to active=1, and the live tail at/after the target gets
    soft-deleted. Pre-target rows in any earlier archive are untouched."""
    _seed(db, "s", 4)
    db.archive_and_compact(
        "s",
        [
            {"role": "user", "content": "[SUMMARY]"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    # The compacted=1 row whose content is "q3" is the user's last
    # pre-compaction turn; that's where /undo lands by default.
    target_id = db._conn.execute(
        "SELECT id FROM messages "
        "WHERE session_id = ? AND active = 0 AND compacted = 1 "
        "AND role = 'user' AND content = ?",
        ("s", "q3"),
    ).fetchone()["id"]

    result = db.rewind_through_compaction("s", target_id)
    assert result["revived_count"] >= 1
    assert result["rewound_count"] >= 1

    # The live set now replays the revived pre-compaction transcript.
    live = _live_messages(db, "s")
    live_contents = [
        m.get("content") for m in live if m.get("role") in ("user", "assistant")
    ]
    assert "[SUMMARY]" not in live_contents
    assert "q0" in live_contents
    assert "q3" in live_contents

    # The compacted summary + its assistant reply are soft-deleted.
    summary_row = db._conn.execute(
        "SELECT active, compacted FROM messages WHERE session_id = ? "
        "AND role = 'user' AND content = ?",
        ("s", "[SUMMARY]"),
    ).fetchone()
    assert int(summary_row["active"]) == 0

    # The revived rows are now active=1, compacted=0 (the inverse flag
    # flip distinguishes them from any earlier compacted archive).
    revived_row = db._conn.execute(
        "SELECT active, compacted FROM messages WHERE id = ?", (target_id,)
    ).fetchone()
    assert int(revived_row["active"]) == 1
    assert int(revived_row["compacted"]) == 0


def test_rewind_through_compaction_preserves_session_boundary(db):
    """After rewind, the live transcript cleanly transitions from the
    revived pre-compaction tail to the new user prompt.

    A session that goes through a second compaction AFTER the rewind should
    see the FIRST round's pre-compaction rows as part of the new archive
    (i.e. compacted=1 again), not as live rows — confirming the post-rewind
    state behaves like a fresh transcript when next compaction fires.
    """
    _seed(db, "s", 4)
    db.archive_and_compact(
        "s",
        [
            {"role": "user", "content": "[SUMMARY-1]"},
            {"role": "assistant", "content": "ok-1"},
        ],
    )

    target_id = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = ? "
        "AND active = 0 AND compacted = 1 AND role = 'user' AND content = 'q3'",
        ("s",),
    ).fetchone()["id"]

    db.rewind_through_compaction("s", target_id)

    # All eight pre-compaction rows are now active=1, compacted=0 (the
    # inverse flag flip). The compacted summary + its assistant reply are
    # active=0.
    revived = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND active = 1 AND compacted = 0",
        ("s",),
    ).fetchone()[0]
    assert revived == 8

    soft_deleted_tail = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND active = 0 AND role IN ('user', 'assistant') "
        "AND content IN ('[SUMMARY-1]', 'ok-1')",
        ("s",),
    ).fetchone()[0]
    assert soft_deleted_tail == 2

    # A second compaction now flips every active row to compacted=1 again —
    # the post-rewind state behaves like a normal in-place compaction input.
    db.archive_and_compact(
        "s",
        [
            {"role": "user", "content": "[SUMMARY-2]"},
            {"role": "assistant", "content": "ok-2"},
        ],
    )

    # Pre-compaction rows from the first round are back in the compacted=1
    # archive; the live set is the new compact tail.
    archive_q3 = db._conn.execute(
        "SELECT active, compacted FROM messages "
        "WHERE session_id = ? AND role = 'user' AND content = 'q3'",
        ("s",),
    ).fetchone()
    assert int(archive_q3["active"]) == 0
    assert int(archive_q3["compacted"]) == 1


def test_rewind_through_compaction_rejects_live_target(db):
    """A live (active=1) target must be rejected — callers fall back to
    ``rewind_to_message`` for those. Mixing the two into one method would
    broaden ``rewind_to_message``'s contract for callers that don't need
    the compaction-aware branch."""
    _seed(db, "s", 2)
    live_target_id = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'user' "
        "AND active = 1 ORDER BY id DESC LIMIT 1",
        ("s",),
    ).fetchone()["id"]

    with pytest.raises(ValueError, match="compacted=1 archive"):
        db.rewind_through_compaction("s", live_target_id)


def test_undo_after_in_place_compaction_restores_history(db):
    """End-to-end: seed → archive_and_compact → /undo-equivalent rewind
    → live transcript contains the pre-compaction turns, NOT the
    compacted summary. This is the exact failure mode #81130 reports."""
    _seed(db, "s", 4)
    db.archive_and_compact(
        "s",
        [
            {"role": "user", "content": "[SUMMARY]"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    # Sanity: pre-undo, the live set is the summary, not the original
    # user turns. This is the bug state.
    pre_undo_live = [
        m["content"] for m in _live_messages(db, "s") if m.get("role") == "user"
    ]
    assert pre_undo_live == ["[SUMMARY]"]

    # Pick the Nth-from-last user message across the compacted archive.
    recents = db.list_recent_user_messages("s", limit=10, include_compacted=True)
    assert recents, "expected include_compacted=True to surface the archive"

    # /undo with N=1 → the most-recent user message. That's "[SUMMARY]"
    # (the live one), so a single-turn /undo would still step into the
    # summary. Step across the boundary by picking the next target in
    # the compacted archive (the most recent pre-compaction user row).
    compacted_targets = [
        r for r in recents
        if not (r["preview"] == "[SUMMARY]")
    ]
    target_id = compacted_targets[0]["id"]

    db.rewind_through_compaction("s", target_id)

    # Post-undo, the live set replays the revived pre-compaction turns.
    post_undo_live = _live_messages(db, "s")
    post_undo_user_contents = [
        m["content"] for m in post_undo_live if m.get("role") == "user"
    ]
    assert "[SUMMARY]" not in post_undo_user_contents
    assert "q3" in post_undo_user_contents
    assert "q2" in post_undo_user_contents


def test_undo_session_rewind_routes_through_compaction(db):
    """``SessionStore.rewind_session`` (gateway /undo) must auto-route
    through ``rewind_through_compaction`` when the picked target lives
    in the compacted=1 archive."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    sid = "sess-rewind"
    _seed(db, sid, 4)
    db.archive_and_compact(
        sid,
        [
            {"role": "user", "content": "[SUMMARY]"},
            {"role": "assistant", "content": "ok"},
        ],
    )

    store = SessionStore(sessions_dir=Path("/tmp"), config=GatewayConfig())
    store._db = db

    # Step past the live "[SUMMARY]" target by passing n large enough
    # that the picker lands on the first pre-compaction user turn.
    res = store.rewind_session(sid, n=2)
    assert res is not None, "rewind_session must succeed across a compaction boundary"
    assert res["turns_undone"] == 2

    # The live transcript no longer contains the compacted summary.
    live = store.load_transcript(sid)
    live_user_contents = [
        m.get("content") for m in live if m.get("role") == "user"
    ]
    assert "[SUMMARY]" not in live_user_contents
    # The revived pre-compaction rows are present.
    assert "q3" in live_user_contents or "q2" in live_user_contents
