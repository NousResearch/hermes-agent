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

from agent.context_compressor import (
    COMPRESSED_SUMMARY_HAS_USER_TURN_KEY,
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
)
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


def _seed_and_compact(db, sid, n_pairs, summary_content, tail_pairs=()):
    """Seed *sid* with ``n_pairs`` user/assistant pairs, then archive them and
    insert a compaction projection whose FIRST row is the synthetic summary.

    The projection mirrors a real in-place compaction: it leads with a
    ``role='user'`` summary carrying the in-process compressed-summary
    metadata flag (persisted in-memory only — SessionDB drops underscore keys,
    so the persisted discriminator is the ``display_kind`` stamp), optionally
    an assistant carry-over, then preserved real tail turns.
    """
    _seed(db, sid, n_pairs)
    projection = [
        {
            "role": "user",
            "content": summary_content,
            COMPRESSED_SUMMARY_METADATA_KEY: True,
            COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: False,
        }
    ]
    if tail_pairs:
        # Each pair is inserted as (assistant carry-over, preserved user turn,
        # assistant reply) — matching a real compaction projection's tail.
        for q, a in tail_pairs:
            projection.append({"role": "assistant", "content": "carry-over"})
            projection.append({"role": "user", "content": q})
            projection.append({"role": "assistant", "content": a})
    else:
        projection.append({"role": "assistant", "content": "ok"})
    db.archive_and_compact(sid, projection)
    return projection


def test_synthetic_summary_row_persists_display_kind(db):
    """A user-role compaction summary persisted through ``archive_and_compact``
    must carry a stable ``display_kind`` marker (#81130 root-cause 3).

    The in-process ``_compressed_summary`` flag is underscore-prefixed and
    deliberately stripped before the wire / not persisted by SessionDB, so the
    ONLY durable signal distinguishing a synthetic summary from a real user
    turn is the ``display_kind`` column. Without it the /undo picker's SQL
    ``display_clause`` (``display_kind IS NULL``) counts the summary as a real
    turn and /undo N can land on it — the empty-context failure the issue
    reports.
    """
    # Content intentionally does NOT start with any known handoff prefix so the
    # content-prefix heuristic (classify_summary_content) cannot be what makes
    # this pass — only the display_kind stamp can.
    _seed_and_compact(
        db, "s", 4, "Compressed earlier turns. Resume from here."
    )
    row = db._conn.execute(
        "SELECT display_kind FROM messages WHERE session_id = ? "
        "AND role = 'user' AND active = 1 AND "
        "content = 'Compressed earlier turns. Resume from here.'",
        ("s",),
    ).fetchone()
    assert row is not None
    assert row["display_kind"] == "hidden"

    # The picker (with include_compacted=True, as /undo uses) must NOT surface
    # the summary — even though its content fails every prefix heuristic, the
    # persisted display_kind excludes it at the SQL display_clause.
    recents = db.list_recent_user_messages("s", limit=10, include_compacted=True)
    previews = [r["preview"] for r in recents]
    assert "Compressed earlier turns" not in " ".join(previews)
    # The preserved tail / archived real turns are still valid targets.
    for q in ("q3", "q0"):
        assert any(q in p for p in previews), f"missing real turn {q} in picker"


def test_undo_multiple_turns_never_lands_on_synthetic_summary(db):
    """``/undo N`` counting across the compaction boundary must never pick the
    synthetic user-role summary as a target — N resolves against REAL user
    turns only (#81130 root-cause 3).

    This is the issue's exact failure mode: with the summary occupying the
    newest live user slot, ``/undo 3`` used to count it and drop to an empty
    live context. Here the summary carries the real ``SUMMARY_PREFIX`` banner
    (what production compaction emits), so it is double-excluded: the content
    heuristic AND the persisted display_kind. The regression is that the
    3rd-from-last user turn is a REAL archived turn, and rewinding to it
    revives the pre-compaction transcript instead of draining the context.
    """
    _seed_and_compact(
        db,
        "s6",
        6,
        SUMMARY_PREFIX + " q1..q4",
        tail_pairs=[("q5", "a5"), ("q6: trigger", "a6")],
    )
    # Sanity: pre-undo the live set is the projection.
    live = _live_messages(db, "s6")
    assert any(SUMMARY_PREFIX in (m.get("content") or "") for m in live)

    recents = db.list_recent_user_messages("s6", limit=10, include_compacted=True)
    summary_in_picker = any(
        (r["preview"] or "").startswith(SUMMARY_PREFIX[:12]) for r in recents
    )
    assert not summary_in_picker, "summary must never be a /undo target"

    # /undo 3 → the third-from-last real user turn across the boundary. The
    # summary is excluded, so this lands on a compacted=1 archive row (a real
    # pre-compaction turn), which must route through rewind_through_compaction
    # rather than soft-deleting the summary into an empty context.
    target = recents[2]
    archived = db._conn.execute(
        "SELECT active, compacted FROM messages WHERE id = ?",
        (target["id"],),
    ).fetchone()
    assert int(archived["active"]) == 0
    assert int(archived["compacted"]) == 1

    db.rewind_through_compaction("s6", target["id"])
    post_live = [
        m.get("content") for m in _live_messages(db, "s6") if m.get("role") == "user"
    ]
    # The summary is gone from the live set; a real pre-compaction turn is back.
    assert not any(
        SUMMARY_PREFIX in (c or "") for c in post_live
    ), "summary must be discarded on a cross-boundary /undo"
    assert any(
        c in ("q0", "q1", "q2", "q3", "q4") for c in post_live
    ), "rewind must restore a pre-compaction real user turn"


def test_rewind_through_compaction_limits_to_nearest_boundary(db):
    """After SEVERAL in-place compactions, ``rewind_through_compaction`` must
    revive only the boundary the target sits in — not every historical
    boundary's accumulated ``compacted=1`` rows (#81130 LOW).

    Two compactions stack two boundaries:
      B1 = original turns [q0,a0,q1,a1]  (ids 1-4)
      B2 = first projection [S1, ok1]    (ids 5-6)
      live = second projection [S2, ok2] (ids 7-8)
    Rewinding to a turn in B1 must revive ONLY ids 1-4; B2 stays archived. If
    the revive swept the whole archive it would pull B2 back too and blow up
    the next context window.
    """
    _seed_and_compact(db, "s", 2, SUMMARY_PREFIX + " S1")   # B1 archived, B2 live
    db.archive_and_compact(  # second compaction: B2 archived, live = [S2, ok2]
        "s",
        [
            {
                "role": "user",
                "content": SUMMARY_PREFIX + " S2",
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: False,
            },
            {"role": "assistant", "content": "ok2"},
        ],
    )

    # Sanity: the accumulated archive has both B1 and B2.
    compacted_all = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND active = 0 AND compacted = 1",
        ("s",),
    ).fetchone()[0]
    assert compacted_all == 6

    q0_id = db._conn.execute(
        "SELECT id FROM messages WHERE session_id = ? AND content = 'q0'",
        ("s",),
    ).fetchone()["id"]
    res = db.rewind_through_compaction("s", q0_id)
    # Only the first boundary (4 rows) came back.
    assert res["revived_count"] == 4

    revived_first_boundary = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND active = 1 AND compacted = 0 AND id <= 4",
        ("s",),
    ).fetchone()[0]
    assert revived_first_boundary == 4

    # The second boundary (first projection) stays archived — not revived.
    second_boundary_still_archived = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND compacted = 1 AND id IN (5, 6)",
        ("s",),
    ).fetchone()[0]
    assert second_boundary_still_archived == 2

    # The live projection was soft-deleted (rewound).
    projection_rewound = db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? "
        "AND active = 0 AND compacted = 0 AND id IN (7, 8)",
        ("s",),
    ).fetchone()[0]
    assert projection_rewound == 2


def _make_cli(db, sid, history):
    """Bare :class:`cli.HermesCLI` with just the attributes ``undo_last``
    touches (pattern from tests/cli/test_cli_copy_command.py)."""
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._session_db = db
    cli.session_id = sid
    cli.agent = None
    cli.conversation_history = history
    return cli


def test_cli_undo_last_reloads_pre_compaction_history(db, capsys):
    """CLI ``/undo`` across a compaction boundary (#81130) — the three novel
    pieces of ``HermesCLI.undo_last``:

    * the in-memory ``conversation_history`` is reloaded from the DB after
      ``rewind_through_compaction`` revives the pre-compaction rows (the
      stale truncated slice must NOT survive);
    * the ``boundary_note`` announcing the revived rows is printed;
    * the ``remaining`` count reflects the reloaded transcript.
    """
    sid = "sess-cli-undo"
    # display_kind-stamped compaction summary (production-faithful) so the
    # /undo picker excludes it and N resolves against real pre-compaction
    # turns only.
    _seed_and_compact(db, sid, 4, "Compressed earlier turns. Resume here.")

    # Pre-undo sanity: the live set is the compacted summary, not the
    # original turns — the bug state.
    pre_undo_live = [
        m["content"] for m in _live_messages(db, sid) if m.get("role") == "user"
    ]
    assert pre_undo_live == ["Compressed earlier turns. Resume here."]

    # The CLI's in-memory transcript is stale — it still holds the
    # pre-compaction turns while the DB has already been compacted. That is
    # exactly the state undo_last's post-compaction reload exists to heal:
    # after the rewind, the revived rows "aren't represented in memory at
    # all" (cli.py undo_last comment).
    history = []
    for i in range(4):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})

    cli = _make_cli(db, sid, history)
    # n=2 steps past the live compaction summary and lands on the first
    # pre-compaction turn in the include_compacted picker (q2), routing
    # through rewind_through_compaction.
    cli.undo_last(n=2, prefill=False)

    # 1) conversation_history reloaded from the DB: the revived
    #    pre-compaction turns are present, the compacted summary is not.
    user_contents = [
        m["content"] for m in cli.conversation_history if m.get("role") == "user"
    ]
    assert user_contents == ["q0", "q1", "q2", "q3"]
    assert "Compressed earlier turns" not in " ".join(user_contents)
    assert len(cli.conversation_history) == 8

    # 2) boundary_note present in the printed output.
    out = capsys.readouterr().out
    assert "revived 8 pre-compaction row(s)" in out
    assert "compaction summary discarded" in out

    # 3) remaining count matches the reloaded transcript.
    assert "8 message(s) remaining in history." in out

