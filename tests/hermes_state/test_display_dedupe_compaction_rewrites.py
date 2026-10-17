"""#135655: a compaction-rewritten copy must not paint a second display generation.

In-place compaction re-inserts the compacted set as fresh live rows. A carried
copy whose content was rewritten (a pruned tool-result digest, a merge carrier)
no longer shares its durable original's display identity, so
``_dedupe_display_generations`` cannot fold the two and the display read paints
the logical message twice: the archived original plus its rewrite. The commit
marks such copies model-only (``display_metadata.model_only``), mirroring
``micro_compaction._merge_adjacent_user_turns``: the model still reads the
rewrite, display projections paint the durable original only. Both commit paths
(the watermark commit and the proved-coverage commit) mark alike.

The carried-tail case is pinned unchanged: when the original is rewind-stamped
(``active=0, compacted=0`` - hidden from display), the copy is the only visible
generation and must stay unmarked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path: Path) -> SessionDB:
    instance = SessionDB(tmp_path / "state.db")
    yield instance
    instance.close()


FULL_OUTPUT = "FULL OUTPUT " + "X" * 3000
DIGEST = "[terminal] ran `` -> exit ?, 1 lines output"


def _seed_head_carry(db: SessionDB, sid: str) -> None:
    """Five rows whose carried copies the prune-style rewrite touches at index 2."""
    db.create_session(sid, source="test")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "start", "timestamp": 100.0},
        {"role": "assistant", "content": "earlier progress", "timestamp": 101.0,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "content": FULL_OUTPUT, "tool_call_id": "c1",
         "tool_name": "terminal", "timestamp": 102.0},
        {"role": "user", "content": "after", "timestamp": 103.0},
        {"role": "assistant", "content": "later answer", "timestamp": 104.0},
    ])


def _rewritten_history(db: SessionDB, sid: str) -> list[dict]:
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[2]["content"] = DIGEST
    return history


def _display(db: SessionDB, sid: str) -> list:
    return [m["content"] for m in
            db.get_messages_as_conversation(sid, include_compacted=True)]


def test_rewritten_carry_does_not_paint_beside_its_original(db: SessionDB) -> None:
    """The #135655 head-carry arm: rewrite the carried tool result the way the
    prune does, commit, and the display must paint the archived full output
    ONCE - not the full output plus its digest. The model read and the row-level
    projections are pinned alongside: the rewrite stays in the model's set but
    out of every display projection."""
    sid = "rewritten-carry"
    _seed_head_carry(db, sid)
    db.archive_and_compact(sid, _rewritten_history(db, sid))

    assert _display(db, sid) == [
        "start", "earlier progress", FULL_OUTPUT, "after", "later answer"], (
        "the archived original is the visible generation")
    live = [m["content"] for m in db.get_messages_as_conversation(sid)]
    assert live == ["start", "earlier progress", DIGEST, "after", "later answer"], (
        "model-only is a display-projection flag, not an archive: the model still reads the rewrite")
    rows = db.get_messages(sid, include_inactive=True)
    digest_rows = [r for r in rows if r["content"] == DIGEST]
    assert len(digest_rows) == 1
    assert (digest_rows[0].get("display_metadata") or {}).get("model_only") is True, (
        "the rewritten copy must be marked model-only")
    assert db.display_message_count(sid) == 5
    assert len(db.get_messages(sid, include_compacted=True)) == 5


def test_proved_coverage_commit_marks_the_same_way(db: SessionDB) -> None:
    """Both commit paths mark: naming every held row as covered takes
    ``_archive_named_rows`` instead of the watermark path, and the rewritten
    copy must stay out of the display there too."""
    sid = "rewritten-carry-proved"
    _seed_head_carry(db, sid)
    history = _rewritten_history(db, sid)
    covered = [row["_row_id"] for row in history if row.get("_row_id")]
    db.archive_and_compact(sid, history, covered_ids=covered)

    assert _display(db, sid) == [
        "start", "earlier progress", FULL_OUTPUT, "after", "later answer"], (
        "the proved-coverage commit must fold the rewritten copy the same way")
    rows = db.get_messages(sid, include_inactive=True)
    digest_rows = [r for r in rows if r["content"] == DIGEST]
    assert len(digest_rows) == 1
    assert (digest_rows[0].get("display_metadata") or {}).get("model_only") is True


def test_rewritten_tail_copy_paints_when_its_original_is_rewound(db: SessionDB) -> None:
    """tail_count>0 rewinds the carried originals (hidden from display); their
    rewritten copies are the only visible generation and must stay unmarked."""
    sid = "tail-rewrite"
    db.create_session(sid, source="test")
    for i in range(6):
        db.append_message(sid, role="user" if i % 2 == 0 else "assistant", content=f"turn {i}")
    history = db.get_messages_as_conversation(sid, include_row_ids=True)
    history[-1]["content"] = "turn 5 (rewritten)"
    summary = [{"role": "user", "content": "[CONTEXT COMPACTION] summary of earlier turns"},
               {"role": "assistant", "content": "Continuing from the summary."}]
    db.archive_and_compact(sid, [*summary, history[-2], history[-1]], tail_count=2)

    assert _display(db, sid) == ["turn 0", "turn 1", "turn 2", "turn 3",
                                 "[CONTEXT COMPACTION] summary of earlier turns",
                                 "Continuing from the summary.",
                                 "turn 4", "turn 5 (rewritten)"]
    rows = db.get_messages(sid, include_inactive=True)
    rewrite = [r for r in rows if r["content"] == "turn 5 (rewritten)"]
    assert len(rewrite) == 1
    assert not (rewrite[0].get("display_metadata") or {}).get("model_only"), (
        "a copy whose original was rewound is the only visible generation")
    original = [r for r in rows if r["content"] == "turn 5"]
    assert len(original) == 1 and original[0]["active"] == 0 and original[0]["compacted"] == 0
