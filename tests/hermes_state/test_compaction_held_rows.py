"""Regression for #121734: a compaction archives only the rows its input held."""

from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    store = SessionDB(tmp_path / "state.db")
    store.create_session("s", source="test")
    return store


def test_foreign_gap_and_unpersisted_turn_survive(db: SessionDB) -> None:
    first = db.append_message("s", role="user", content="held first")
    foreign = db.append_message("s", role="assistant", content="foreign gap")
    last = db.append_message("s", role="user", content="held last")
    # The compressor's trailing current turn has no row id. The other surface
    # wrote the gap before the largest held id, not merely above a watermark.
    held = (first, last)
    count = db.archive_and_compact("s", [
        {"role": "user", "content": "summary of held first and last"},
        {"role": "assistant", "content": "current turn not yet durable"},
    ], watermark=last, held_row_ids=held)

    assert count == 3
    assert [m["content"] for m in db.get_messages("s")] == [
        "summary of held first and last", "current turn not yet durable", "foreign gap",
    ]
    archived = {m["id"]: m for m in db.get_messages("s", include_inactive=True)}
    assert (archived[first]["active"], archived[first]["compacted"]) == (0, 1)
    assert (archived[last]["active"], archived[last]["compacted"]) == (0, 1)
    assert (archived[foreign]["active"], archived[foreign]["compacted"]) == (0, 0)
    assert db.get_session("s")["message_count"] == count


def test_stale_held_row_refuses_atomic_commit(db: SessionDB) -> None:
    first = db.append_message("s", role="user", content="held")
    db.archive_and_compact("s", [{"role": "user", "content": "other commit"}])
    before = db.get_messages("s", include_inactive=True)
    with pytest.raises(RuntimeError, match="Held compression rows"):
        db.archive_and_compact("s", [{"role": "user", "content": "stale summary"}], held_row_ids=(first,))
    assert db.get_messages("s", include_inactive=True) == before


def test_carried_held_tail_uses_its_id_not_foreign_position(db: SessionDB) -> None:
    first = db.append_message("s", role="user", content="summarized")
    db.append_message("s", role="assistant", content="foreign gap")
    last = db.append_message("s", role="user", content="carried")
    db.archive_and_compact("s", [
        {"role": "user", "content": "summary"}, {"role": "user", "content": "carried"},
    ], held_row_ids=(first, last), tail_row_ids=(last,), tail_count=1)
    archived = {m["id"]: m for m in db.get_messages("s", include_inactive=True)}
    assert (archived[first]["active"], archived[first]["compacted"]) == (0, 1)
    assert (archived[last]["active"], archived[last]["compacted"]) == (0, 0)
    assert [m["content"] for m in db.get_messages("s")] == ["summary", "carried", "foreign gap"]
