"""Regression for #138355: in-place compaction retains display identity."""

from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    store.create_session("session", source="test")
    return store


@pytest.mark.parametrize("named_coverage", [False, True])
def test_in_place_compaction_carries_timestamp_before_archiving_donors(db, named_coverage):
    """Rebuilt copies share the archived row's display group in both write paths."""
    original = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    db.append_messages_batch("session", original)
    # Rebuilds loaded durable content without retaining the writer's timestamp.
    compacted = [{"role": message["role"], "content": message["content"]} for message in original]
    kwargs = {}
    if named_coverage:
        kwargs["covered_ids"] = [message["id"] for message in db.get_messages("session")]

    db.archive_and_compact("session", compacted, **kwargs)

    visible = db.get_messages("session", include_compacted=True)
    assert [(message["role"], message["content"]) for message in visible] == [
        ("user", "question"),
        ("assistant", "answer"),
    ]
