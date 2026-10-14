"""Tests for issue #138355 — the in-place compaction paths ran
``_carry_parent_timestamps`` AFTER the rewind/archive that deactivates every
donor row, so carried copies rebuilt from durable state (dicts without a
``timestamp`` key) were stamped ``now`` instead of adopting the durable
original's timestamp. Their ``_display_dedupe_key`` then differed from the
archived original's and the display projection rendered the same logical
message twice.

Pinned here for both in-place paths (``archive_and_compact``'s watermark
branch and the ``_archive_named_rows`` proved-coverage branch): rebuilt
copies adopt the durable timestamp, the display projection stays
deduplicated, and a dict that already carries a timestamp is still never
overwritten (the carry keeps its never-overwrite contract #59661).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB

ROWS = (
    ("user", "u0", 1000.0),
    ("assistant", "a0", 1010.0),
    ("user", "keep me", 1015.0),
    ("assistant", "tail-a", 1020.0),
)
DURABLE_TS = {text: ts for _, text, ts in ROWS}


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("s", source="test")
    for role, text, ts in ROWS:
        d.append_message("s", role=role, content=text, timestamp=ts)
    return d


def _rebuilt() -> list[dict]:
    """The rebuild shape: fresh dicts from durable state, no ``timestamp`` key."""
    return [{"role": role, "content": text} for role, text, _ in ROWS]


def _visible(db: SessionDB) -> list[dict]:
    return db.get_messages("s", include_compacted=True)


def _ts_by_content(db: SessionDB) -> dict[str, float]:
    return {m["content"]: m["timestamp"] for m in _visible(db)}


def test_rebuilt_copies_adopt_durable_timestamps_archive_all(db):
    """Watermark branch, rewind_ids empty: the archive-all still leaves donors visible to the carry."""
    db.archive_and_compact("s", _rebuilt())

    assert [m["content"] for m in _visible(db)] == [text for _, text, _ in ROWS]
    assert _ts_by_content(db) == DURABLE_TS


def test_rebuilt_copies_adopt_durable_timestamps_tail_rewind(db):
    """Watermark branch, rewind_ids non-empty: tail originals are rewound before the carry too."""
    db.archive_and_compact("s", _rebuilt(), tail_count=2)

    assert [m["content"] for m in _visible(db)] == [text for _, text, _ in ROWS]
    assert _ts_by_content(db) == DURABLE_TS


def test_rebuilt_copies_adopt_durable_timestamps_proved_coverage(db):
    """Proved-coverage branch (_archive_named_rows): same ordering requirement."""
    active_ids = [row[0] for row in db._conn.execute(
        "SELECT id FROM messages WHERE session_id = 's' AND active = 1 ORDER BY id").fetchall()]
    db.archive_and_compact("s", _rebuilt(), covered_ids=active_ids, tail_count=2)

    assert [m["content"] for m in _visible(db)] == [text for _, text, _ in ROWS]
    assert _ts_by_content(db) == DURABLE_TS


def test_explicit_timestamp_still_not_overwritten_inplace(db):
    """A dict that carries its own timestamp keeps it: the earlier carry stays authoritative."""
    messages = _rebuilt()
    messages[2]["timestamp"] = 5555.5
    db.archive_and_compact("s", messages)

    keeps = [m for m in _visible(db) if m["content"] == "keep me"]
    assert {m["timestamp"] for m in keeps} == {1015.0, 5555.5}
