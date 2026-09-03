"""list_recent_user_messages must skip legacy compaction handoffs (#80622).

Legacy standalone ``[CONTEXT COMPACTION — REFERENCE ONLY]`` handoffs persisted
pre-#80622 are durable ``role='user'`` rows with NO ``display_kind``, so the
SQL-side display filter cannot exclude them. Every /undo-class command pairs an
in-memory count that (post-#80622) excludes handoffs via
``is_user_originated_turn`` with this DB picker — if the picker still counted
handoffs, the on-disk soft-delete would target a different turn than the
in-memory cut (memory/disk transcript divergence).

Drives the real SQL + decode path through SessionDB.
"""

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB

HANDOFF_CONTENT = (
    f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\n"
    f"User asked: 'old task'\n\n{_SUMMARY_END_MARKER}"
)


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    yield session_db
    session_db.close()


def test_legacy_handoff_rows_are_not_recent_user_messages(db):
    db.create_session(session_id="s1", source="cli", model="m")
    db.append_message("s1", role="user", content="first question")
    db.append_message("s1", role="assistant", content="first answer")
    # Legacy shape: durable role=user handoff with NO display_kind.
    db.append_message("s1", role="user", content=HANDOFF_CONTENT)
    db.append_message("s1", role="user", content="second question")
    db.append_message("s1", role="assistant", content="second answer")

    recents = db.list_recent_user_messages("s1", limit=10)
    previews = [r["preview"] for r in recents]

    assert len(recents) == 2
    assert previews[0].startswith("second question")
    assert previews[1].startswith("first question")
    assert not any("[CONTEXT COMPACTION" in p for p in previews)


def test_handoff_skip_respects_limit_with_headroom(db):
    """The requested limit is still honored when handoff rows are dropped."""
    db.create_session(session_id="s2", source="cli", model="m")
    for i in range(3):
        db.append_message("s2", role="user", content=HANDOFF_CONTENT)
        db.append_message("s2", role="user", content=f"question {i}")

    recents = db.list_recent_user_messages("s2", limit=2)

    assert [r["preview"] for r in recents] == ["question 2", "question 1"]


def test_display_kind_rows_still_excluded(db):
    """The pre-existing SQL-side display_kind filter is unchanged."""
    db.create_session(session_id="s3", source="cli", model="m")
    db.append_message("s3", role="user", content="real question")
    db.append_message(
        "s3",
        role="user",
        content="background agent finished",
        display_kind="async_delegation_complete",
    )

    recents = db.list_recent_user_messages("s3", limit=10)

    assert [r["preview"] for r in recents] == ["real question"]
