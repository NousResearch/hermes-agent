"""Projected source filters retain current reset and pinned-session semantics."""

import time

import pytest

from hermes_state import SessionDB
from hermes_state_compression import _CHAIN_CAP


@pytest.fixture
def db(tmp_path):
    with SessionDB(db_path=tmp_path / "projection.db") as database:
        yield database


@pytest.mark.parametrize("hops", [101, _CHAIN_CAP + 1], ids=["past-old-bound", "current-bound"])
def test_long_chain_source_filters_match_the_python_walk(db, hops):
    expected_index = min(hops, _CHAIN_CAP)
    parent = None
    for index in range(hops + 1):
        session_id = f"long-{index:04d}"
        source = "webui" if index == expected_index else "telegram"
        db.create_session(session_id, source, parent_session_id=parent)
        if parent is not None:
            db.end_session(parent, "compression")
        parent = session_id

    expected_tip = f"long-{expected_index:04d}"
    assert db.get_compression_tip("long-0000") == expected_tip
    rows = db.list_sessions_rich(source="webui")
    assert [row["id"] for row in rows] == [expected_tip]
    assert rows[0]["source"] == "webui"
    assert db.list_sessions_rich(source="telegram") == []
    assert db.session_count(source="webui", exclude_children=True) == 1
    assert db.session_count(source="telegram", exclude_children=True) == 0
    assert db.session_count_by_source(exclude_children=True) == {"webui": 1}


def test_reset_sibling_cannot_change_projected_source_membership(db):
    db.create_session("root", "telegram")
    db.append_message("root", "user", "original conversation")
    db.end_session("root", "compression")
    db.create_session("tip", "webui", parent_session_id="root")
    db.append_message("tip", "user", "continued conversation")
    db.create_session("reset", "discord", parent_session_id="root", model_config={"_reset_from": "root"})
    db.append_message("reset", "user", "separate reset conversation")
    now = time.time()
    db._conn.execute("UPDATE sessions SET last_activity_at = ? WHERE id = 'reset'", (now + 10,))
    db._conn.commit()

    assert db.get_compression_tip("root") == "tip"
    rows = db.list_sessions_rich(source="webui")
    assert [row["id"] for row in rows] == ["tip"]
    assert db.session_count(source="webui", exclude_children=True) == 1
    assert db.session_count(source="discord", exclude_children=True) == 1
    assert db.session_count_by_source(exclude_children=True) == {"discord": 1, "webui": 1}


@pytest.mark.parametrize("filters", [{"source": "webui"}, {"exclude_sources": ["telegram"]}], ids=["include", "exclude"])
@pytest.mark.parametrize("project", [True, False], ids=["projected", "raw"])
def test_pinned_backfill_obeys_the_same_source_filter_as_the_page(db, filters, project):
    for suffix, root_source, tip_source in (("a", "telegram", "webui"), ("b", "webui", "telegram")):
        root, tip = f"root-{suffix}", f"tip-{suffix}"
        db.create_session(root, root_source)
        db.append_message(root, "user", "original conversation")
        db.end_session(root, "compression")
        db.create_session(tip, tip_source, parent_session_id=root)
        db.append_message(tip, "user", "continued conversation")
        db.set_session_pinned(root, True)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (time.time() - 3600, root))
    db.create_session("foreground", "webui")
    db.append_message("foreground", "user", "newer conversation fills the page")
    db._conn.commit()

    rows = db.list_sessions_rich(limit=1, include_pinned=True, project_compression_tips=project, **filters)
    expected_pin = "tip-a" if project else "root-b"
    assert {row["id"] for row in rows} == {"foreground", expected_pin}
    assert all(row["source"] == "webui" for row in rows)
