"""Projected source filters retain current reset and pinned-session semantics."""

import time

import pytest

from persistence.sqlite.conversation._store.compression import _CHAIN_CAP, get_compression_tip
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    end_session,
    list_sessions_rich,
    session_count,
    session_count_by_source,
    set_session_pinned,
)


@pytest.fixture
def db(backend, scope_a):
    return backend.open(scope_a).holder()


@pytest.mark.parametrize("hops", [101, _CHAIN_CAP + 1], ids=["past-old-bound", "current-bound"])
def test_long_chain_source_filters_match_the_python_walk(db, hops):
    expected_index = min(hops, _CHAIN_CAP)
    parent = None
    for index in range(hops + 1):
        session_id = f"long-{index:04d}"
        source = "webui" if index == expected_index else "telegram"
        create_session(db, session_id, source, parent_session_id=parent)
        if parent is not None:
            end_session(db, parent, "compression")
        parent = session_id

    expected_tip = f"long-{expected_index:04d}"
    assert get_compression_tip(db, "long-0000") == expected_tip
    rows = list_sessions_rich(db, source="webui")
    assert [row["id"] for row in rows] == [expected_tip]
    assert rows[0]["source"] == "webui"
    assert list_sessions_rich(db, source="telegram") == []
    assert session_count(db, source="webui", exclude_children=True) == 1
    assert session_count(db, source="telegram", exclude_children=True) == 0
    assert session_count_by_source(db, exclude_children=True) == {"webui": 1}


def test_reset_sibling_cannot_change_projected_source_membership(db):
    create_session(db, "root", "telegram")
    append_message(db, "root", "user", "original conversation")
    end_session(db, "root", "compression")
    create_session(db, "tip", "webui", parent_session_id="root")
    append_message(db, "tip", "user", "continued conversation")
    create_session(db, "reset", "discord", parent_session_id="root", model_config={"_reset_from": "root"})
    append_message(db, "reset", "user", "separate reset conversation")
    now = time.time()
    db._conn.execute("UPDATE sessions SET last_activity_at = ? WHERE id = 'reset'", (now + 10,))
    db._conn.commit()

    assert get_compression_tip(db, "root") == "tip"
    rows = list_sessions_rich(db, source="webui")
    assert [row["id"] for row in rows] == ["tip"]
    assert session_count(db, source="webui", exclude_children=True) == 1
    assert session_count(db, source="discord", exclude_children=True) == 1
    assert session_count_by_source(db, exclude_children=True) == {"discord": 1, "webui": 1}


@pytest.mark.parametrize("filters", [{"source": "webui"}, {"exclude_sources": ["telegram"]}], ids=["include", "exclude"])
@pytest.mark.parametrize("project", [True, False], ids=["projected", "raw"])
def test_pinned_backfill_obeys_the_same_source_filter_as_the_page(db, filters, project):
    for suffix, root_source, tip_source in (("a", "telegram", "webui"), ("b", "webui", "telegram")):
        root, tip = f"root-{suffix}", f"tip-{suffix}"
        create_session(db, root, root_source)
        append_message(db, root, "user", "original conversation")
        end_session(db, root, "compression")
        create_session(db, tip, tip_source, parent_session_id=root)
        append_message(db, tip, "user", "continued conversation")
        set_session_pinned(db, root, True)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (time.time() - 3600, root))
    create_session(db, "foreground", "webui")
    append_message(db, "foreground", "user", "newer conversation fills the page")
    db._conn.commit()

    rows = list_sessions_rich(db, limit=1, include_pinned=True, project_compression_tips=project, **filters)
    expected_pin = "tip-a" if project else "root-b"
    assert {row["id"] for row in rows} == {"foreground", expected_pin}
    assert all(row["source"] == "webui" for row in rows)


@pytest.mark.parametrize("include_pinned", [False, True], ids=["page", "pinned"])
@pytest.mark.parametrize("sources", [["local"], ["local", "webui"]], ids=["local", "mixed"])
def test_projected_sources_preserve_local_creator_filters_before_pagination(db, sources, include_pinned):
    families = (
        ("cli", "local", "cli", "local"),
        ("desktop", "local", "desktop", "local"),
        ("webui", "telegram", "telegram", "webui"),
        ("switched", "local", "cli", "telegram"),
    )
    for index, (suffix, root_source, creator, tip_source) in enumerate(families):
        root, tip = f"root-{suffix}", f"tip-{suffix}"
        create_session(db, root, root_source)
        append_message(db, root, "user", "original conversation")
        end_session(db, root, "compression")
        create_session(db, tip, tip_source, parent_session_id=root)
        append_message(db, tip, "user", "continued conversation")
        db._conn.execute("UPDATE sessions SET created_source = ? WHERE id IN (?, ?)", (creator, root, tip))
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (1700000000 + index, root))
        set_session_pinned(db, root, suffix in {"cli", "desktop", "switched"})
    db._conn.commit()

    rows = list_sessions_rich(
        db, sources=sources, local_created_sources=["cli"], limit=1, include_pinned=include_pinned,
    )
    expected = {"tip-webui"} if "webui" in sources else {"tip-cli"}
    if include_pinned:
        expected.add("tip-cli")
    assert {row["id"] for row in rows} == expected
