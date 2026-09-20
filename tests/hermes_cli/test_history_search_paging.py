"""History search must retrieve titles and page conversations, not message hits."""
import asyncio

import pytest
from fastapi import HTTPException
from hermes_state import SessionDB
from hermes_cli.web_routers import sessions


@pytest.fixture
def db(tmp_path, monkeypatch):
    store = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(sessions, "_open_session_db_for_profile", lambda *a, **kw: store)
    monkeypatch.setattr(store, "close", lambda: None)
    yield store
    store._conn.close()


def seed(db, sid, title, content="ordinary text", source="cli"):
    db.create_session(session_id=sid, source=source, model="test")
    db.set_session_title(sid, title)
    db.append_message(session_id=sid, role="user", content=content)


def search(**kw):
    return asyncio.run(sessions.search_sessions(**kw))


def test_titles_rank_before_content_and_pages_are_distinct(db):
    for i in range(25):
        seed(db, f"s{i:02}", f"launch notes {i}", "ordinary text")
    seed(db, "exact", "launch")
    seed(db, "content", "Other", "launch content")
    for _ in range(120):
        db.append_message(session_id="content", role="user", content="launch repeated")
    first = search(q="launch", limit=20)
    assert first["results"][0]["session_id"] == "exact"
    assert first["has_more"] is True
    second = search(q="launch", limit=20, offset=first["next_offset"])
    ids = [r["session_id"] for p in (first, second) for r in p["results"]]
    assert len(ids) == len(set(ids)) == 27
    assert ids[-1] == "content"
    assert second["has_more"] is False
    assert second["next_offset"] is None
    # Many messages in one conversation must not mask later distinct matches.
    for i in range(25):
        seed(db, f"msg{i}", f"Other {i}", "launch content")
    pages, offset = [], 0
    while True:
        page = search(q="launch", limit=7, offset=offset)
        pages.extend(r["session_id"] for r in page["results"])
        if not page["has_more"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert len(pages) == len(set(pages)) == 52


def test_title_literal_filters_blank_and_offset(db):
    seed(db, "literal", "100%_done")
    seed(db, "near", "100xAdone")
    seed(db, "cron", "100%_done cron", source="cron")
    db.set_session_archived("literal", True)
    page = search(q="100%_done", exclude_sources="cron")
    assert [r["session_id"] for r in page["results"]] == ["literal"]
    assert page["results"][0]["archived"] is True
    assert search(q="  ")["results"] == []
    with pytest.raises(HTTPException) as error:
        search(q="launch", offset=-1)
    assert error.value.status_code == 422


def test_distinct_lineage_roots_resolving_to_one_tip_surface_once(db, monkeypatch):
    # Isolate the resolver collision observed on legacy stores: distinct roots
    # can resolve to one surfaced id. The endpoint must emit that id only once.
    seed(db, "root_a", "Root A", "pengu audit text")
    seed(db, "root_b", "Root B", "pengu audit text again")
    seed(db, "tip", "Live continuation", "pengu live text")
    original = db.get_compression_tip
    monkeypatch.setattr(
        db, "get_compression_tip",
        lambda root: "tip" if root in {"root_a", "root_b"} else original(root),
    )
    page = search(q="pengu", limit=20)
    assert [row["session_id"] for row in page["results"]] == ["tip"]
