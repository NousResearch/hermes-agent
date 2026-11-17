"""GET /api/sessions/search must match session titles, not only ids and FTS5 message content.

Regression for #66242: the search endpoint composed only
``search_sessions_by_id`` (id/lineage LIKE) with ``search_messages`` (FTS5 over
message content), so a term that lived solely in a manually-set
``sessions.title`` returned zero results. The DB layer already had title
matching — ``list_sessions_rich(search_query=...)`` LIKE-matches titles across
the whole compression chain — and the endpoint never called it. The fix
originally backfilled remaining slots through that helper. Regression for
#151685: all title matches outrank content hits while id hits stay first.

These tests drive the real SessionDB through the real route handler (no fake
DB) so the title lane is exercised end to end, including the SQLite
read-only connection the endpoint opens.

Lane precedence (see :func:`hermes_cli.web_routers.sessions.search_sessions`):
id hits, then exact/prefix/substring title hits, then bm25 content hits.
Rows matched only by id via the title helper retain tail precedence.
"""

import asyncio
import threading

import pytest

import hermes_cli.web_server_sessions as _web_server_sessions
import hermes_cli.web_routers.sessions as _rt_sessions
from hermes_state import SessionDB


@pytest.fixture
def real_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    # A session whose message content never contains the search needle —
    # only its (manually set) title does.
    db.create_session(session_id="titled_session", source="cli", model="m")
    db.set_session_title("titled_session", "Building the Modpack Server")
    db.append_message("titled_session", role="user", content="hello world")
    # A content-matching session: the needle appears in message text only.
    db.create_session(session_id="content_session", source="cli", model="m")
    db.append_message("content_session", role="user", content="modpack build notes")
    # An unrelated session that must never surface.
    db.create_session(session_id="other_session", source="cli", model="m")
    db.append_message("other_session", role="user", content="completely unrelated")
    yield db
    try:
        db.close()
    except Exception:
        pass


@pytest.fixture
def db_path_for_profile(real_db, monkeypatch):
    """Point the endpoint's profile DB opener at the real temp state.db."""
    monkeypatch.setattr(
        _web_server_sessions, "_session_db_path_for_profile", lambda profile: real_db.db_path
    )


def test_search_finds_session_by_partial_title(db_path_for_profile):
    """A term that appears ONLY in the title is findable (#66242); the snippet
    falls back to the session preview, and rows carry the same shape as id hits."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Modpack", limit=20))
    ids = [row["session_id"] for row in response["results"]]
    assert "titled_session" in ids
    row = next(r for r in response["results"] if r["session_id"] == "titled_session")
    assert row["snippet"]  # preview or the title-match fallback text
    assert row["title"] == "Building the Modpack Server"


def test_search_partial_title_match_outranks_content_hit(db_path_for_profile):
    """A mid-title match outranks content, with deduplication by lineage root."""
    response = asyncio.run(_rt_sessions.search_sessions(q="modpack", limit=20))
    ids = [row["session_id"] for row in response["results"]]
    assert set(ids) == {"content_session", "titled_session"}
    assert ids[0] == "titled_session"  # title match precedes a message-only hit
    assert "other_session" not in ids


def test_search_title_lane_respects_limit(db_path_for_profile):
    """The merged result stays capped at limit across all lanes."""
    response = asyncio.run(_rt_sessions.search_sessions(q="modpack", limit=1))
    assert len(response["results"]) == 1


def test_search_title_lane_runs_off_event_loop(db_path_for_profile, monkeypatch):
    """The title backfill runs on the endpoint's worker thread, not the loop
    thread (#60747 discipline): the whole _search body already runs in
    asyncio.to_thread, so the real SessionDB calls must never land on the loop."""
    loop_thread = threading.get_ident()
    db_threads: list[int] = []
    real_search_messages = SessionDB.search_messages

    def spy(self, *args, **kwargs):
        db_threads.append(threading.get_ident())
        return real_search_messages(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "search_messages", spy)
    asyncio.run(_rt_sessions.search_sessions(q="Modpack", limit=20))
    assert db_threads
    assert all(tid != loop_thread for tid in db_threads)


@pytest.fixture
def ranked_db(real_db, db_path_for_profile):
    """Title hits that rank exact / prefix / substring, each with a *different*
    session holding a real bm25 message-content hit for the same needle.

    The two lanes therefore both fire for every needle, so the assertion is
    about lane ORDER, not about which lane found something. Title-only
    sessions get content free of the needle (and ids free of it too, so the
    id lane cannot promote them first).
    """
    db = real_db
    # --- exact title hit: normalized title == needle ---------------
    db.create_session(session_id="doc_a", source="cli", model="m")
    db.set_session_title("doc_a", "Quetzalcoatl")
    db.append_message("doc_a", role="user", content="sidebar only")
    db.create_session(session_id="doc_b", source="cli", model="m")
    db.append_message("doc_b", role="user", content="the Quetzalcoatl expedition log")
    # --- prefix title hit: title startswith needle ------------------
    db.create_session(session_id="doc_c", source="cli", model="m")
    db.set_session_title("doc_c", "Cormorant Migration Notes")
    db.append_message("doc_c", role="user", content="sidebar only")
    db.create_session(session_id="doc_d", source="cli", model="m")
    db.append_message("doc_d", role="user", content="cormorant sightings at the pier")
    # --- substring-only title hit: needle sits mid-title ------------
    db.create_session(session_id="doc_e", source="cli", model="m")
    db.set_session_title("doc_e", "Notes on the Wandering Albatross")
    db.append_message("doc_e", role="user", content="sidebar only")
    db.create_session(session_id="doc_f", source="cli", model="m")
    db.append_message("doc_f", role="user", content="wandering route over the ridge")
    return db


def test_search_exact_title_outranks_content_hit(ranked_db):
    """An exact session-title match is rank 0: the user just read that title in
    the sidebar, so it must sit ahead of the bm25 content hit for the same word."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Quetzalcoatl", limit=20))
    ids = [row["session_id"] for row in response["results"]]
    assert ids[0] == "doc_a", ids
    # Both lanes really fired: the content-only hit is still in the results.
    assert "doc_b" in ids, ids
    assert ids.index("doc_a") < ids.index("doc_b")


def test_search_prefix_title_outranks_content_hit(ranked_db):
    """A prefix title match outranks a content hit too (rank 1), ahead of bm25."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Cormorant", limit=20))
    ids = [row["session_id"] for row in response["results"]]
    assert ids[0] == "doc_c", ids
    assert "doc_d" in ids, ids
    assert ids.index("doc_c") < ids.index("doc_d")


def test_search_mid_title_match_outranks_content_hit(ranked_db):
    """A word in the middle of the title must outrank a message-content hit."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Wandering", limit=20))
    ids = [row["session_id"] for row in response["results"]]
    assert "doc_f" in ids and "doc_e" in ids, ids
    assert ids.index("doc_e") < ids.index("doc_f"), ids


def test_search_exact_title_returned_at_limit_one(ranked_db):
    """With one slot the exact title wins it outright; it used to be starved by
    the content lane filling `limit` first."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Quetzalcoatl", limit=1))
    assert [row["session_id"] for row in response["results"]] == ["doc_a"]


def test_search_mid_title_returned_at_limit_one(ranked_db):
    """A content hit must not starve a mid-title match at the limit."""
    response = asyncio.run(_rt_sessions.search_sessions(q="Wandering", limit=1))
    assert [row["session_id"] for row in response["results"]] == ["doc_e"]
