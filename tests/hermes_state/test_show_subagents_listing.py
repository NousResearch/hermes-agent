"""``sessions.show_subagents`` re-admits delegate runs to human-facing session lists (#97202)."""

import pytest

from hermes_cli.session_listing import subagent_listing_scope
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(session_id="parent", source="desktop", model="m")
    store.create_session(session_id="sub", source="desktop", model="m", parent_session_id="parent",
                         model_config={"_delegate_from": "parent"})
    # A compression continuation is not a subagent run and must stay hidden either way.
    store.create_session(session_id="parent-cont", source="desktop", model="m", parent_session_id="parent")
    store.end_session("parent", "compression")
    yield store
    store.close()


def _ids(rows):
    return {row["id"] for row in rows}


def test_subagent_runs_stay_hidden_by_default(db):
    assert "sub" not in _ids(db.list_sessions_rich(limit=50, project_compression_tips=False))
    assert db.session_count(exclude_children=True) == 1


def test_include_subagents_lists_delegate_runs_with_a_matching_count(db):
    rows = db.list_sessions_rich(limit=50, include_subagents=True, project_compression_tips=False)
    assert _ids(rows) == {"parent", "sub"}
    assert next(r for r in rows if r["id"] == "sub")["parent_session_id"] == "parent"
    assert db.session_count(exclude_children=True, include_subagents=True) == 2


def _home(tmp_path, show):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(f"sessions:\n  show_subagents: {str(show).lower()}\n")
    return home


def test_listing_scope_follows_the_store_home_config(tmp_path):
    assert subagent_listing_scope(_home(tmp_path, False)) == (False, None)
    assert subagent_listing_scope(_home(tmp_path, True)) == (True, None)


def test_listing_scope_drops_the_subagent_exclusion_from_the_recents_shape(tmp_path):
    home = _home(tmp_path, True)
    assert subagent_listing_scope(home, exclude_sources=["cron", "subagent"]) == (True, ["cron"])
    assert subagent_listing_scope(home, exclude_sources=["subagent"]) == (True, None)


def test_listing_scope_leaves_source_scoped_and_messaging_slices_alone(tmp_path):
    home = _home(tmp_path, True)
    assert subagent_listing_scope(home, source="cron") == (False, None)
    assert subagent_listing_scope(home, sources=["telegram"]) == (False, None)
    assert subagent_listing_scope(home, exclude_sources=["cron", "cli", "desktop"]) == (False, ["cron", "cli", "desktop"])
