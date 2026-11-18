"""The listing reads re-admit delegate runs on request, which ``sessions.show_subagents`` asks for (#97202)."""

import pytest

from persistence.sqlite.conversation._store.sessions import (
    create_session, end_session, list_sessions_rich, session_count,
)


@pytest.fixture
def db(backend, scope_a):
    store = backend.open(scope_a).holder()
    create_session(store, session_id="parent", source="desktop", model="m")
    create_session(store, session_id="sub", source="desktop", model="m", parent_session_id="parent",
                   model_config={"_delegate_from": "parent"})
    # A compression continuation is not a subagent run and must stay hidden either way.
    create_session(store, session_id="parent-cont", source="desktop", model="m", parent_session_id="parent")
    end_session(store, "parent", "compression")
    return store


def _ids(rows):
    return {row["id"] for row in rows}


def test_subagent_runs_stay_hidden_by_default(db):
    assert "sub" not in _ids(list_sessions_rich(db, limit=50, project_compression_tips=False))
    assert session_count(db, exclude_children=True) == 1


def test_include_subagents_lists_delegate_runs_with_a_matching_count(db):
    rows = list_sessions_rich(db, limit=50, include_subagents=True, project_compression_tips=False)
    assert _ids(rows) == {"parent", "sub"}
    assert next(r for r in rows if r["id"] == "sub")["parent_session_id"] == "parent"
    assert session_count(db, exclude_children=True, include_subagents=True) == 2
