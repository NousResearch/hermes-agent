"""Path-scoped session operations must not cross case-distinct workspaces."""

import pytest

from hermes_constants import get_hermes_home
from hermes_state import SessionDB


@pytest.fixture
def db():
    with SessionDB(get_hermes_home() / "state.db") as store:
        yield store


@pytest.mark.parametrize("prefix, separator", [
    ("/work/Agent_100%", "/"),
    (r"C:\work\Agent_100%", "\\"),
])
def test_directory_filters_keep_listing_archive_and_prune_in_one_workspace(db, prefix, separator):
    paths = {
        "root": prefix,
        "child": prefix + separator + "src",
        "case_sibling": prefix.replace("Agent", "agent") + separator + "src",
        "prefix_sibling": prefix + "-backup" + separator + "src",
        "wildcard_sibling": prefix.replace("_", "X").replace("%", "Y") + separator + "src",
    }
    for sid, cwd in paths.items():
        db.create_session(sid, source="cli", cwd=cwd)
        db.end_session(sid, "done")
    expected = {"root", "child"}

    for selected_prefix in (prefix, prefix + separator):
        rows = db.list_sessions_rich(cwd_prefix=selected_prefix)
        assert {row["id"] for row in rows} == expected
        assert db.session_count(cwd_prefix=selected_prefix) == len(rows)
        assert {row["id"] for row in db.search_sessions(workspace_key=selected_prefix)} == expected
        candidates = db.list_prune_candidates(cwd_prefix=selected_prefix)
        assert {row["id"] for row in candidates} == expected
        assert db.count_prune_matches(cwd_prefix=selected_prefix) == len(candidates)

    assert db.archive_sessions(cwd_prefix=prefix) == len(expected)
    assert {row["id"] for row in db.list_sessions_rich(archived_only=True)} == expected
    assert db.prune_sessions(older_than_days=None, cwd_prefix=prefix) == len(expected)
    assert {row["id"] for row in db.list_sessions_rich()} == set(paths) - expected


@pytest.mark.parametrize("prefix, separator", [
    ("/boards/Agent_100%/workspaces", "/"),
    (r"C:\boards\Agent_100%\workspaces", "\\"),
])
def test_worker_retagging_preserves_other_workspaces(db, prefix, separator):
    paths = {
        "root": prefix,
        "worker": prefix + separator + "task",
        "case_sibling": prefix.replace("Agent", "agent") + separator + "task",
        "prefix_sibling": prefix + "-backup" + separator + "task",
        "wildcard_sibling": prefix.replace("_", "X").replace("%", "Y") + separator + "task",
    }
    for sid, cwd in paths.items():
        db.create_session(sid, source="cli", cwd=cwd)
    expected = {"root", "worker"}

    assert db.retag_kanban_worker_sessions(prefix) == len(expected)
    assert {row["id"] for row in db.list_sessions_rich(source="kanban")} == expected
    assert {row["id"] for row in db.list_sessions_rich(source="cli")} == set(paths) - expected
    assert db.retag_kanban_worker_sessions(prefix) == 0
