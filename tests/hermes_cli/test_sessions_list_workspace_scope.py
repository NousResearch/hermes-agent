"""``hermes sessions list`` scopes to the current workspace; ``--all`` spans the store.

The default is the identity ``-c`` / ``--resume latest`` use (``_resolve_workspace_key``: git
repo root, else cwd), applied in SQL so ``--limit`` counts matches instead of a global page, and
``--workspace <needle>`` keeps its cross-folder search semantics.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def db(tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def _session(db, sid, preview, *, cwd=None, git_repo_root=None, source="cli"):
    db.create_session(sid, source, cwd=cwd, git_repo_root=git_repo_root)
    db.append_message(sid, "user", preview)


def _list(db, capsys, *, scope, all=False, limit=20, workspace=None):
    args = Namespace(limit=limit, source=None, workspace=workspace, all=all)
    sessions_cmd._cmd_list(db, args)
    return capsys.readouterr().out


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    """Two workspaces: the current one (``repo``) and somewhere else, each with its own rows."""
    key = str(tmp_path / "repo")
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: key)
    return key


def test_default_lists_only_this_workspace(db, scoped, capsys):
    _session(db, "here_a", "here a", git_repo_root=scoped)
    _session(db, "here_b", "here b", git_repo_root=scoped)
    _session(db, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(db, capsys, scope=scoped)
    assert "here_a" in out and "here_b" in out
    assert "there" not in out


def test_legacy_row_cwd_under_the_workspace_counts(db, scoped, capsys):
    """Rows predating per-session git metadata are matched by cwd under the repo root."""
    _session(db, "legacy", "legacy row", cwd=f"{scoped}/sub/dir")

    assert "legacy" in _list(db, capsys, scope=scoped)


def test_all_lists_every_workspace(db, scoped, capsys):
    _session(db, "here", "here", git_repo_root=scoped)
    _session(db, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(db, capsys, scope=scoped, all=True)
    assert "here" in out and "there" in out
    assert "use --all for every session" not in out  # nothing is being hidden


def test_scope_is_named_in_the_listing(db, scoped, capsys):
    _session(db, "here", "here", git_repo_root=scoped)

    out = _list(db, capsys, scope=scoped)
    assert "Sessions in repo — use --all for every session" in out


def test_empty_workspace_points_at_all(db, scoped, capsys):
    _session(db, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(db, capsys, scope=scoped)
    assert "No sessions in repo." in out
    assert "--all" in out
    assert "there" not in out


def test_limit_counts_scope_matches_not_a_global_page(db, scoped, capsys):
    """A workspace whose rows are older than a global page must still page inside its own rows."""
    for i in range(3):
        _session(db, f"here_{i}", f"here {i}", git_repo_root=scoped)
    for i in range(10):  # newer, and in another workspace
        _session(db, f"there_{i}", f"there {i}", git_repo_root="/elsewhere/other")

    out = _list(db, capsys, scope=scoped, limit=2)
    rows = [line.split()[-1] for line in out.splitlines() if line.startswith("here ")]
    assert len(rows) == 2  # the probe row is never rendered
    assert "--limit 4" in out
    assert "there_" not in out


def test_workspace_needle_searches_across_workspaces(db, scoped, capsys):
    """An explicit --workspace is a cross-folder search, so the cwd scope must not narrow it."""
    _session(db, "other_proj", "other proj", git_repo_root="/elsewhere/my-project")

    out = _list(db, capsys, scope=scoped, workspace="my-project")
    assert "other_proj" in out
    assert "Sessions in repo" not in out
