"""``hermes sessions list`` scopes to the current workspace; ``--all`` spans the store.

The default is the identity ``-c`` / ``--resume latest`` use (``_resolve_workspace_key``: git
repo root, else cwd), applied in SQL so ``--limit`` counts matches instead of a global page, and
``--workspace <needle>`` keeps its cross-folder search semantics.
"""

from argparse import Namespace

import pytest

from hermes_cli import sessions_cmd


@pytest.fixture
def store(tmp_path):
    from persistence_assembly import open_persistence, profile_scope

    with open_persistence(profile_scope(home=tmp_path)) as store:
        yield store


def _session(store, sid, preview, *, cwd=None, git_repo_root=None, source="cli"):
    from tests.fixtures.storage_seed import seed_messages, seed_session

    seed_session(store, sid, source, cwd=cwd, git_repo_root=git_repo_root)
    seed_messages(store, sid, {"role": "user", "content": preview})


def _list(store, capsys, *, all=False, limit=20, workspace=None):
    args = Namespace(limit=limit, source=None, workspace=workspace, all=all)
    sessions_cmd._cmd_list(args, persistence=store)
    return capsys.readouterr().out


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    """The current workspace (``repo``); rows elsewhere belong to another folder."""
    key = str(tmp_path / "repo")
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: key)
    return key


def test_default_lists_only_this_workspace(store, scoped, capsys):
    _session(store, "here_a", "here a", git_repo_root=scoped)
    _session(store, "here_b", "here b", git_repo_root=scoped)
    _session(store, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(store, capsys)
    assert "here_a" in out and "here_b" in out
    assert "there" not in out


def test_legacy_row_cwd_under_the_workspace_counts(store, scoped, capsys):
    """Rows predating per-session git metadata are matched by cwd under the repo root."""
    _session(store, "legacy", "legacy row", cwd=f"{scoped}/sub/dir")

    assert "legacy" in _list(store, capsys)


def test_all_lists_every_workspace(store, scoped, capsys):
    _session(store, "here", "here", git_repo_root=scoped)
    _session(store, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(store, capsys, all=True)
    assert "here" in out and "there" in out
    assert "use --all for every session" not in out  # nothing is being hidden


def test_scope_is_named_in_the_listing(store, scoped, capsys):
    _session(store, "here", "here", git_repo_root=scoped)

    out = _list(store, capsys)
    assert "Sessions in repo — use --all for every session" in out


def test_empty_workspace_points_at_all(store, scoped, capsys):
    _session(store, "there", "there", git_repo_root="/elsewhere/other")

    out = _list(store, capsys)
    assert "No sessions in repo." in out
    assert "--all" in out
    assert "there" not in out


def test_limit_counts_scope_matches_not_a_global_page(store, scoped, capsys):
    """A workspace whose rows are older than a global page must still page inside its own rows."""
    for i in range(3):
        _session(store, f"here_{i}", f"here {i}", git_repo_root=scoped)
    for i in range(10):  # newer, and in another workspace
        _session(store, f"there_{i}", f"there {i}", git_repo_root="/elsewhere/other")

    out = _list(store, capsys, limit=2)
    rows = [line.split()[-1] for line in out.splitlines() if line.startswith("here ")]
    assert len(rows) == 2  # the probe row is never rendered
    assert "--limit 4" in out
    assert "there_" not in out


def test_workspace_needle_searches_across_workspaces(store, scoped, capsys):
    """An explicit --workspace is a cross-folder search, so the cwd scope must not narrow it."""
    _session(store, "other_proj", "other proj", git_repo_root="/elsewhere/my-project")

    out = _list(store, capsys, workspace="my-project")
    assert "other_proj" in out
    assert "Sessions in repo" not in out
