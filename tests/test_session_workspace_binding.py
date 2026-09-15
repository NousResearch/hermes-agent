"""Session <-> workspace grouping key (hermes_state.workspace_key).

The key is what `hermes sessions list --workspace` groups/filters on. It is a
coarse workspace identity derived from fields already recorded on sessions
(git_repo_root, cwd) — no git shelling, no new columns. Branch is deliberately
NOT part of the key.
"""

from hermes_state import workspace_key


def test_repo_root_is_the_key_when_known():
    row = {"git_repo_root": "/www/app", "cwd": "/www/app/src", "git_branch": "feat"}
    assert workspace_key(row) == "/www/app"


def test_falls_back_to_cwd_for_non_git_sessions():
    assert workspace_key({"cwd": "/work/notes"}) == "/work/notes"
    assert workspace_key({"git_repo_root": "", "cwd": "/work/notes"}) == "/work/notes"


def test_none_when_unbound():
    assert workspace_key({}) is None
    assert workspace_key({"cwd": "", "git_repo_root": ""}) is None
    assert workspace_key({"cwd": "   "}) is None


def test_branch_does_not_affect_the_key():
    # Two sessions on the same repo, different branches, group together.
    a = {"git_repo_root": "/www/app", "git_branch": "main"}
    b = {"git_repo_root": "/www/app", "git_branch": "feature-x"}
    assert workspace_key(a) == workspace_key(b) == "/www/app"


def test_repo_root_wins_over_a_differing_cwd():
    # A worktree/subdir session still groups under its repo root, not its cwd.
    row = {"git_repo_root": "/www/app", "cwd": "/www/app/.worktrees/x"}
    assert workspace_key(row) == "/www/app"
