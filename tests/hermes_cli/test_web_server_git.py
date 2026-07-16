import subprocess
from pathlib import Path

import pytest

from hermes_cli import web_server

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture
def client():
    previous = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    test_client = TestClient(web_server.app)
    test_client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                delattr(web_server.app.state, "auth_required")
            except AttributeError:
                pass
        else:
            web_server.app.state.auth_required = previous


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("one\ntwo\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    # A tracked modification + a brand-new untracked file (the new-file case the
    # rail/review must surface).
    (root / "a.txt").write_text("one\ntwo\nthree\n")
    (root / "new.py").write_text("print(1)\nprint(2)\n")
    return root


def test_status_reports_branch_and_change_counts(client, repo):
    body = client.get("/api/git/status", params={"path": str(repo)}).json()

    assert body["branch"] == body["defaultBranch"]
    assert body["branch"]
    assert body["detached"] is False
    # 1 tracked-modified + 1 untracked = 2 changed paths.
    assert body["changed"] == 2
    assert body["untracked"] == 1
    # +1 (a.txt) folded with +2 (untracked new.py) since `git diff HEAD` skips untracked.
    assert body["added"] == 3
    assert {f["path"] for f in body["files"]} == {"a.txt", "new.py"}


def test_status_returns_null_outside_repo(client, tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()

    assert client.get("/api/git/status", params={"path": str(plain)}).json() is None


def test_review_list_classifies_modified_and_untracked(client, repo):
    body = client.get("/api/git/review/list", params={"path": str(repo)}).json()

    files = {f["path"]: f for f in body["files"]}
    assert files["a.txt"]["status"] == "M"
    assert files["a.txt"]["added"] == 1
    assert files["new.py"]["status"] == "?"
    assert files["new.py"]["added"] == 2  # untracked insertions counted from disk


def test_review_diff_shows_change_and_synthesizes_untracked(client, repo):
    tracked = client.get(
        "/api/git/review/diff", params={"path": str(repo), "file": "a.txt"}
    ).json()["diff"]
    assert "+three" in tracked

    untracked = client.get(
        "/api/git/review/diff", params={"path": str(repo), "file": "new.py"}
    ).json()["diff"]
    assert "print(1)" in untracked  # all-add diff for a file git doesn't track yet


def test_stage_commit_roundtrip_clears_changes(client, repo):
    assert client.post("/api/git/review/stage", json={"path": str(repo), "file": "a.txt"}).json() == {"ok": True}
    staged = client.get("/api/git/status", params={"path": str(repo)}).json()
    assert staged["staged"] >= 1

    assert client.post(
        "/api/git/review/commit", json={"path": str(repo), "message": "tracked change", "push": False}
    ).json() == {"ok": True}

    after = client.get("/api/git/status", params={"path": str(repo)}).json()
    # The tracked change is committed; only the untracked file remains.
    assert after["changed"] == 1
    assert after["untracked"] == 1


def test_commit_with_nothing_staged_commits_all_changes(client, repo):
    assert client.post(
        "/api/git/review/commit", json={"path": str(repo), "message": "commit all", "push": False}
    ).json() == {"ok": True}

    assert client.get("/api/git/status", params={"path": str(repo)}).json()["changed"] == 0


def test_worktrees_and_branch_lifecycle(client, repo):
    worktrees = client.get("/api/git/worktrees", params={"path": str(repo)}).json()["worktrees"]
    assert any(tree["isMain"] and tree["path"] == str(repo) for tree in worktrees)

    added = client.post(
        "/api/git/worktree/add", json={"path": str(repo), "branch": "feature/x"}
    ).json()
    assert added["branch"] == "feature/x"
    assert Path(added["path"]).is_dir()

    branches = client.get("/api/git/branches", params={"path": str(repo)}).json()["branches"]
    assert any(b["name"] == "feature/x" and b["checkedOut"] for b in branches)

    removed = client.post(
        "/api/git/worktree/remove", json={"path": str(repo), "worktreePath": added["path"], "force": True}
    ).json()
    assert removed["removed"]


def test_worktree_add_initializes_plain_folder(client, tmp_path):
    folder = tmp_path / "plain-project"
    folder.mkdir()
    (folder / "notes.txt").write_text("not committed\n")

    added = client.post(
        "/api/git/worktree/add", json={"path": str(folder), "branch": "feature/plain"}
    ).json()

    assert added["branch"] == "feature/plain"
    assert Path(added["path"]).is_dir()
    assert (folder / ".git").exists()
    _git(folder, "rev-parse", "--verify", "HEAD")

    status = client.get("/api/git/status", params={"path": str(folder)}).json()
    assert status["branch"] == status["defaultBranch"]
    assert status["branch"]
    # Existing files are not silently committed by repo initialization.
    assert any(file["path"] == "notes.txt" and file["untracked"] for file in status["files"])


def test_commit_context_includes_diff_and_untracked(client, repo):
    body = client.get("/api/git/review/commit-context", params={"path": str(repo)}).json()

    assert "+three" in body["diff"]
    assert "new.py" in body["diff"]  # untracked files listed since they carry no diff


def test_ship_info_degrades_without_gh(client, repo, monkeypatch):
    monkeypatch.setattr(web_server._web_git.shutil, "which", lambda _name: None)

    assert client.get("/api/git/review/ship-info", params={"path": str(repo)}).json() == {
        "ghReady": False,
        "pr": None,
    }


def test_git_endpoints_require_auth(repo):
    unauth = TestClient(web_server.app)

    assert unauth.get("/api/git/status", params={"path": str(repo)}).status_code == 401
    assert unauth.post("/api/git/review/stage", json={"path": str(repo)}).status_code == 401


# --- locked-deployment read confinement for the git diff `file` param --------
#
# review_diff / file_diff_vs_head shell out to `git diff --no-index -- /dev/null
# <file>` for untracked files, which reads ANY path the caller names. On a
# locked dashboard that let a confined tenant read credential files or escape
# the managed root through the review panel, bypassing the #57505 control.
# The guard confines the resolved file to the locked root and blocks credential
# files. Local mode is unaffected (see test_review_diff_* above).


@pytest.fixture
def locked_repo(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(root))
    repo_dir = root / "repo"
    repo_dir.mkdir()
    _git(repo_dir, "init", "-q")
    _git(repo_dir, "config", "user.email", "t@example.com")
    _git(repo_dir, "config", "user.name", "Test")
    (repo_dir / "tracked.txt").write_text("one\n")
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-qm", "init")
    (repo_dir / "new.py").write_text("print(1)\n")   # benign untracked
    (repo_dir / "auth.json").write_text('{"token": "s3cr3t"}')  # credential in-repo
    return root, repo_dir


def test_review_diff_blocks_absolute_path_escape_in_locked_mode(client, locked_repo, tmp_path):
    root, repo_dir = locked_repo
    secret = tmp_path / "host-secret.env"     # outside the locked root
    secret.write_text("HOST_SECRET=leaked")

    response = client.get(
        "/api/git/review/diff", params={"path": str(repo_dir), "file": str(secret)}
    )

    assert response.status_code == 403


def test_review_diff_blocks_credential_file_in_locked_mode(client, locked_repo):
    _root, repo_dir = locked_repo

    response = client.get(
        "/api/git/review/diff", params={"path": str(repo_dir), "file": "auth.json"}
    )

    assert response.status_code == 403


def test_review_diff_allows_benign_untracked_file_in_locked_mode(client, locked_repo):
    _root, repo_dir = locked_repo

    response = client.get(
        "/api/git/review/diff", params={"path": str(repo_dir), "file": "new.py"}
    )

    assert response.status_code == 200
    assert "print(1)" in response.json()["diff"]


def test_file_diff_blocks_credential_and_escape_in_locked_mode(client, locked_repo, tmp_path):
    _root, repo_dir = locked_repo
    secret = tmp_path / "outside.env"
    secret.write_text("X=1")

    assert client.get(
        "/api/git/file-diff", params={"path": str(repo_dir), "file": "auth.json"}
    ).status_code == 403
    assert client.get(
        "/api/git/file-diff", params={"path": str(repo_dir), "file": str(secret)}
    ).status_code == 403


# commit-context has no `file` arg: it returns the whole-repo staged/HEAD
# diff for the caller-supplied cwd. Guard that cwd the same way so a locked
# deployment cannot point path at an outside-root repository and read its
# diff contents.


def test_commit_context_blocks_repo_outside_locked_root(client, locked_repo, tmp_path):
    _root, _repo_dir = locked_repo
    outside = tmp_path / "outside-repo"
    outside.mkdir()
    _git(outside, "init", "-q")
    _git(outside, "config", "user.email", "t@example.com")
    _git(outside, "config", "user.name", "Test")
    (outside / "secret.txt").write_text("HOST_SECRET=leaked\n")
    _git(outside, "add", "-A")
    _git(outside, "commit", "-qm", "init")
    (outside / "secret.txt").write_text("HOST_SECRET=changed\n")

    response = client.get(
        "/api/git/review/commit-context", params={"path": str(outside)}
    )

    assert response.status_code == 403
    body = response.json()
    # Must not leak the outside-repo diff (detail-only error, no "diff" payload).
    assert "diff" not in body or "HOST_SECRET" not in str(body)


def test_commit_context_allows_repo_under_locked_root(client, locked_repo):
    _root, repo_dir = locked_repo

    response = client.get(
        "/api/git/review/commit-context", params={"path": str(repo_dir)}
    )

    assert response.status_code == 200
    assert "new.py" in response.json()["diff"]
