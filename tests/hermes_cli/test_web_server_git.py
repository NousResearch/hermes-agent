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
    (root / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    # A tracked modification + a brand-new untracked file (the new-file case the
    # rail/review must surface).
    (root / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (root / "new.py").write_text("print(1)\nprint(2)\n", encoding="utf-8")
    return root










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






def test_worktree_add_initializes_plain_folder(client, tmp_path):
    folder = tmp_path / "plain-project"
    folder.mkdir()
    (folder / "notes.txt").write_text("not committed\n", encoding="utf-8")

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




def test_git_endpoints_require_auth(repo):
    unauth = TestClient(web_server.app)

    assert unauth.get("/api/git/status", params={"path": str(repo)}).status_code == 401
    assert unauth.post("/api/git/review/stage", json={"path": str(repo)}).status_code == 401


# ── remote-gateway worktree parity (#81724) ─────────────────────────────────
# The desktop's Electron git ops learned remote-branch conversion and
# no-upstream-tracking base branching; the backend REST mirror (what a remote
# gateway serves) must behave identically or worktree flows break exactly and
# only on remote connections.


@pytest.fixture
def repo_with_remote(tmp_path):
    """A committed repo with an `origin` remote carrying main + a feature
    branch that has NO local head (the teammate-branch case)."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True, capture_output=True)

    root = tmp_path / "clone"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "a.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    _git(root, "remote", "add", "origin", str(origin))
    _git(root, "push", "-q", "origin", "main")
    _git(root, "branch", "feature")
    _git(root, "push", "-q", "origin", "feature")
    _git(root, "branch", "-D", "feature")
    _git(root, "fetch", "-q", "origin")
    return root


def test_branches_include_remote_tracking_refs(client, repo_with_remote):
    branches = client.get(
        "/api/git/branches", params={"path": str(repo_with_remote)}
    ).json()["branches"]
    by_name = {branch["name"]: branch for branch in branches}

    # A teammate's branch (no local head) is reachable, flagged as remote.
    assert "origin/feature" in by_name
    assert by_name["origin/feature"]["isRemote"] is True
    assert by_name["origin/feature"]["checkedOut"] is False
    assert by_name["origin/feature"]["worktreePath"] is None

    # Locals carry the flag too, and shadowed remotes/HEAD aliases are noise.
    assert by_name["main"]["isRemote"] is False
    assert "origin/main" not in by_name
    assert all(not branch["name"].endswith("/HEAD") for branch in branches)


def test_worktree_add_existing_remote_branch_tracks_not_detaches(client, repo_with_remote):
    added = client.post(
        "/api/git/worktree/add",
        json={"path": str(repo_with_remote), "existingBranch": "origin/feature"},
    ).json()

    # A remote-tracking ref cannot be checked out directly — the mirror must
    # create the local tracking branch, like `git switch feature` would.
    assert added["branch"] == "feature"
    tree = Path(added["path"])
    assert tree.is_dir()

    head = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=tree, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert head == "feature"  # NOT detached

    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "feature@{upstream}"],
        cwd=tree, check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert upstream == "origin/feature"


def test_worktree_add_from_origin_base_does_not_track(client, repo_with_remote):
    added = client.post(
        "/api/git/worktree/add",
        json={"path": str(repo_with_remote), "branch": "fresh", "base": "origin/main"},
    ).json()
    assert added["branch"] == "fresh"

    # Branching off origin/main must yield a standalone local branch, not one
    # silently wired to the remote's upstream (parity with the Electron op).
    probe = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "fresh@{upstream}"],
        cwd=repo_with_remote, capture_output=True, text=True,
    )
    assert probe.returncode != 0
