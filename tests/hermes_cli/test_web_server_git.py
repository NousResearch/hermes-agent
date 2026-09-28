import asyncio
import subprocess
import threading
from pathlib import Path

import pytest

from hermes_cli import web_server
from hermes_cli.web_routers import git as git_router

pytest.importorskip("starlette.testclient")
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_gh_auth_probe_state():
    previous = (git_router._gh_auth_cache, git_router._gh_auth_probe_task, git_router._gh_auth_probe_started)
    git_router._gh_auth_cache, git_router._gh_auth_probe_task, git_router._gh_auth_probe_started = None, None, 0.0
    try:
        yield
    finally:
        git_router._gh_auth_cache, git_router._gh_auth_probe_task, git_router._gh_auth_probe_started = previous


class _GhProbe:
    """Stand-in for ``_probe_gh_auth``: blocks until released, reports how many ran and how many overlapped."""

    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
        self.calls, self.running, self.peak, self.logged_in = 0, 0, 0, False
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.calls += 1
            self.running += 1
            self.peak = max(self.peak, self.running)
        answer = self.logged_in  # read at START, like the real `gh auth status`
        self.started.set()
        assert self.release.wait(timeout=2)
        with self._lock:
            self.running -= 1
        return {"available": True, "authenticated": answer}


def test_gh_auth_concurrent_cache_misses_share_one_probe(monkeypatch):
    """Overlapping requests never start a second `gh` while one is in flight (#111509)."""
    probe = _GhProbe()
    monkeypatch.setattr(git_router, "_probe_gh_auth", probe)

    async def exercise():
        tasks = [asyncio.create_task(git_router.gh_auth_status_route()) for _ in range(5)]
        while not probe.started.is_set():
            await asyncio.sleep(0)
        probe.release.set()
        return await asyncio.gather(*tasks)

    assert asyncio.run(exercise()) == [{"available": True, "authenticated": False}] * 5
    assert (probe.calls, probe.peak) == (1, 1)


def test_gh_auth_refresh_waits_out_a_probe_started_before_it(monkeypatch):
    """``refresh=true`` issued after `gh auth login` must not adopt the answer of a probe that started
    while still logged out — and must not run a second `gh` concurrently to get its own."""
    probe = _GhProbe()
    monkeypatch.setattr(git_router, "_probe_gh_auth", probe)

    async def exercise():
        stale = asyncio.create_task(git_router.gh_auth_status_route())  # cache miss while logged out
        while not probe.started.is_set():
            await asyncio.sleep(0)
        probe.logged_in = True  # `gh auth login` completes
        refreshed = asyncio.create_task(git_router.gh_auth_status_route(refresh=True))
        await asyncio.sleep(0)
        probe.release.set()
        return await stale, await refreshed, await git_router.gh_auth_status_route()

    stale, refreshed, cached = asyncio.run(exercise())
    assert stale == {"available": True, "authenticated": False}
    assert refreshed == {"available": True, "authenticated": True}
    assert cached == {"available": True, "authenticated": True}  # the TTL cache holds the fresh answer
    assert (probe.calls, probe.peak) == (2, 1)


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


def _out(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _seed_origin(tmp_path):
    """A bare origin whose `main` and `feature` sit one commit past tag `v0`."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main", "--bare")
    work = tmp_path / "seed"
    work.mkdir()
    _git(work, "init", "-q", "-b", "main")
    ident = ("-c", "user.email=t@example.com", "-c", "user.name=Test")
    _git(work, *ident, "commit", "-q", "--allow-empty", "-m", "c1")
    _git(work, "tag", "v0")
    _git(work, *ident, "commit", "-q", "--allow-empty", "-m", "c2")
    _git(work, "branch", "feature")
    _git(work, "push", "-q", str(origin), "main", "feature", "v0")
    return origin, _out(work, "rev-parse", "HEAD")


def _narrow_clone(tmp_path, *, seed_tracking_ref=False):
    """A tag-pinned narrow clone (--single-branch --branch <tag>) whose remote.origin.fetch
    maps only the tag, the shape older installers made (#125686)."""
    origin, tip = _seed_origin(tmp_path)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--single-branch", "--branch", "v0", str(origin), str(clone))
    if seed_tracking_ref:
        _git(clone, "fetch", "-q", "origin", "+refs/heads/feature:refs/remotes/origin/feature")
    return clone, tip


def _normal_clone(tmp_path):
    origin, tip = _seed_origin(tmp_path)
    clone = tmp_path / "normal"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    return clone, tip


def _fetch_config(clone):
    return _out(clone, "config", "--get-all", "remote.origin.fetch")


def test_worktree_add_from_origin_base_on_tag_pinned_clone(tmp_path):
    """``git fetch origin main`` on a tag-only refspec writes FETCH_HEAD and leaves
    ``origin/main`` missing, so ``worktree add`` died with invalid reference."""
    clone, tip = _narrow_clone(tmp_path)
    from hermes_cli.web_git import worktree_add

    added = worktree_add(str(clone), {"base": "origin/main", "name": "x"})

    assert _out(added["path"], "rev-parse", "HEAD") == tip
    upstream = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", f"{added['branch']}@{{upstream}}"],
        cwd=clone, capture_output=True, text=True,
    )
    assert upstream.returncode != 0


@pytest.mark.parametrize(
    "seed_tracking_ref", [False, True], ids=["tracking-ref-missing", "tracking-ref-present"],
)
def test_worktree_add_existing_branch_on_tag_pinned_clone_tracks_remote(client, tmp_path, seed_tracking_ref):
    """Missing ref: "origin/feature" used to be read as a local branch name. Present ref:
    `--track` still cannot wire upstream through a tag-only fetch refspec."""
    clone, tip = _narrow_clone(tmp_path, seed_tracking_ref=seed_tracking_ref)

    added = client.post(
        "/api/git/worktree/add", json={"path": str(clone), "existingBranch": "origin/feature"},
    ).json()

    assert added["branch"] == "feature"
    assert _out(added["path"], "rev-parse", "HEAD") == tip  # the remote feature, not tag v0
    assert _out(added["path"], "rev-parse", "--abbrev-ref", "feature@{upstream}") == "origin/feature"


@pytest.mark.parametrize("clone_of", [_narrow_clone, _normal_clone], ids=["narrow", "normal"])
def test_worktree_add_missing_remote_branch_leaves_fetch_config_alone(client, tmp_path, clone_of):
    """A configured refspec whose source is missing makes every later plain `git fetch`
    fail, so a typo'd "origin/<branch>" must never be registered on the remote."""
    clone, _ = clone_of(tmp_path)
    before = _fetch_config(clone)

    resp = client.post(
        "/api/git/worktree/add", json={"path": str(clone), "existingBranch": "origin/typo"},
    )

    assert resp.status_code != 200
    assert _fetch_config(clone) == before
    _git(clone, "fetch", "-q")


def test_worktree_add_existing_branch_on_normal_clone_adds_no_fetch_refspec(client, tmp_path):
    clone, _ = _normal_clone(tmp_path)
    before = _fetch_config(clone)

    added = client.post(
        "/api/git/worktree/add", json={"path": str(clone), "existingBranch": "origin/feature"},
    ).json()

    assert added["branch"] == "feature"
    assert _fetch_config(clone) == before


def test_worktree_add_local_slash_branch_named_like_a_remote_stays_local(client, tmp_path):
    """A local "origin/feature" branch must be checked out as itself, not swapped for a
    new `feature` tracking the remote branch of the same name."""
    clone, tip = _narrow_clone(tmp_path)
    _git(clone, "branch", "origin/feature")
    local = _out(clone, "rev-parse", "origin/feature")
    assert local != tip

    added = client.post(
        "/api/git/worktree/add", json={"path": str(clone), "existingBranch": "origin/feature"},
    ).json()

    assert added["branch"] == "origin/feature"
    assert _out(added["path"], "rev-parse", "HEAD") == local
    assert _out(clone, "for-each-ref", "refs/remotes") == ""  # never went to the network


def test_worktree_add_glob_base_is_not_fetched(tmp_path):
    """`base` comes from the API; inside a refspec "origin/*" would fetch every branch."""
    clone, _ = _narrow_clone(tmp_path)
    from hermes_cli.web_git import worktree_add

    with pytest.raises(RuntimeError):
        worktree_add(str(clone), {"base": "origin/*", "name": "glob"})

    assert _out(clone, "for-each-ref", "refs/remotes") == ""


def test_worktree_add_base_refreshes_valid_branch_names_the_sanitizer_would_rewrite(tmp_path):
    """"fix+1" is a valid branch the sanitizer rewrites; its base must still be fetched."""
    origin, _ = _seed_origin(tmp_path)
    seed = tmp_path / "seed"
    _git(seed, "branch", "fix+1")
    _git(seed, "push", "-q", str(origin), "fix+1")
    clone = tmp_path / "normal"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(seed, "-c", "user.email=t@example.com", "-c", "user.name=Test",
         "commit", "-q", "--allow-empty", "-m", "moved after the clone")
    _git(seed, "push", "-q", str(origin), "HEAD:fix+1")
    from hermes_cli.web_git import worktree_add

    added = worktree_add(str(clone), {"base": "origin/fix+1", "name": "x"})

    assert _out(added["path"], "rev-parse", "HEAD") == _out(seed, "rev-parse", "HEAD")
