"""Release-channel checks and updates against actual repositories and HTTP feeds."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
from threading import Thread
from types import SimpleNamespace
import subprocess
import urllib.request
from urllib.parse import urlsplit

import pytest

from hermes_cli import main, update_cmd
from hermes_cli.source_releases import resolve_source_release
from hermes_cli.update_channel import set_install_channel


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture(params=["utf-8", "utf-8-sig"])
def releases(tmp_path, monkeypatch, request):
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-b", "main")
    git(origin, "config", "user.name", "Release Fixture")
    git(origin, "config", "user.email", "fixture@example.invalid")
    git(origin, "config", "commit.gpgsign", "false")
    git(origin, "config", "tag.gpgsign", "false")
    commits = []
    for label in ("old", "stable", "canary", "unpublished"):
        (origin / "content.txt").write_text(label, encoding="utf-8")
        git(origin, "add", ".")
        git(origin, "commit", "-m", label)
        commits.append(git(origin, "rev-parse", "HEAD"))
    tags = {"stable": "v1.2.3", "canary": "v1.2.3+canary.20260911T125822Z"}
    git(origin, "tag", "-a", tags["stable"], commits[1], "-m", "stable")
    git(origin, "tag", "-a", tags["canary"], commits[2], "-m", "canary")
    git(origin, "tag", "v99.0.0", commits[3])
    git(origin, "tag", "v99.0.1+canary.20260912T125822Z", commits[3])
    checkout = tmp_path / "checkout"
    git(tmp_path, "clone", str(origin), str(checkout))
    git(checkout, "config", "user.name", "Release Fixture")
    git(checkout, "config", "user.email", "fixture@example.invalid")
    git(checkout, "config", "commit.gpgsign", "false")
    git(checkout, "checkout", "--detach", commits[0])
    monkeypatch.setattr(main, "PROJECT_ROOT", checkout)
    # The shared test interpreter belongs to a different real install. These
    # fixtures own the throwaway checkout; never relaunch its command in the
    # interpreter's owning install (which would receive pytest's arguments).
    from hermes_cli import update_owning_install
    monkeypatch.setattr(update_owning_install, "owning_install_root", lambda _root: None)
    monkeypatch.setenv("HERMES_INSTALL_ROOT", str(checkout))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    responses = {}
    requests = []
    for channel, tag in tags.items():
        responses[f"/releases/{channel}/index.html"] = (
            f'<meta name="hermes-build" content="{tag}">'
        )
        responses[f"/repos/NousResearch/hermes-agent/releases/tags/{tag}"] = {
            "tag_name": tag, "draft": False, "prerelease": channel == "canary",
        }
        responses[f"/repos/NousResearch/hermes-agent/commits/{tag}"] = {
            "sha": commits[1 if channel == "stable" else 2],
        }
    responses["/releases/stable/release-candidates.json"] = {
        "tag": tags["stable"], "commit": commits[1],
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            data = responses.get(self.path)
            self.send_response(404 if data is None else 200)
            self.end_headers()
            if data is not None:
                self.wfile.write((data if isinstance(data, str) else json.dumps(data)).encode(request.param))

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    open_url = urllib.request.urlopen

    def local_urlopen(request, *args, **kwargs):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        parsed = urlsplit(url)
        return open_url(f"http://127.0.0.1:{server.server_port}{parsed.path}"
                        + (f"?{parsed.query}" if parsed.query else ""), *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", local_urlopen)
    # Legacy pointer tests below retain their HTTP/tag boundary. New CLI callers
    # consume the protocol reader, whose complete schema is tested independently.
    from hermes_cli import source_releases
    def resolve_channel(name, repository):
        record = {"name": name, "repository": repository, "state": "active",
                  "policy": "source-branch" if name == "main" else "preview"}
        manifest = None
        if name == "main":
            record["delivery"] = {"kind": "source-branch", "branch": "main"}
        else:
            manifest = {"request": {"commit": commits[1 if name == "stable" else 2],
                "sourceVersion": tags[name].removeprefix("v"), "buildId": "legacy-fixture"}}
        return SimpleNamespace(requested=record, terminal=record, manifest=manifest)
    monkeypatch.setattr(source_releases, "_resolve_channel", resolve_channel)
    yield SimpleNamespace(root=checkout, origin=origin, commits=commits,
                          tags=tags, responses=responses, requests=requests)
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.mark.parametrize("git_cmd", [["git"], None], ids=["git", "no-git"])
def test_stable_resolution_uses_promoted_pointer_not_highest_tag(releases, git_cmd):
    assert resolve_source_release("stable", git_cmd, releases.root) == (
        releases.tags["stable"], releases.commits[1],
    )
    assert releases.requests


def test_truncated_source_release_response_is_unavailable_on_public_path(monkeypatch):
    from hermes_cli import source_releases

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            raise http.client.IncompleteRead(b'{"tag":', 12)

    def urlopen(request, timeout=30):
        assert request.full_url.endswith("/releases/stable/release-candidates.json")
        assert timeout == 30
        return Response()

    monkeypatch.setattr(source_releases.urllib.request, "urlopen", urlopen)
    assert resolve_source_release("stable") == (None, None)


def test_retirement_no_git_production_checkout_refuses_unverifiable_stamp(tmp_path):
    """F1 regression: pyproject.toml is not installed-source ordering evidence.

    Production checkouts carry the inert 0.0.0 placeholder (and it is writable
    in any tree), so the version comparison the previous repair used proved
    nothing and the strict ZIP path failed open — a retirement target at
    sourceVersion 1.2.1 admitted an install stamped with an unrelated (newer)
    commit. The install stamp is the only no-Git authority: anything but the
    pinned target commit refuses the strict apply with the explicit-
    destination remedy; the passive check stays permissive.
    """
    from hermes_cli.source_releases import _retirement_commit_proof

    # The production manifest, unmodified.
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "0.0.0"\n', encoding="utf-8")
    request = {"commit": "a" * 40, "sourceVersion": "1.2.1", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    # An install stamped with a different commit cannot show it predates the
    # qualified build: the strict apply refuses instead of authorizing the
    # rollback, whatever pyproject.toml says.
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is False

    # A checkout with no stamp at all is equally unverifiable.
    (tmp_path / "install-stamp.json").unlink()
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is False

    # An exact target stamp is positive proof on both paths.
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "a" * 40}), encoding="utf-8")
    assert _retirement_commit_proof(request, terminal, None, tmp_path, True) is True


def test_retirement_no_git_without_stamp_stays_permissive_passively(tmp_path):
    """No stamp, no Git: the passive checkers keep main's permissive answer
    (reported as ancestryUnverified) instead of stranding the install; only
    the strict apply path refuses unverified ordering."""
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.0.0", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is False
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)


def test_retirement_downgrade_refuses_when_ancestry_is_unavailable(monkeypatch, tmp_path):
    from hermes_cli.source_releases import _retirement_commit_proof

    class Result:
        returncode = 0
        stdout = "d" * 40 + "\n"

    def run(argv, **_kwargs):
        assert argv[1:3] == ["rev-list", "--ancestry-path"]
        return Result()

    monkeypatch.setattr("subprocess.run", run)
    # A non-shallow checkout (rev-parse reports false) with the target present
    # and a descendant HEAD: the downgrade refusal is decisive without a fetch.
    monkeypatch.setattr(
        "subprocess.run",
        lambda argv, **_kw: Result() if argv[1:3] == ["rev-list", "--ancestry-path"]
        else subprocess.CompletedProcess(args=argv, returncode=0, stdout="false\n"))
    request = {"commit": "a" * 40, "sourceVersion": "1.0.0", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    with pytest.raises(ValueError, match="downgrade"):
        _retirement_commit_proof(request, terminal, ["git"], tmp_path, True)


def test_retirement_rejects_divergent_history(tmp_path):
    from hermes_cli.source_releases import _retirement_commit_proof

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Release Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "content.txt").write_text("base", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    (root / "content.txt").write_text("destination", encoding="utf-8")
    git(root, "commit", "-am", "destination")
    destination = git(root, "rev-parse", "HEAD")
    base = git(root, "rev-parse", "HEAD~1")
    git(root, "checkout", "-b", "divergent", base)
    (root / "content.txt").write_text("installed", encoding="utf-8")
    git(root, "commit", "-am", "installed")

    request = {"commit": destination, "sourceVersion": "1.0.0", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 2}}
    with pytest.raises(ValueError, match="equal to or an ancestor"):
        _retirement_commit_proof(request, terminal, ["git"], root, True)


def test_retirement_rejects_divergent_history_after_target_fetch(tmp_path, monkeypatch):
    """The C/D equal-sequence shallow hole: the apply fetch must not admit divergence."""
    from hermes_cli import source_releases

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Release Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "content.txt").write_text("base", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")
    base = git(root, "rev-parse", "HEAD")
    (root / "content.txt").write_text("target", encoding="utf-8")
    git(root, "commit", "-am", "target")
    target = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-b", "side", base)
    (root / "content.txt").write_text("installed", encoding="utf-8")
    git(root, "commit", "-am", "installed")
    installed = git(root, "rev-parse", "HEAD")
    origin = tmp_path / "origin.git"
    git(root, "remote", "add", "origin", str(origin))
    git(tmp_path, "clone", "--bare", str(root), str(origin))
    # Depth-1 style install at the sibling: the target is invisible locally.
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach", installed)
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0
    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    # resolve_source_target patches _resolve_channel per test; guard reads the
    # current destination manifest through the real reader, so stub it.
    resolution = SimpleNamespace(manifest=None)
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda name, repository: resolution)
    with pytest.raises(ValueError, match="equal to or an ancestor"):
        source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True)


def test_retirement_target_fetch_unshallows_older_install(tmp_path, monkeypatch):
    """A depth-1 install OLDER than the qualified target follows the retirement."""
    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    labels = ("installed", "target")
    commits = []
    for label in labels:
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
        commits.append(git(seed, "rev-parse", "HEAD"))
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", commits[0] + ":refs/heads/old")
    fresh = tmp_path / "fresh"
    # A depth-1 clone pinned at the OLDER commit (the installer shape).
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    assert subprocess.run(["git", "cat-file", "-e", commits[1] + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0
    request = {"commit": commits[1], "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    # The guard reads the current destination manifest; stub it so the test
    # never touches the live channel host.
    monkeypatch.setattr(source_releases, "_resolve_channel",
                        lambda name, repository: SimpleNamespace(manifest=None))
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True


def test_retirement_unshallows_grafted_divergence_before_refusing(tmp_path, monkeypatch):
    """A depth-limited fetch can leave the pinned target grafted with its parents
    unfetched: an OLDER HEAD then reads as divergent. The apply path refills the
    history (custody lane) and re-judges before refusing; the passive check fails
    open to the unverified answer instead."""
    from hermes_cli import source_releases, update_custody

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    labels = ("installed", "middle", "target")
    commits = []
    for label in labels:
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
        commits.append(git(seed, "rev-parse", "HEAD"))
    installed, middle, target = commits
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    fresh = tmp_path / "fresh"
    # The installer shape: a depth-1 clone pinned at the OLDER commit, then a
    # depth-1 fetch materializes the pinned target as a second graft: target
    # present, the connecting `middle` commit absent.
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "fetch", "--depth=1", "origin", target)
    git(fresh, "checkout", "--detach")
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode == 0
    assert subprocess.run(["git", "cat-file", "-e", middle + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0
    # Both transports see the grafted shape as "divergent" before the refill.
    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}

    fetch_args = []
    real_run_git = update_custody.run_git

    def spying_run_git(git_cmd, args, **kwargs):
        if args[:1] == ["fetch"]:
            fetch_args.append(args)
        return real_run_git(git_cmd, args, **kwargs)

    monkeypatch.setattr(update_custody, "run_git", spying_run_git)

    # Passive: cannot decide, fails open without fetching.
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, False) is False
    assert fetch_args == []

    # Apply: refill, then prove the older install safe.
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True
    assert any("--unshallow" in args for args in fetch_args)


def test_retirement_recovers_on_first_attempt_after_initially_missing_target(tmp_path, monkeypatch):
    """A target absent at entry is fetched by the pinned fetch, but a stale
    depth-1 boundary (from an earlier check of an intermediate build) can still
    disconnect the histories, so the post-fetch classification must refill the
    grafted history before refusing — the first attempt proves the older
    install instead of requiring a second invocation (#128305 F1)."""
    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    labels = ("installed", "intermediate", "target")
    commits = []
    for label in labels:
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
        commits.append(git(seed, "rev-parse", "HEAD"))
    installed, intermediate, target = commits
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    fresh = tmp_path / "fresh"
    # The installer shape: a depth-1 clone pinned at the OLDER commit.
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    # A previous depth-1 channel check while the remote sat at the intermediate
    # build leaves a second shallow boundary on the checkout.
    git(fresh, "fetch", "--depth=1", "origin", intermediate)
    # The qualified target is initially absent locally.
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0
    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    monkeypatch.setattr(source_releases, "_resolve_channel",
                        lambda name, repository: SimpleNamespace(manifest=None))
    # First attempt: fetch the pinned target, refill the surviving boundary,
    # and prove the older install safe — no second invocation required.
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True
    assert git(fresh, "rev-parse", "HEAD") == installed


def test_retirement_existing_target_refill_clears_stale_lock(tmp_path, monkeypatch):
    """The grafted-history refill rides the same guarded preparation as the
    pinned-target fetch: an abandoned ``shallow.lock`` (older than the age
    floor, no live git) is cleared before the fetch instead of failing every
    attempt with exit 128 (#128305 F2)."""
    import os
    import time as _time

    import hermes_cli.gitlock as gitlock
    from hermes_cli import source_releases

    # Pin the declared "no live git" precondition: clear_stale_git_locks uses a
    # machine-global pgrep/tasklist guard and skips the whole sweep when any git
    # process is running, so under the canonical parallel per-file runner an
    # unrelated test's git subprocess would otherwise make this fixture
    # nondeterministically keep the lock and fail the proof. The guard's own
    # behavior is pinned separately in test_gitlock.py.
    monkeypatch.setattr(gitlock, "_git_proc_running", lambda: False)

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    labels = ("installed", "middle", "target")
    commits = []
    for label in labels:
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
        commits.append(git(seed, "rev-parse", "HEAD"))
    installed, middle, target = commits
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "fetch", "--depth=1", "origin", target)
    git(fresh, "checkout", "--detach")
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode == 0
    assert subprocess.run(["git", "cat-file", "-e", middle + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0
    # Abandoned lock debris: older than the guarded age floor, no live git.
    lock = fresh / ".git" / "shallow.lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (_time.time() - 3600,) * 2)
    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    monkeypatch.setattr(source_releases, "_resolve_channel",
                        lambda name, repository: SimpleNamespace(manifest=None))
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True
    assert not lock.exists()
    assert git(fresh, "rev-parse", "HEAD") == installed


def test_retirement_admits_install_sitting_on_the_qualified_target(tmp_path):
    """Equal-and-equal: an install already on the qualified build proves safe without a fetch."""
    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    (seed / "content.txt").write_text("target", encoding="utf-8")
    git(seed, "add", "content.txt")
    git(seed, "commit", "-m", "target")
    target = git(seed, "rev-parse", "HEAD")
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    assert git(fresh, "rev-parse", "HEAD") == target
    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True


def test_retirement_passive_check_never_fetches(tmp_path, monkeypatch):
    """A passive resolution answers without fetching or subprocess git calls."""
    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    labels = ("installed", "target")
    commits = []
    for label in labels:
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
        commits.append(git(seed, "rev-parse", "HEAD"))
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", commits[0] + ":refs/heads/old")
    checkout = tmp_path / "checkout"
    # A depth-1 clone pinned at the OLDER commit (the installer shape).
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(checkout))
    git(checkout, "checkout", "--detach")
    assert subprocess.run(["git", "cat-file", "-e", commits[1] + "^{commit}"], cwd=checkout,
                          capture_output=True).returncode != 0

    seen = []
    def no_fetch(argv, *args, **kwargs):
        seen.append(argv)
        if "fetch" in argv[:2]:
            pytest.fail(f"the passive path must not fetch: {argv}")
        return subprocess.CompletedProcess(args=argv, returncode=1, stdout="", stderr="")
    monkeypatch.setattr(source_releases.subprocess, "run", no_fetch)
    request = {"commit": commits[1], "sourceVersion": "1.0.1", "sequence": 2,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 2}}
    # The target is invisible locally: permissive with the ancestry flagged unverified.
    assert source_releases._retirement_commit_proof(request, terminal, ["git"], checkout, False) is False
    # The strict (apply) path would have fetched this same shape:
    assert any("rev-list" in argv for argv in seen)


def test_retirement_stamp_outside_admitted_identities_is_unverified(tmp_path):
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.0.0", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    # No version file and no Git: nothing proves ordering in either
    # direction, and the stamp names a commit outside the admitted
    # identities. The strict apply path refuses the unverified rollback
    # instead of authorizing it; the passive checkers keep the unverified
    # permissive answer instead of stranding the ZIP/desktop mode.
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is False


def test_retirement_apply_fetch_runs_under_custody(monkeypatch, tmp_path):
    """The pinned-target fetch must ride the updater's custody runner, not a bare
    ``subprocess.run``: on Windows a killed updater would orphan a child git that is
    outside the kill-on-close job while the checkout lock is already released."""
    from hermes_cli import source_releases, update_custody

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Release Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "content.txt").write_text("installed", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "installed")
    request = {"commit": "a" * 40, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}

    calls = []

    def custody_run_git(git_cmd, args, **kwargs):
        calls.append(args)
        if args[:1] == ["fetch"]:
            # Simulate a fetch that downloads nothing verifiable (rc 0, empty).
            return subprocess.CompletedProcess(args=[*git_cmd, *args], returncode=0,
                                               stdout="", stderr="")
        result = subprocess.run([*git_cmd, *args], cwd=kwargs.get("cwd"),
                                capture_output=True, text=True)
        return subprocess.CompletedProcess(args=[*git_cmd, *args], returncode=result.returncode,
                                           stdout=result.stdout, stderr=result.stderr)

    # The destination-manifest guard reads through the real channel reader; stub it.
    monkeypatch.setattr(source_releases, "_resolve_channel",
                        lambda name, repository: SimpleNamespace(manifest=None))
    monkeypatch.setattr(update_custody, "run_git", custody_run_git)
    with pytest.raises(ValueError, match="not newer"):
        source_releases._git_retirement_proof(request, terminal, ["git"], root, True)
    assert calls and calls[0][:1] == ["fetch"]


def test_retirement_destination_reread_only_when_destination_is_newer(monkeypatch, tmp_path):
    """The destination re-read can only refuse; it must not gate the answer on a
    second publication fetch when the destination has not moved past the build."""
    from hermes_cli import source_releases

    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Release Fixture")
    git(root, "config", "user.email", "fixture@example.invalid")
    (root / "content.txt").write_text("installed", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "installed")
    request = {"commit": "a" * 40, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}

    rereads = []

    def failing_reread(name, repository):
        rereads.append(name)
        raise AssertionError("no destination re-read when the sequence is not newer")

    monkeypatch.setattr(source_releases, "_resolve_channel", failing_reread)
    # Equal sequence: the first read already supplied the pinned target.
    assert source_releases._git_retirement_proof(request, terminal, ["git"], root, False) is False
    assert rereads == []

    # A strictly newer destination re-enables the guard; a failing re-read
    # there degrades to the passive answer instead of raising.
    monkeypatch.setattr(
        source_releases, "_resolve_channel",
        lambda name, repository: (_ for _ in ()).throw(OSError("publication read failed")))
    newer = {"name": "stable", "head": {"sequence": 2}}
    assert source_releases._git_retirement_proof(request, newer, ["git"], root, False) is False


def test_channel_compare_branch_resolves_retirement_passively(tmp_path, monkeypatch, capsys):
    """``hermes update --check``'s channel verdict must not fetch the pinned target:
    the strict fetch belongs to the update that applies the retirement."""
    from hermes_cli import source_releases, update_cmd_check, update_custody

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    (seed / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n', encoding="utf-8")
    git(seed, "add", "pyproject.toml")
    git(seed, "commit", "-m", "installed")
    installed = git(seed, "rev-parse", "HEAD")
    (seed / "content.txt").write_text("target", encoding="utf-8")
    git(seed, "add", "content.txt")
    git(seed, "commit", "-m", "target")
    target = git(seed, "rev-parse", "HEAD")
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    # A depth-1 clone pinned at the OLDER commit (the installer shape).
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode != 0

    record = {"name": "stable", "repository": "file:///fixture/origin.git",
              "state": "retired", "policy": "preview"}
    manifest = {"request": {"commit": target, "sourceVersion": "1.0.0",
                            "buildId": "b", "sequence": 1,
                            "repository": "file:///fixture/origin.git"}}
    resolution = SimpleNamespace(requested=record, terminal=record, manifest=manifest)
    monkeypatch.setattr(source_releases, "_resolve_channel", lambda name, repository: resolution)
    monkeypatch.setattr(source_releases, "source_repository",
                        lambda git_cmd, cwd: "file:///fixture/origin.git")

    fetches = []
    real_run_git = update_custody.run_git

    def spying_run_git(git_cmd, args, **kwargs):
        if args[:1] == ["fetch"]:
            fetches.append(args)
        return real_run_git(git_cmd, args, **kwargs)

    monkeypatch.setattr(update_custody, "run_git", spying_run_git)

    assert update_cmd_check.channel_compare_branch("stable", ["git"], fresh) is None
    out = capsys.readouterr().out
    # The pinned-commit verdict prints without any fetch: the strict fetch of
    # the pinned retirement target belongs to the update that applies it.
    assert "Update channel: stable" in out and "Selected release available" in out
    assert fetches == []


@pytest.mark.parametrize("channel", ["stable", "canary"])
@pytest.mark.parametrize("start", ["old", "ahead", "local"])
def test_source_check_and_apply_land_on_selected_release(releases, monkeypatch, capsys, channel, start):
    if start != "old":
        git(releases.root, "checkout", "-b", "my-work", releases.commits[3])
    if start == "local":
        (releases.root / "my-work.txt").write_text("committed local work\n", encoding="utf-8")
        git(releases.root, "add", ".")
        git(releases.root, "commit", "-m", "local work")
        (releases.root / "notes.txt").write_text("uncommitted notes\n", encoding="utf-8")
    branch_sha = git(releases.root, "rev-parse", "HEAD")
    set_install_channel(channel, releases.root)
    before = git(releases.root, "rev-parse", "HEAD")
    update_cmd._cmd_update_check()
    assert releases.tags[channel] in capsys.readouterr().out
    assert git(releases.root, "rev-parse", "HEAD") == before

    # Exercise the real selection/fetch/checkout path, not dependency installation
    # or live service management. No host OS is simulated.
    opts = update_cmd._UpdateOptions(
        pre_update_version=None, gw_input_fn=None,
        assume_yes=True, keep_stash=False, switch_branch=False, discard_local_changes=False,
    )
    monkeypatch.setattr(update_cmd, "_resolve_update_options", lambda *_: opts)
    monkeypatch.setattr(update_cmd, "_begin_update_receipt_and_plan", lambda *_: None)
    monkeypatch.setattr(main, "_run_pre_update_backup", lambda *_: None)
    monkeypatch.setattr(main, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(update_cmd, "_prepare_git_command", lambda **_: (False, ["git"], False))
    applied = []
    monkeypatch.setattr(update_cmd, "_complete_source_update", lambda request: applied.append(request))
    args = SimpleNamespace(branch=None, channel=None, force_venv=True)
    update_cmd._cmd_update_impl(args, False)
    expected = releases.commits[1 if channel == "stable" else 2]
    assert len(applied) == 1
    assert applied[0]["expected_sha"] == expected
    assert applied[0]["source"] == str(releases.root.resolve())
    assert git(releases.root, "rev-parse", "HEAD") == expected
    if start != "old":
        assert git(releases.root, "rev-parse", "my-work") == branch_sha
    if start == "local":
        assert (releases.root / "notes.txt").read_text(encoding="utf-8") == "uncommitted notes\n"
    update_cmd._cmd_update_check()
    assert "Up to date with" in capsys.readouterr().out


def test_source_check_honors_transient_channel_without_rewriting_record(releases, capsys):
    set_install_channel("stable", releases.root)
    args = SimpleNamespace(branch=None, channel="canary", check=True)
    main.cmd_update(args)
    assert releases.tags["canary"] in capsys.readouterr().out
    update_cmd._cmd_update_check()
    assert releases.tags["stable"] in capsys.readouterr().out


def test_fork_origin_uses_its_own_published_release_not_the_official_pointer(releases):
    url = "https://github.com/Fixture/hermes-agent.git"
    git(releases.root, "config", "remote.origin.url", url)
    git(releases.root, "config", f"url.{releases.origin}.insteadOf", url)
    tag = releases.tags["stable"]
    git(releases.origin, "tag", "-f", tag, releases.commits[3])
    releases.responses["/repos/Fixture/hermes-agent/releases/latest"] = {
        "tag_name": tag, "draft": False, "prerelease": False,
    }
    releases.responses[f"/repos/Fixture/hermes-agent/commits/{tag}"] = {"sha": releases.commits[3]}
    assert resolve_source_release("stable", ["git"], releases.root) == (tag, releases.commits[3])
    assert not any(path.startswith("/releases/") for path in releases.requests)


def test_zip_fallback_keeps_selected_repository_and_commit(releases, monkeypatch):
    from hermes_cli import update_cmd_zip

    seen = []
    monkeypatch.setattr(update_cmd_zip, "_abort_zip_update_if_dirty_tree", lambda: None)
    class DownloadBoundary(Exception):
        pass
    def download(branch, url, target_sha=None):
        seen.append((url, target_sha))
        raise DownloadBoundary
    monkeypatch.setattr(update_cmd_zip, "_download_and_swap_zip", download)
    with pytest.raises(DownloadBoundary):
        update_cmd_zip._update_via_zip(
            SimpleNamespace(branch=None), target_sha=releases.commits[2],
            target_repository="Fixture/hermes-agent", completion_request={})
    assert seen == [(f"https://github.com/Fixture/hermes-agent/archive/{releases.commits[2]}.zip",
                     releases.commits[2])]


@pytest.mark.parametrize("git_cmd", [["git"], None], ids=["git", "no-git"])
def test_selected_draft_never_falls_back_to_other_tags(releases, git_cmd):
    releases.responses[f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags['stable']}"]["draft"] = True
    assert resolve_source_release("stable", git_cmd, releases.root) == (None, None)
    assert not any("/tags?" in path for path in releases.requests)


def test_main_check_still_uses_branch_without_release_requests(releases, capsys):
    set_install_channel("main", releases.root)
    update_cmd._cmd_update_check()
    assert "behind origin/main" in capsys.readouterr().out
    assert releases.requests == []


@pytest.mark.parametrize("channel", ["stable", "canary"])
def test_origin_tag_cannot_substitute_a_fork_commit(releases, channel):
    git(releases.origin, "tag", "-f", releases.tags[channel], releases.commits[3])
    assert resolve_source_release(channel, ["git"], releases.root) == (None, None)
    # Without git, the same selection remains pinned to the official commit.
    assert resolve_source_release(channel) == (
        releases.tags[channel], releases.commits[1 if channel == "stable" else 2],
    )


@pytest.mark.parametrize("channel", ["stable", "canary"])
def test_missing_pointers_fall_back_only_to_published_releases(releases, channel):
    releases.responses.pop("/releases/stable/release-candidates.json")
    releases.responses.pop(f"/releases/{channel}/index.html")
    published = releases.responses[f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags[channel]}"]
    releases.responses["/repos/NousResearch/hermes-agent/releases/latest"] = published
    releases.responses["/repos/NousResearch/hermes-agent/releases?per_page=100&page=1"] = [
        {"tag_name": "v99.0.1+canary.20260912T125822Z", "draft": True, "prerelease": True},
        published,
    ]
    assert resolve_source_release(channel, ["git"], releases.root) == (
        releases.tags[channel], releases.commits[1 if channel == "stable" else 2],
    )
    assert not any("/tags?" in path for path in releases.requests)
    assert f"/repos/NousResearch/hermes-agent/releases/tags/{releases.tags[channel]}" not in releases.requests


def test_retirement_strict_apply_does_not_unshallow_a_proven_safe_history(tmp_path):
    """The shared post-fetch classifier judges the fetched objects first: when
    the pinned fetch already proved ``HEAD`` an ancestor of the target, the
    verdict returns without ``--unshallow`` — a shallow boundary can survive
    the fetch while the relation is already decided, and unshallowing it would
    drag the whole blob history behind the boundary (the reason the updater
    owns ``fetch_full_commit_graph`` and its ``blob:none`` conversion)."""
    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    # Substantial history BEFORE the install point, so a refill has real mass.
    for label in ("history-1", "history-2", "installed", "target"):
        (seed / "content.txt").write_text(label, encoding="utf-8")
        git(seed, "add", "content.txt")
        git(seed, "commit", "-m", label)
    target = git(seed, "rev-parse", "HEAD")
    installed = git(seed, "rev-parse", "HEAD~1")
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    # The pinned-target fetch has run (the shape _git_retirement_proof's
    # missing-target branch hands the classifier): a full fetch of the target
    # links it to the already-present installed parent, HEAD -> target is
    # provable, and the checkout is still shallow (the install point keeps its
    # boundary for the history BEFORE it).
    git(fresh, "fetch", "--no-tags", "origin", target)
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"], cwd=fresh,
                          capture_output=True).returncode == 0
    assert subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=fresh,
                          capture_output=True, text=True).stdout.strip() == "true"

    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    assert source_releases._classify_strict_ancestry(request, ["git"], fresh,
                                                     _FixtureGit(fresh)) is True
    assert git(fresh, "rev-parse", "HEAD") == installed
    # The boundary is still there: nothing refilled the history.
    assert subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=fresh,
                          capture_output=True, text=True).stdout.strip() == "true"


class _FixtureGit:
    """The ``run_git`` seam the classifier takes, backed by real git."""

    def __init__(self, cwd):
        self.cwd = str(cwd)

    def __call__(self, *args):
        return subprocess.run(["git", *args], cwd=self.cwd, capture_output=True,
                              text=True, stdin=subprocess.DEVNULL)


def test_retirement_necessary_refill_converts_to_blob_none_not_raw_unshallow(tmp_path, monkeypatch):
    """F2 regression: the necessary-refill branch (ancestry inconclusive after
    the pinned fetch) must establish ancestry without hydrating the blob
    history behind the boundary — the filter selection of the repo's
    ``fetch_full_commit_graph`` owner: a depth-limited full clone converts to
    ``blob:none`` instead of a raw ``--unshallow``, and an existing
    partial-clone filter is preserved."""
    import hermes_cli.update_custody as update_custody

    from hermes_cli import source_releases

    origin = tmp_path / "origin.git"
    origin.mkdir()
    git(origin, "init", "--bare", "-b", "main")
    seed = tmp_path / "seed"
    seed.mkdir()
    git(seed, "clone", str(origin), str(seed))
    git(seed, "config", "user.name", "Release Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    # Substantial blob mass BEFORE the install point: a raw --unshallow would
    # hydrate all of it; the policy-aware refill must not. An intermediate
    # commit sits between the install point and the target so the depth-1
    # graft below cuts the chain and the refill is genuinely necessary.
    labels = (["history-%d" % i for i in range(1, 9)]
              + ["installed", "intermediate", "target"])
    commits = {}
    for label in labels:
        (seed / ("%s.txt" % label)).write_text(label * 2000, encoding="utf-8")
        git(seed, "add", "-A")
        git(seed, "commit", "-m", label)
        commits[label] = git(seed, "rev-parse", "HEAD")
    installed, target = commits["installed"], commits["target"]
    git(seed, "push", "origin", "HEAD:refs/heads/main")
    git(seed, "push", "origin", installed + ":refs/heads/old")

    request = {"commit": target, "sourceVersion": "1.0.0", "sequence": 1,
               "repository": "NousResearch/hermes-agent"}
    terminal = {"name": "stable", "head": {"sequence": 1}}

    # The installer shape: a depth-1 clone pinned at the OLDER commit, then a
    # depth-1 check of the INTERMEDIATE build grafts it with its parents
    # unfetched, so target->...->installed reads divergent — ancestry is
    # genuinely inconclusive and the refill branch is required.
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--depth=1", "--branch", "old", origin.as_uri(), str(fresh))
    git(fresh, "checkout", "--detach")
    git(fresh, "fetch", "--depth=1", "origin", commits["intermediate"])
    assert subprocess.run(["git", "cat-file", "-e", target + "^{commit}"],
                          cwd=fresh, capture_output=True).returncode != 0

    fetch_args_seen = []
    real_run_git = update_custody.run_git

    def spying_run_git(git_cmd, args, **kwargs):
        if args[:1] == ["fetch"]:
            fetch_args_seen.append(args)
        return real_run_git(git_cmd, args, **kwargs)

    monkeypatch.setattr(update_custody, "run_git", spying_run_git)

    assert source_releases._git_retirement_proof(request, terminal, ["git"], fresh, True) is True
    unshallow_fetches = [args for args in fetch_args_seen if "--unshallow" in args]
    assert unshallow_fetches, "the inconclusive shape must enter the refill branch"
    for args in unshallow_fetches:
        # The policy conversion, not a raw hydrating --unshallow.
        assert "--filter=blob:none" in args, (
            "the refill must convert the depth-limited full clone to blob:none: %r" % (args,))
    # The conversion is recorded exactly as the gitlock owner records it.
    assert subprocess.run(["git", "config", "--get", "remote.origin.promisor"],
                          cwd=fresh, capture_output=True, text=True).stdout.strip() == "true"
    assert subprocess.run(["git", "config", "--get", "remote.origin.partialclonefilter"],
                          cwd=fresh, capture_output=True,
                          text=True).stdout.strip() == "blob:none"
    assert git(fresh, "rev-parse", "--is-shallow-repository") == "false"

    # An existing partial-clone filter is preserved by the refill lane. (On a
    # promisor clone the end-to-end probes lazily complete the graph, so the
    # inconclusive branch is exercised through the refill entry point.)
    partial = tmp_path / "partial"
    git(tmp_path, "clone", "--depth=1", "--filter=blob:limit=1k", "--branch", "old",
        origin.as_uri(), str(partial))
    git(partial, "checkout", "--detach")
    git(partial, "fetch", "--depth=1", "origin", commits["intermediate"])
    configured = subprocess.run(["git", "config", "--get", "remote.origin.partialclonefilter"],
                                cwd=partial, capture_output=True, text=True).stdout.strip()
    assert configured, "the fixture clone must carry a partial-clone filter"
    fetch_args_seen.clear()
    source_releases._guarded_history_refill(["git"], partial, target, 900)
    refills = [args for args in fetch_args_seen if "--unshallow" in args]
    assert refills, "the refill lane must run on a grafted partial clone too"
    for args in refills:
        assert "--filter=" + configured in args, (
            "the checkout's own filter must survive: %r" % (args,))
    assert subprocess.run(
        ["git", "config", "--get", "remote.origin.partialclonefilter"],
        cwd=partial, capture_output=True, text=True).stdout.strip() == configured
    assert source_releases._git_retirement_proof(request, terminal, ["git"], partial, True) is True


def test_retirement_no_git_strict_apply_refuses_same_version_unknown_stamp(tmp_path):
    """The strict no-Git apply path cannot treat unverified ordering as
    authorization: an install stamped with an unrelated commit is exactly the
    same-version different-build rollback the retirement pins, so it refuses
    with the explicit-destination remedy instead of applying — whatever the
    inert pyproject.toml placeholder says."""
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.2.3", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)


def test_retirement_no_git_strict_apply_refuses_regardless_of_manifest_version(tmp_path):
    """The manifest version is not ordering evidence in either direction: the
    same unknown stamp refuses with a synthetic OLDER manifest value too, so
    the refusal provably comes from the stamp authority, not a version read."""
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.2.3", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n', encoding="utf-8")
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "c" * 40}), encoding="utf-8")
    with pytest.raises(ValueError, match="select the destination channel explicitly"):
        _retirement_commit_proof(request, terminal, None, tmp_path, True)


def test_retirement_no_git_strict_apply_admits_exact_target_stamp(tmp_path):
    """The stated policy boundary: an install stamp naming the pinned target
    commit is positive proof, the only ordering authority this transport has,
    and it admits the apply on both strict and passive paths."""
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.2.3", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "a" * 40}), encoding="utf-8")
    assert _retirement_commit_proof(request, terminal, None, tmp_path, True) is True
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is True


def test_retirement_no_git_passive_check_stays_permissive_on_same_version(tmp_path):
    """The passive checker (source_check, strict=False) keeps the unverified
    permissive answer even for the same-version shape — only the apply path
    refuses."""
    from hermes_cli.source_releases import _retirement_commit_proof

    request = {"commit": "a" * 40, "sourceVersion": "1.2.3", "sequence": 1}
    terminal = {"name": "stable", "head": {"sequence": 1}}
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")
    (tmp_path / "install-stamp.json").write_text(
        json.dumps({"commit": "c" * 40}), encoding="utf-8")
    assert _retirement_commit_proof(request, terminal, None, tmp_path, False) is False
