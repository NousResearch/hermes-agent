"""Boot-time post-update bootstrap: identity, records, locks, single-flight.

The record files are an optimization layer over idempotent steps; these
tests assert the contracts that keep that safe: identity resolution from
real git trees and stamps, record scoping (per-install AND per-home), and the lock protocol including the double-check under lock.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import boot_bootstrap
from hermes_cli.boot_bootstrap import (
    _RecordLock,
    current_install_identity,
    needs_bootstrap,
    read_git_head,
    read_last_known,
    record_path,
    run_boot_bootstrap,
    _write_record,
)


def _git(args, cwd):
    env = dict(os.environ)
    env.update({
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
    })
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(["init", "-b", "main"], root)
    (root / "f.txt").write_text("1", encoding="utf-8")
    _git(["add", "."], root)
    _git(["commit", "-m", "one"], root)
    return root


def _head_sha(root):
    return _git(["rev-parse", "HEAD"], root).stdout.strip()


# ── read_git_head ────────────────────────────────────────────────────


def test_git_selection_uses_pm_public_package_reader(repo, monkeypatch):
    from types import SimpleNamespace
    import pm

    command = shutil.which("git")
    assert command is not None
    binary = Path(command)
    requests = []

    def installed(name):
        requests.append(name)
        return SimpleNamespace(binary=binary)

    monkeypatch.setattr(pm, "installed_package", installed)

    assert read_git_head(repo) == _head_sha(repo)
    assert requests == ["git"]


def test_read_git_head_detached(repo):
    sha = _head_sha(repo)
    _git(["checkout", "--detach", sha], repo)
    assert read_git_head(repo) == sha


def test_read_git_head_packed_refs(repo):
    sha = _head_sha(repo)
    _git(["pack-refs", "--all"], repo)
    # Loose ref is gone; only packed-refs carries the branch now.
    assert not (repo / ".git" / "refs" / "heads" / "main").exists()
    assert read_git_head(repo) == sha


def test_read_git_head_worktree_gitfile(repo, tmp_path):
    wt = tmp_path / "wt"
    _git(["worktree", "add", str(wt)], repo)
    assert (wt / ".git").is_file()  # gitfile pointer, not a directory
    assert read_git_head(wt) == _head_sha(wt)


def test_read_git_head_reftable(tmp_path):
    """The repo format that kills hand-rolled .git parsers.

    A reftable repo stores refs in neither loose files nor packed-refs,
    and its HEAD is a decoy (``ref: refs/heads/.invalid``) kept only so
    pre-reftable tools fail loudly instead of misreading. Asking git
    answers. If git here is too old for reftable, nothing to test.
    """
    root = tmp_path / "rt"
    root.mkdir()
    try:
        _git(["init", "--ref-format=reftable", "-b", "main", "."], root)
    except subprocess.CalledProcessError:
        pytest.skip("git too old for --ref-format=reftable")
    (root / "f.txt").write_text("1", encoding="utf-8")
    _git(["add", "."], root)
    _git(["commit", "-m", "one"], root)

    head = (root / ".git" / "HEAD").read_text(encoding="utf-8")
    assert ".invalid" in head, "reftable decoy HEAD is the point of this test"

    assert read_git_head(root) == _head_sha(root)


def test_read_git_head_missing_and_garbage(tmp_path):
    assert read_git_head(tmp_path) is None
    (tmp_path / ".git").write_text("not a gitdir pointer", encoding="utf-8")
    assert read_git_head(tmp_path) is None


# ── current_install_identity ─────────────────────────────────────────


def test_identity_broken_tree_is_none(tmp_path):
    assert current_install_identity(tmp_path) is None
    (tmp_path / "install-stamp.json").write_text("garbage", encoding="utf-8")
    assert current_install_identity(tmp_path) is None


# ── record paths ─────────────────────────────────────────────────────


def test_record_paths_key_on_install_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    a = record_path(tmp_path / "install-a")
    b = record_path(tmp_path / "install-b")
    assert a != b
    # The key is a FOLDER (installs/<SHA16>/bootstrap/<profile>.json),
    # not a filename suffix: same grandparent tree, different key dirs.
    assert a.parent != b.parent
    assert a.parent.parent.parent == b.parent.parent.parent  # installs/
    assert a.name == b.name  # the profile filename is the shared part


@pytest.mark.skipif(
    os.name == "nt", reason="requires symlink privilege on Windows"
)
def test_symlinked_root_canonicalizes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    real = tmp_path / "real-install"
    real.mkdir()
    link = tmp_path / "link-install"
    link.symlink_to(real)
    assert record_path(real) == record_path(link)


# ── needs_bootstrap ──────────────────────────────────────────────────


def test_needs_bootstrap_broken_tree_never_fires(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert needs_bootstrap(tmp_path / "nope") is None
    assert run_boot_bootstrap(tmp_path / "nope") == {"home": "skipped"}
    assert not record_path(tmp_path / "nope").exists()


# ── lock protocol ────────────────────────────────────────────────────


def test_lock_loser_skips(tmp_path):
    record = tmp_path / "r.json"
    first = _RecordLock(record)
    second = _RecordLock(record)
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_stale_lock_is_broken(tmp_path):
    record = tmp_path / "r.json"
    lock_path = record.with_name(record.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": 1, "startedAt": time.time() - 3600}), encoding="utf-8"
    )
    lock = _RecordLock(record)
    assert lock.acquire()
    lock.release()


def test_fresh_lock_is_respected(tmp_path):
    record = tmp_path / "r.json"
    lock_path = record.with_name(record.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "startedAt": time.time()}), encoding="utf-8"
    )
    assert not _RecordLock(record).acquire()


# ── run_boot_bootstrap ───────────────────────────────────────────────


@pytest.fixture
def fake_steps(monkeypatch):
    calls = {"home": 0}

    def home_step():
        calls["home"] += 1
        return {"ok": True}

    from hermes_cli import post_update

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("h", home_step),))
    return calls


def test_double_check_under_lock(repo, tmp_path, monkeypatch, fake_steps):
    """A racer that finished between our read and our acquire wins."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    sha = _head_sha(repo)

    real_acquire = _RecordLock.acquire

    def acquire_after_racer_finished(self):
        got = real_acquire(self)
        if got and self.path.name.endswith(".json.lock"):
            # Simulate the previous holder completing just before us.
            _write_record(record_path(repo), sha, {})
        return got

    monkeypatch.setattr(_RecordLock, "acquire", acquire_after_racer_finished)
    result = run_boot_bootstrap(repo)
    assert result["home"] == "done-by-other"
    assert fake_steps["home"] == 0


def test_maybe_run_never_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        boot_bootstrap, "run_boot_bootstrap",
        lambda root: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    boot_bootstrap.maybe_run_boot_bootstrap(tmp_path)  # must not raise


def test_sealed_tree_bootstrap_end_to_end(tmp_path, monkeypatch):
    """The desktop-bundle-swap scenario. A sealed tree (install-stamp.json,
    no .git) must bootstrap on first boot, no-op on the second, and RE-RUN
    when the stamp's commit changes — that is the only signal a bundle
    swap emits."""
    import json as _json

    from hermes_cli import post_update

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    sealed = tmp_path / "payload"
    sealed.mkdir()
    (sealed / "install-stamp.json").write_text(
        _json.dumps({"commit": "aaaa1111", "payload": "full", "updateMechanism": "electron-updater"})
    )

    calls = {"n": 0}

    def count():
        calls["n"] += 1
        return {"ok": True}

    monkeypatch.setattr(post_update, "BOOT_HOME_STEPS", (("h", count),))

    assert run_boot_bootstrap(sealed)["home"] != "skipped"
    assert calls["n"] == 1
    assert run_boot_bootstrap(sealed)["home"] == "skipped"
    assert calls["n"] == 1  # second boot: identity unchanged, no work

    # The bundle swap: same root, new stamp commit.
    (sealed / "install-stamp.json").write_text(
        _json.dumps({"commit": "bbbb2222", "payload": "full", "updateMechanism": "electron-updater"})
    )

    assert run_boot_bootstrap(sealed)["home"] != "skipped"
    assert calls["n"] == 2, "a swapped bundle must re-run the bootstrap"
