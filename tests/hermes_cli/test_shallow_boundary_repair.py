import subprocess
from pathlib import Path

import hermes_cli.gitlock as gitlock


def git(repo, *args, check=True):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


def fixture(tmp_path):
    origin = tmp_path / "origin"; origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    git(origin, "config", "user.email", "t@example.com"); git(origin, "config", "user.name", "t")
    git(origin, "commit", "--allow-empty", "-qm", "c0")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{origin}", str(clone)], check=True)
    git(origin, "commit", "--allow-empty", "-qm", "c1")
    git(origin, "commit", "--allow-empty", "-qm", "c2")
    git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    git(origin, "commit", "--allow-empty", "-qm", "c3")
    git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    return clone


def corrupt_fixture(clone):
    path = clone / ".git" / "shallow"
    lines = path.read_text().splitlines()
    for removed in lines:
        path.write_text("\n".join(x for x in lines if x != removed) + "\n")
        fsck = git(clone, "fsck", "--connectivity-only", check=False)
        if "broken link" in (fsck.stdout + fsck.stderr) or "missing commit" in (fsck.stdout + fsck.stderr):
            return removed
    raise AssertionError("fixture did not create a broken shallow boundary")


def test_repair_restores_boundary_for_reflog_only_commit_with_unfetched_parent(tmp_path):
    clone = fixture(tmp_path)
    corrupt_fixture(clone)
    assert git(clone, "rev-list", "--count", "--all", "--reflog", check=False).returncode != 0
    assert "broken link" in git(clone, "fsck", "--connectivity-only", check=False).stdout
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert git(clone, "rev-list", "--count", "--all", "--reflog").returncode == 0
    fsck = git(clone, "fsck", "--connectivity-only")
    assert "broken link" not in (fsck.stdout + fsck.stderr)
    assert git(clone, "gc", "-q").returncode == 0


def test_repair_is_noop_on_healthy_shallow_checkout(tmp_path):
    clone = fixture(tmp_path); path = clone / ".git" / "shallow"; before = path.read_bytes()
    assert gitlock.repair_broken_shallow_boundaries(clone) == 0
    assert path.read_bytes() == before


def test_repair_is_noop_on_full_clone_without_shallow_file(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir(); git(repo, "init", "-q")
    assert gitlock.repair_broken_shallow_boundaries(repo) == 0


def test_repair_never_raises_on_broken_repo(tmp_path):
    assert gitlock.repair_broken_shallow_boundaries(tmp_path / "missing") == 0


def test_repair_does_not_touch_reflogs(tmp_path):
    clone = fixture(tmp_path); corrupt_fixture(clone)
    before = git(clone, "reflog", "show", "--all").stdout
    gitlock.repair_broken_shallow_boundaries(clone)
    assert git(clone, "reflog", "show", "--all").stdout == before


def test_repair_is_idempotent(tmp_path):
    clone = fixture(tmp_path); corrupt_fixture(clone)
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert gitlock.repair_broken_shallow_boundaries(clone) == 0


def test_repair_uses_a_bounded_number_of_git_subprocesses(tmp_path, monkeypatch):
    counts = []
    real_run = gitlock.subprocess.run

    for count in (4, 40):
        root = tmp_path / str(count)
        root.mkdir()
        clone = fixture(root)
        for index in range(count):
            git(clone, "commit", "--allow-empty", "-qm", f"extra-{index}")
        corrupt_fixture(clone)
        calls = 0

        def counting_run(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_run(*args, **kwargs)

        monkeypatch.setattr(gitlock.subprocess, "run", counting_run)
        assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
        assert git(clone, "rev-list", "--count", "--all", "--reflog").returncode == 0
        counts.append(calls)
        monkeypatch.setattr(gitlock.subprocess, "run", real_run)

    assert all(count < 15 for count in counts)
    assert abs(counts[0] - counts[1]) <= 2


def test_repair_handles_commit_messages_containing_blank_lines(tmp_path):
    clone = fixture(tmp_path)
    git(clone, "commit", "--allow-empty", "-m", "subject\n\nbody line\n\ndeadbeef commit 123")
    corrupt_fixture(clone)
    assert gitlock.repair_broken_shallow_boundaries(clone) >= 1
    assert git(clone, "rev-list", "--count", "--all", "--reflog").returncode == 0
    fsck = git(clone, "fsck", "--connectivity-only")
    assert "broken link" not in (fsck.stdout + fsck.stderr)
