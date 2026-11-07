"""A source checkout publishes identity only after successful completion."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.source_completion import complete_source_checkout
from hermes_cli.source_stamp import write_source_stamp


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Hermes Test")
    git("config", "user.email", "hermes@example.invalid")
    (root / "tracked").write_text("release\n", encoding="utf-8")
    git("add", "tracked")
    git("commit", "-qm", "release")
    git("tag", "v0.21.4")
    return root


def _completion_dependencies(monkeypatch, maintenance):
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda root: None)
    monkeypatch.setattr("hermes_cli.source_build.build_update_products", lambda root, *, desktop: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._run_post_update_maintenance", maintenance)


def test_successful_source_completion_writes_checkout_identity(tmp_path, monkeypatch):
    root = _repo(tmp_path)

    def maintenance(**_kwargs):
        assert not (root / "install-stamp.json").exists()
        return True

    _completion_dependencies(monkeypatch, maintenance)

    assert complete_source_checkout(root, desktop=False, assume_yes=True)
    assert (root / "install-stamp.json").is_file()


def test_failed_source_completion_does_not_publish_identity(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    _completion_dependencies(monkeypatch, lambda **_kwargs: True)

    def broken_build(root, *, desktop):
        raise RuntimeError("web UI build: npm exited 1")

    monkeypatch.setattr("hermes_cli.source_build.build_update_products", broken_build)
    owed: list = []
    assert not complete_source_checkout(root, desktop=False, assume_yes=True, followups=owed)
    # The tail is owed, so the checkout identity is not published.
    assert [step for step, _ in owed] == ["build"]
    assert not (root / "install-stamp.json").exists()


def test_unsafe_sqlite_withholds_success_but_not_the_finished_tail(tmp_path, monkeypatch):
    # Was: an unsafe SQLite verdict also withheld the stamp, so the tail stayed owed forever
    # (re-running it cannot fix the interpreter, codemap §8 V1). The install still reports failure.
    root = _repo(tmp_path)
    _completion_dependencies(monkeypatch, lambda **_kwargs: False)
    owed: list = []
    assert not complete_source_checkout(root, desktop=False, assume_yes=True, followups=owed)
    assert owed == []
    assert (root / "install-stamp.json").is_file()


def _sandboxed_marker(tmp_path, monkeypatch):
    """Point every update-lock reader at a marker inside this test's sandbox."""
    marker = tmp_path / ".hermes-update-in-progress"
    monkeypatch.setattr("hermes_cli.update_lock.update_marker_path", lambda: marker)
    return marker


@pytest.fixture
def foreign_pid():
    """A live process that is not an ancestor of this test: a stand-in updater."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                            stdin=subprocess.DEVNULL)
    yield proc.pid
    proc.kill()
    proc.wait()


def test_completion_holds_the_update_lock_during_the_tail(tmp_path, monkeypatch):
    """Two completions must not build the same products concurrently (#123376).

    A gateway restarted while an update's tail runs reaches this same function
    through venv_sync; without the shared lock both build the same output dirs
    and race on install-stamp.json.
    """
    root = _repo(tmp_path)
    marker = _sandboxed_marker(tmp_path, monkeypatch)
    seen = {}

    def build(_root, *, desktop):
        from hermes_cli.update_lock import read_live_update

        seen["holder"] = read_live_update()

    monkeypatch.setattr("hermes_cli.source_build.build_update_products", build)
    monkeypatch.setattr("hermes_cli.venv_sync.publish_launchers", lambda root: None)
    monkeypatch.setattr("hermes_cli.update_cmd_maint._run_post_update_maintenance", lambda **_kwargs: True)

    assert complete_source_checkout(root, desktop=False, assume_yes=True)

    assert seen["holder"] is not None, "the tail must hold the shared lock while it builds"
    assert seen["holder"].pid == os.getpid()
    assert not marker.exists(), "a completed tail must release the lock for the next launch"


def test_completion_refuses_to_stack_on_a_live_update(tmp_path, monkeypatch, foreign_pid):
    root = _repo(tmp_path)
    marker = _sandboxed_marker(tmp_path, monkeypatch)
    marker.write_text(f"{foreign_pid}\n{int(time.time())}\n", encoding="utf-8")
    _completion_dependencies(monkeypatch, lambda **_kwargs: True)

    with pytest.raises(RuntimeError, match="an update is still running"):
        complete_source_checkout(root, desktop=False, assume_yes=True)


def test_completion_runs_under_a_parents_claim_without_releasing_it(tmp_path):
    """A tail spawned by an orchestrator that holds the lock (venv_sync's
    interrupted-update finish, the updater's own completion child) runs under
    its parent's claim and leaves the parent's marker untouched.

    Real topology: a parent process claims, then runs the tail as its child. The test
    process's own parent can be invisible (a pid-namespaced runner), so it is never used."""
    root = _repo(tmp_path)
    marker = tmp_path / ".hermes-update-in-progress"
    tail = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "from pathlib import Path\n"
        "import hermes_cli.source_build, hermes_cli.update_cmd_maint, hermes_cli.update_lock, hermes_cli.venv_sync\n"
        "from hermes_cli.source_completion import complete_source_checkout\n"
        "hermes_cli.venv_sync.publish_launchers = lambda root: None\n"
        "hermes_cli.source_build.build_update_products = lambda root, *, desktop: None\n"
        "hermes_cli.update_cmd_maint._run_post_update_maintenance = lambda **_kwargs: True\n"
        f"hermes_cli.update_lock.update_marker_path = lambda: Path({str(marker)!r})\n"
        f"print(json.dumps(complete_source_checkout(Path({str(root)!r}), desktop=False, assume_yes=True)))\n"
    )
    parent = (
        "import os, subprocess, sys, time\n"
        f"open({str(marker)!r}, 'w', encoding='utf-8').write(f'{{os.getpid()}}\\n{{int(time.time())}}\\n')\n"  # windows-footgun: ok — a write
        "sys.exit(subprocess.call([sys.executable, '-c', sys.argv[1]]))\n"
    )
    run = subprocess.run([sys.executable, "-c", parent, tail], capture_output=True, timeout=120,
                         text=True, encoding="utf-8", errors="replace", check=False)

    assert run.returncode == 0 and run.stdout.strip().splitlines()[-1:] == ["true"], run.stdout + run.stderr
    assert marker.exists(), "the parent still owns its claim after our tail"


def _verify_bootstrap_receipt(root: Path) -> subprocess.CompletedProcess:
    """The same verifier the Windows install/update E2E runs after an update."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "verify-bootstrap-version-stamp.py"
    return subprocess.run([sys.executable, "-B", str(script), "--stamp", str(root / ".hermes-bootstrap-complete"),
                           "--repo", str(root)], capture_output=True, text=True, encoding="utf-8", check=False)


def test_publishing_checkout_identity_moves_an_installer_receipt_to_head(tmp_path):
    # write_source_stamp is the one seam: the completion handoff, the PM updater's finish and
    # boot-time adoption all publish identity through it, and the receipt must follow every one.
    root = _repo(tmp_path)
    release = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, capture_output=True, text=True, encoding="utf-8", check=True).stdout.strip()
    # What install.sh / install.ps1's complete stage leaves behind at the installed release.
    (root / ".hermes-bootstrap-complete").write_text(json.dumps({
        "schemaVersion": 1, "pinnedCommit": release, "pinnedBranch": branch, "completedAt": "2026-06-19T00:00:00.000Z",
    }), encoding="utf-8")
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "update"], cwd=root, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
    assert write_source_stamp(root) is not None
    result = _verify_bootstrap_receipt(root)
    assert result.returncode == 0, result.stdout + result.stderr


def test_publishing_checkout_identity_never_invents_an_installer_receipt(tmp_path):
    # The receipt's presence is what marks a script install; a manual clone stays one.
    root = _repo(tmp_path)

    assert write_source_stamp(root) is not None
    assert not (root / ".hermes-bootstrap-complete").exists()


def test_shallow_checkout_publishes_its_release_after_fetching_the_commit_graph(tmp_path):
    # Pre-PM installers cloned --depth 1; the completion fetches commits (not trees) so
    # the stamp can still name the release the checkout is built on.
    from hermes_cli.gitlock import fetch_full_commit_graph

    server = _repo(tmp_path)
    subprocess.run(["git", "config", "uploadpack.allowFilter", "true"], cwd=server, check=True)
    for message in ("one", "two", "three"):
        subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", message], cwd=server, check=True,
                       capture_output=True, env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                                                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"})
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "clone", "-q", "--depth", "1", server.as_uri(), str(checkout)], check=True, capture_output=True)

    assert fetch_full_commit_graph(checkout)
    stamp = write_source_stamp(checkout)

    assert stamp is not None
    assert (stamp["baseVersion"], stamp["distance"]) == ("0.21.4", 3)
    assert not fetch_full_commit_graph(checkout)


def test_full_checkout_refreshes_release_tags_before_publishing_identity(tmp_path):
    from hermes_cli.gitlock import fetch_full_commit_graph

    server = _repo(tmp_path)
    env = {"HOME": str(tmp_path), "PATH": os.environ["PATH"]}

    def git(root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, env=env, check=True, capture_output=True, text=True,
        ).stdout.strip()

    git(server, "config", "uploadpack.allowFilter", "true")
    git(server, "tag", "-d", "v0.21.4")
    versions = (("v2026.9.7", "0.21.1"), ("v2026.9.24", "0.21.5"))
    for tag, version in versions:
        (server / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
        git(server, "add", "pyproject.toml")
        git(server, "commit", "-qm", "release")
        git(server, "tag", tag)
    git(server, "commit", "-q", "--allow-empty", "-m", "after release")
    checkout = tmp_path / "checkout"
    git(server, "clone", "-q", "--no-tags", server.as_uri(), str(checkout))
    old_tag, old_version = versions[0]
    git(checkout, "fetch", "-q", "origin", f"refs/tags/{old_tag}:refs/tags/{old_tag}")
    commit = git(checkout, "rev-parse", "HEAD")
    assert git(checkout, "rev-parse", "--is-shallow-repository") == "false"
    before = write_source_stamp(checkout)
    assert before is not None
    assert before["baseVersion"] == old_version

    fetch_full_commit_graph(checkout)
    stamp = write_source_stamp(checkout)

    assert stamp is not None
    assert (stamp["baseVersion"], stamp["distance"]) == (versions[1][1], 1)
    assert stamp["commit"] == commit == git(checkout, "rev-parse", "HEAD")
