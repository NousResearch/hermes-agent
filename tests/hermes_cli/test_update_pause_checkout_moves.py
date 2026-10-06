"""Every checkout-writing step binds the paused gateways' tree gate first (#132338 R3, thread 5).

The Windows pause records what "the tree before this update" was, and the gate that lets paused
gateways start holds them while git may have left a half-written tree at an unmoved HEAD. Only the
principal pull recorded its target, so a failure in an EARLIER step (the switch onto an existing
local target branch, a fork's early upstream sync) was judged against whatever refs a later fetch
left, and a failure in a LATER step (the upstream sync after a successful pull) had no baseline at
its starting HEAD at all: both torn trees were admitted.

Real git (local repositories and a bare "upstream"), real update steps, real durable pause record
and gate. git's mid-move failure is real too: a read-only directory, as an unprivileged user.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from hermes_cli import update_cmd
from hermes_cli import update_pause_record as pause_record

pytestmark = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores read-only dirs"),
]


def git(root, *args) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit(root, message, **files) -> str:
    for name, text in files.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    monkeypatch.setattr(pause_record, "install_root", lambda: root)
    locked = root / "locked"
    yield root
    if locked.exists():
        locked.chmod(0o755)


def pause(root) -> dict:
    token = pause_record.stamp_tree({"resume_needed": True, "profiles": {"default": 999999}}, root)
    pause_record.write(token)
    return token


def gate(root) -> bool:
    """The verdict recovery reaches: the durable record, as the next launch reads it."""
    return pause_record.tree_is_whole(pause_record.read(pause_record.record_path())["token"], root)[0]


def bare_upstream(root, tmp_path, tip) -> None:
    remote = tmp_path / "upstream.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(root), str(remote)], check=True)
    git(remote, "update-ref", "refs/heads/main", tip)
    git(root, "remote", "add", "upstream", str(remote))


def test_a_torn_first_branch_switch_stays_held_after_a_ref_only_fetch(repo):
    a = commit(repo, "A", **{"a.py": "A=1\n"})
    git(repo, "checkout", "-q", "-b", "feature")  # parked on a merged branch: the update switches to main
    git(repo, "checkout", "-q", "main")
    b = commit(repo, "B", **{"a.py": "A=2\n"})
    git(repo, "update-ref", "refs/remotes/origin/main", b)
    git(repo, "checkout", "-q", "feature")
    git(repo, "branch", "-q", "-D", "main")  # no local main: the switch falls back to `checkout -B`
    token = pause(repo)
    # `checkout -B` writes the tree, then creates the branch ref: an unwritable ref directory stops
    # it between the two, as a kill would (HEAD unmoved, the target's bytes already in the tree).
    (repo / ".git" / "refs" / "heads").chmod(0o555)
    try:
        with pytest.raises(SystemExit):
            update_cmd._prepare_checkout_for_update(
                ["git"], "main", "feature", is_fork=False, assume_yes=True, gateway_mode=False,
                gw_input_fn=None, switch_branch=False, _windows_gateway_resume=token)
    finally:
        (repo / ".git" / "refs" / "heads").chmod(0o755)
    assert git(repo, "rev-parse", "HEAD") == a and (repo / "a.py").read_text() == "A=2\n", "fixture: not torn"
    c = git(repo, "commit-tree", "-p", a, "-m", "C", git(repo, "rev-parse", f"{a}^{{tree}}"))
    git(repo, "update-ref", "refs/remotes/origin/main", c)  # a later fetch: refs no longer reach a.py
    assert gate(repo) is False, "a torn switch was admitted once a fetch moved the refs"


def test_a_torn_early_upstream_sync_stays_held_after_a_ref_only_fetch(repo, tmp_path):
    a = commit(repo, "A", **{"a.py": "A=1\n", "locked__z.py": "Z=1\n"})
    b = commit(repo, "B", **{"a.py": "A=2\n", "locked__z.py": "Z=2\n"})
    bare_upstream(repo, tmp_path, b)
    git(repo, "reset", "-q", "--hard", a)
    git(repo, "update-ref", "refs/remotes/origin/main", a)  # the fork matches origin, trails upstream
    token = pause(repo)
    (repo / "locked").chmod(0o555)
    update_cmd._prepare_checkout_for_update(
        ["git"], "main", "main", is_fork=True, assume_yes=True, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, _windows_gateway_resume=token)
    assert git(repo, "rev-parse", "HEAD") == a and (repo / "a.py").read_text() == "A=2\n", "fixture: not torn"
    git(tmp_path / "upstream.git", "update-ref", "refs/heads/main", a)  # upstream moves on (here: back)
    git(repo, "fetch", "-q", "upstream", "+refs/heads/main:refs/remotes/upstream/main")  # a later real fetch
    assert gate(repo) is False, "a torn upstream sync was admitted once a fetch moved the refs"


@pytest.mark.parametrize("second_move", ["torn", "refused", "complete"])
def test_a_second_move_is_gated_from_the_head_the_first_one_reached(repo, tmp_path, second_move):
    commit(repo, "A", **{"a.py": "A=1\n", "locked__z.py": "Z=1\n", "notes.txt": "mine\n"})
    o = commit(repo, "O", **{"origin.py": "O=1\n"})
    # Refused: upstream also touches the user's file, which the user edits again after the update.
    edits = {"notes.txt": "upstream\n"} if second_move == "refused" else {}
    u = commit(repo, "U", **{"a.py": "A=2\n", "locked__z.py": "Z=2\n", "new.py": "N=1\n", **edits})
    bare_upstream(repo, tmp_path, u)
    git(repo, "reset", "-q", "--hard", "HEAD~2")
    git(repo, "update-ref", "refs/remotes/origin/main", o)
    (repo / "notes.txt").write_text("my edit\n", encoding="utf-8")  # dirty before the update: unchanged bytes pass
    token = pause(repo)
    if second_move == "torn":
        (repo / "locked").chmod(0o555)
    elif second_move == "refused":
        (repo / "new.py").write_text("in the way\n", encoding="utf-8")  # git refuses before writing a file
    update_cmd._pull_updates(
        ["git"], "main", None, prompt_for_restore=False, gw_input_fn=None, discard_local_changes=False,
        keep_stash=False, sync_upstream=True, assume_yes=True, _windows_gateway_resume=token)
    head = git(repo, "rev-parse", "HEAD")
    if second_move == "torn":
        assert head == o and (repo / "a.py").read_text() == "A=2\n", "fixture: not torn"
        assert gate(repo) is False, "the failed second move's torn tree was admitted at the first move's HEAD"
    else:
        assert head == (o if second_move == "refused" else u)
        (repo / "notes.txt").write_text("my later edit\n", encoding="utf-8")  # life goes on after the update
        assert gate(repo) is True, "a move that left the tree whole held the paused set"
