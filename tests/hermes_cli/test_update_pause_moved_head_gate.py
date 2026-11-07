"""The paused gateways' tree gate judges a MOVED HEAD too (#132338 R3 follow-up).

``git checkout <existing branch>`` can exit 0 and move HEAD while leaving a file unwritten
("error: unable to unlink old ..."): the index and HEAD name the new commit, the file keeps the old
bytes. The gate used to judge only an unmoved HEAD, so that torn tree was admitted. Now the commit a
recorded move reached names the expected tree: its paths are judged the moment git returns (before
a dependency sync or stash restore rewrites one), or at resume time when the updater died first.

Real git, real update step, real durable record and gate; git's failure is a read-only directory.
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
def parked(tmp_path, monkeypatch):
    """Parked on a merged ``feature`` at A; local and origin ``main`` at B, which rewrites locked/z.py."""
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "user.email", "fixture@example.com")
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", root)
    monkeypatch.setattr(pause_record, "install_root", lambda: root)
    a = commit(root, "A", **{"a.py": "A=1\n", "locked__z.py": "Z=1\n", "package-lock.json": "{}\n"})
    git(root, "branch", "-q", "feature")
    b = commit(root, "B", **{"a.py": "A=2\n", "locked__z.py": "Z=2\n", "package-lock.json": "{\"v\": 2}\n"})
    git(root, "update-ref", "refs/remotes/origin/main", b)
    git(root, "checkout", "-q", "feature")
    yield root, a, b
    (root / "locked").chmod(0o755)


def pause(root) -> dict:
    token = pause_record.stamp_tree({"resume_needed": True, "profiles": {"default": 999999}}, root)
    pause_record.write(token)
    return token


def gate(root) -> bool:
    """The verdict recovery reaches: the durable record, as the next launch reads it."""
    return pause_record.tree_is_whole(pause_record.read(pause_record.record_path())["token"], root)[0]


@pytest.mark.parametrize("unlink_fails", [True, False])
def test_a_switch_that_moved_head_past_an_unwritten_file_stays_held(parked, unlink_fails):
    root, _a, b = parked
    token = pause(root)
    if unlink_fails:
        (root / "locked").chmod(0o555)
    update_cmd._prepare_checkout_for_update(
        ["git"], "main", "feature", is_fork=False, assume_yes=True, gateway_mode=False,
        gw_input_fn=None, switch_branch=False, _windows_gateway_resume=token)
    assert git(root, "rev-parse", "HEAD") == b, "fixture: the switch did not move HEAD"
    assert (root / "a.py").read_text() == "A=2\n"
    # Later, npm rewrites a lockfile the move also changed: not the move's doing.
    (root / "package-lock.json").write_text("{\"v\": 2, \"churn\": 1}\n", encoding="utf-8")
    if unlink_fails:
        assert (root / "locked" / "z.py").read_text() == "Z=1\n", "fixture: git wrote the file"
        assert gate(root) is False, "a moved HEAD whose tree is not its commit's was admitted"
        (root / "locked").chmod(0o755)
        git(root, "checkout", "--", "locked/z.py")  # the user repairs it: the set may start
    assert gate(root) is True, "a switch that left the tree whole held the paused set"


def test_a_move_the_updater_died_before_judging_is_judged_at_resume(parked):
    root, _a, b = parked
    token = pause(root)
    pause_record.mark_move(token, b)  # recorded before git ran; the updater dies after git returns
    (root / "locked").chmod(0o555)
    git(root, "checkout", "-q", "main")  # exits 0 with "unable to unlink old 'locked/z.py'"
    assert git(root, "rev-parse", "HEAD") == b and (root / "locked" / "z.py").read_text() == "Z=1\n"
    assert gate(root) is False, "an unjudged move to a recorded target was admitted torn"
