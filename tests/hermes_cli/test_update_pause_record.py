"""Paused-gateway record and update-claim adoption, with real processes and a real checkout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import update_pause_record as pause_record

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows cells live in wine2e")


def _child(code: str, *argv: str, env: dict | None = None) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *argv], cwd=REPO, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(REPO), **(env or {})})


def test_record_written_by_a_killed_updater_is_orphaned_only_after_its_death(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = _child("""
        import sys, time
        from hermes_cli import update_pause_record as r
        r.write(r.stamp_tree({"resume_needed": True, "profiles": {"default": 4242}}), owner=r.identity())
        print("written", flush=True)
        time.sleep(120)
    """, env={"HERMES_HOME": str(tmp_path)})
    try:
        assert owner.stdout.readline().strip() == "written"
        body = pause_record.read()
        assert body["owner"]["pid"] == owner.pid and body["owner"]["ct"].startswith("ct:")
        assert body["token"]["profiles"] == {"default": 4242}
        assert pause_record.orphaned_record() is None  # live owner: its own resume owns the set
    finally:
        owner.send_signal(signal.SIGKILL)  # windows-footgun: ok — module skips on Windows
        owner.wait(timeout=10)
    orphan = pause_record.orphaned_record()
    assert orphan is not None and orphan["token"]["profiles"] == {"default": 4242}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


def test_resume_waits_for_a_whole_tree(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    for name in ("a.py", "b.py", "local.txt"):
        (root / name).write_text("v1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "v1")
    (root / "local.txt").write_text("user edit\n", encoding="utf-8")  # dirty before the update: not git's doing
    token = pause_record.stamp_tree({"resume_needed": True}, root)
    assert pause_record.tree_is_whole(token, root) == (True, "")

    (root / "a.py").write_text("v2\n", encoding="utf-8")  # git wrote a.py, then died before b.py and HEAD
    whole, why = pause_record.tree_is_whole(token, root)
    assert not whole and "a.py" in why

    (root / "a.py").write_text("v1\n", encoding="utf-8")
    marker = root / ".git" / "hermes-update-pull"
    marker.write_text("pid\n", encoding="utf-8")
    assert pause_record.tree_is_whole(token, root)[0] is False
    marker.unlink()

    _git(root, "commit", "-qam", "v2")  # HEAD moved: only a dependency sync for it makes it whole
    whole, why = pause_record.tree_is_whole(token, root)
    assert not whole and "dependencies" in why


_HOLDER = """
    import subprocess, sys, time
    from hermes_cli.update_lock import UpdateLock
    lock = UpdateLock()
    assert lock.acquire() and lock.acquired
    host = subprocess.Popen([sys.executable, sys.argv[1], "hermes_cli.main", *sys.argv[2:]], stdout=subprocess.PIPE, text=True)
    print(host.stdout.read().strip(), flush=True)
    host.wait()
"""
# A stand-in for the host the update relaunched: its argv names the host command, and the
# ``hermes update`` its agent starts is its child.
_HOST = """
import subprocess, sys
probe = "from hermes_cli.update_lock import UpdateLock; l = UpdateLock(); print(l.acquire(), l.holder is not None)"
print(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip())
"""


# The intermediate is a stand-in script, not a real gateway/update: nothing touches the checkout.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("host_argv, adopts", [(("gateway", "run"), False), (("status",), True)])
def test_update_started_from_a_relaunched_gateway_does_not_share_the_claim(tmp_path, host_argv, adopts):
    script = tmp_path / "host.py"
    script.write_text(_HOST, encoding="utf-8")
    holder = _child(_HOLDER, str(script), *host_argv, env={"HERMES_HOME": str(tmp_path)})
    out, _ = holder.communicate(timeout=60)
    assert holder.returncode == 0, out
    assert out.strip() == ("True False" if adopts else "False True"), out
