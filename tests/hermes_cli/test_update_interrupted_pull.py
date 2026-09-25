"""A `hermes update` killed while git writes the new tree must leave a recoverable install.

Git rewrites the checkout file by file and moves HEAD last, so a kill in between leaves HEAD on the
old commit with some files already new — a mix that fails at import in every entry point. The
updater brackets the move with a marker; the next launch (``_early_recovery``, before any other
checkout import) puts the old tree back so ``hermes update`` can simply run again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import _early_recovery as er
from hermes_cli import update_cmd


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout.strip()


_MULTI = "top = 1\nx = 0\ny = 0\nz = 0\nend = 1\n"


# Runs an entry module with the repair replaced by a probe that lists the checkout modules imported so
# far (the entry module, its package's __init__ and what hermes_bootstrap needs excluded: those run
# before any code in the entry can), then stops.
_ENTRY_SPY = """
import importlib, json, os, sys
import hermes_bootstrap
from hermes_cli import _early_recovery as er

venv, entry = os.path.realpath(sys.prefix), sys.argv[1]
importlib.import_module(entry.rpartition(".")[0] or "hermes_cli")
before = set(sys.modules)

def probe():
    loaded = (n for n in set(sys.modules) - before if not f"{entry}.".startswith(n + "."))
    files = {n: os.path.realpath(str(getattr(sys.modules[n], "__file__", None))) for n in loaded}
    print(json.dumps(sorted(n for n, f in files.items() if f.startswith(os.getcwd()) and not f.startswith(venv))))
    raise SystemExit(0)

er.restore_interrupted_pull = probe
importlib.import_module(entry)
"""


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """An install at commit A whose fetched ``origin/main`` is B (modifies, deletes, adds, flips a mode)."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    files = {"utils.py": "OLD = 1\n", "other.py": "a = 1\n", "gone.py": "x = 1\n", "cut.py": "c = 1\n",
             "blank.py": "b = 1\n", "half.py": "h = 1\n", "tool.sh": "echo\n", "multi.py": _MULTI}
    for name, body in files.items():
        (origin / name).write_text(body, encoding="utf-8", newline="")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "A")
    for name, body in {"utils.py": "NEW = 1\n", "other.py": "a = 2\n", "cut.py": "c = 2\n", "multi.py": "top = 2\n" + _MULTI[8:],
                       "blank.py": "b = 2\n", "half.py": "h = 2  # long enough to span pages\n"}.items():
        (origin / name).write_text(body, encoding="utf-8", newline="")
    (origin / "gone.py").unlink()
    (origin / "newpkg").mkdir()
    (origin / "newpkg" / "__init__.py").write_text("from utils import NEW\n", encoding="utf-8", newline="")
    (origin / "newpkg" / "sub").mkdir()
    (origin / "newpkg" / "sub" / "mod.py").write_text("m = 1\n", encoding="utf-8", newline="")
    _git(origin, "add", "-A")
    _git(origin, "update-index", "--chmod=+x", "tool.sh")
    _git(origin, "commit", "-qm", "B")
    root = tmp_path / "install"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "reset", "-q", "--hard", "HEAD~1")
    monkeypatch.setattr("hermes_cli.main.PROJECT_ROOT", root)
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "origin/main")


def _pull(root: Path) -> None:
    update_cmd._pull_updates(["git"], "main", None, prompt_for_restore=False, gw_input_fn=None,
                             discard_local_changes=False, keep_stash=False)


def test_live_checkout_under_pytest_does_not_probe_recovery_marker(tmp_path, monkeypatch):
    """The test guard must run before any stat under the executing checkout's git directory."""
    root = tmp_path / "live-checkout"

    class ForbiddenMarker:
        def is_file(self):
            raise AssertionError("live checkout marker was probed")

    monkeypatch.setattr(er, "_project_root", lambda: root)
    monkeypatch.setattr(er, "interrupted_pull_marker", lambda _root: ForbiddenMarker())
    monkeypatch.setattr(er, "_pytest_owns_live_checkout", lambda candidate: candidate == root)

    assert er.restore_interrupted_pull() is False


def test_killed_pull_is_restored_on_next_launch_and_update_reruns(checkout, monkeypatch):
    root, a, b = checkout
    real = update_cmd._git_run

    def dying_git_run(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            # Git rewrites a file as unlink, create, write: the kill lands inside one of those.
            (root / "utils.py").write_text("NEW = 1\n", encoding="utf-8", newline="")
            (root / "cut.py").unlink()
            (root / "blank.py").write_bytes(b"")
            (root / "half.py").write_bytes(b"h = 2  # long")  # a multi-page write cut short
            (root / "newpkg").mkdir()
            (root / "newpkg" / "__init__.py").write_text("from utils import NEW\n", encoding="utf-8", newline="")
            (root / "newpkg" / "sub").mkdir()  # created for its next file, which the kill cut off
            (root / ".git" / "index.lock").touch()
            raise KeyboardInterrupt  # SIGKILL: nothing after this line of the updater runs
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", dying_git_run)
    with pytest.raises(KeyboardInterrupt):
        _pull(root)
    monkeypatch.setattr(update_cmd, "_git_run", real)
    assert _git(root, "rev-parse", "HEAD") == a  # the torn state: HEAD old, some files already new
    marker = er.interrupted_pull_marker(root)
    recorded = marker.read_text(encoding="utf-8")
    assert f"pid={os.getpid()}" in recorded and f"target={b}" in recorded  # the commit, not the ref name
    # The user re-applies their stash to a file the update also changes (git had not written it yet).
    (root / "other.py").write_text("a = 1  # my edit\n", encoding="utf-8", newline="")

    # Another `hermes` launched while an update is mid-pull must not race its git.
    updater = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        marker.write_text(recorded.replace(f"pid={os.getpid()}", f"pid={updater.pid}"), encoding="utf-8",
                          newline="")
        assert er.restore_interrupted_pull(root) is False
        assert marker.exists() and (root / ".git" / "index.lock").exists()
    finally:
        updater.kill()
        updater.wait()

    # A retry in a container gets the killed updater's pid: our own pid is never a live owner.
    marker.write_text(recorded, encoding="utf-8", newline="")
    if sys.platform != "win32":  # a git dir that cannot lock (NFS without lockd) still repairs, unguarded
        import errno
        import fcntl

        def no_locks(*_a):
            raise OSError(errno.ENOLCK, "No locks available")

        monkeypatch.setattr(fcntl, "flock", no_locks)
    assert er.restore_interrupted_pull(root) is True, "restored files mean the caller must relaunch"

    assert _git(root, "rev-parse", "HEAD") == a
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == "M other.py"
    assert (root / "other.py").read_text(encoding="utf-8") == "a = 1  # my edit\n", "the user's edit survives"
    assert not (root / "newpkg").exists() and not (root / ".git" / "index.lock").exists()
    assert not marker.exists()
    (root / "other.py").write_text("a = 1\n", encoding="utf-8", newline="")
    _pull(root)  # `hermes update` again: a normal fast-forward
    assert _git(root, "rev-parse", "HEAD") == b and not marker.exists()

    # Every console script (`hermes`, `hermes-agent`, `hermes-acp`) repairs before its entry module imports
    # any other checkout module past hermes_bootstrap: any of them may be a half-written file.
    repo = os.path.realpath(Path(er.__file__).parent.parent)
    for entry in ("hermes_cli.main", "agent.legacy_cli", "run_agent", "acp_adapter.entry"):
        run = subprocess.run([sys.executable, "-c", _ENTRY_SPY, entry], cwd=repo, capture_output=True, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONPATH": repo}, timeout=120)
        assert run.stdout.strip().splitlines()[-1:] == ["[]"], (entry, run.stdout[-500:], run.stderr[-2000:])


def test_restore_never_touches_user_work_when_git_wrote_nothing(checkout, capsys, monkeypatch):
    """sys.exit on a merge conflict is not a kill, and a marker git never acted on restores nothing."""
    root, a, b = checkout
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    _git(root, "checkout", "-q", "-b", "mywork")
    (root / "other.py").write_text("a = 'mine'\n", encoding="utf-8", newline="")
    _git(root, "commit", "-qam", "local work that conflicts upstream")
    with pytest.raises(SystemExit):
        _pull(root)
    marker = er.interrupted_pull_marker(root)
    assert not marker.exists()

    # Even a leftover marker (an older updater, or a kill mid-reconcile) stays out of the user's way:
    # following the printed advice leaves a merge in progress, and edits git never wrote are theirs.
    stale = f"pid=0\npre={_git(root, 'rev-parse', 'HEAD')}\ntarget={b}\nstash=\n"
    marker.write_text(stale, encoding="utf-8", newline="")
    merge = subprocess.run(["git", "-C", str(root), "merge", "origin/main"],
                           capture_output=True, text=True, encoding="utf-8")
    assert (root / ".git" / "MERGE_HEAD").exists(), merge.stdout + merge.stderr
    (root / "utils.py").write_text("OLD = 1  # resolved by hand\n", encoding="utf-8", newline="")
    before = _git(root, "status", "--porcelain", "--untracked-files=all")
    monkeypatch.setattr(er, "_merge_advice_shown", False, raising=False)
    capsys.readouterr()
    assert er.restore_interrupted_pull(root) is False and er.restore_interrupted_pull(root) is False
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == before
    # MERGE_HEAD is the update's own target: say how to get out of it, once per launch.
    assert capsys.readouterr().err.count(f"git -C {root} merge --abort") == 1
    _git(root, "reset", "-q", "--hard")  # the user gives up on the merge
    (root / "utils.py").write_text("OLD = 1  # my stash, re-applied\n", encoding="utf-8", newline="")
    # tool.sh only changes mode upstream and git never reached it: nothing to restore, no relaunch.
    assert er.restore_interrupted_pull(root) is False
    assert (root / "utils.py").read_text(encoding="utf-8") == "OLD = 1  # my stash, re-applied\n"
    assert not marker.exists(), "git wrote nothing: the marker is spent"
    # A target git no longer knows (gc, re-clone) can never be compared against: drop the marker.
    marker.write_text(stale.replace(b, "0" * 40), encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is False and not marker.exists()

    # Killed inside the custom-branch `git merge`: its files are the merge of both sides, not origin's
    # blob, and still git's (torn ones too), while the user's own edit survives.
    _git(root, "reset", "-q", "--hard", a)
    (root / "multi.py").write_text(_MULTI.replace("end = 1", "end = 'mine'"), encoding="utf-8", newline="")
    _git(root, "commit", "-qam", "local work that merges cleanly")
    pre = _git(root, "rev-parse", "HEAD")
    merged = _git(root, "merge-tree", "--write-tree", pre, b)
    merged_multi = _git(root, "show", f"{merged}:multi.py") + "\n"
    assert merged_multi == "top = 2\nx = 0\ny = 0\nz = 0\nend = 'mine'\n"  # neither side's blob
    (root / "multi.py").write_text(merged_multi, encoding="utf-8", newline="")
    (root / "utils.py").write_text("NEW = 1\n", encoding="utf-8", newline="")
    (root / "half.py").write_bytes(b"h = 2  # long")
    (root / "other.py").write_text("a = 1  # my edit\n", encoding="utf-8", newline="")
    marker.write_text(f"pid=0\npre={pre}\ntarget={b}\nstash=\n", encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists()
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == "M other.py"


_RACER = """
import sys
from pathlib import Path
from hermes_cli import _early_recovery as er
print("ready", flush=True)
sys.stdin.readline()
print(er.restore_interrupted_pull(Path(sys.argv[1])))
"""


def test_concurrent_launches_take_turns_and_all_rerun_from_the_restored_tree(tmp_path):
    """Launches racing on one torn checkout (a restarting gateway next to the user's CLI) restore once.

    None may break another's git (index.lock), print recovery advice while another is restoring or
    has finished, or carry on importing from a tree that changed under it.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.invalid")
    _git(origin, "config", "user.name", "t")
    names = [f"m{i}.py" for i in range(300)]  # enough work that the launches overlap
    for i, name in enumerate(names):
        (origin / name).write_text(f"V = 'old {i}'\n" * 50, encoding="utf-8", newline="")
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "A")
    for i, name in enumerate(names):
        (origin / name).write_text(f"V = 'new {i}'\n" * 50, encoding="utf-8", newline="")
    _git(origin, "commit", "-qam", "B")
    root = tmp_path / "install"
    _git(tmp_path, "clone", "-q", str(origin), str(root))
    _git(root, "reset", "-q", "--hard", "HEAD~1")
    pre, target = _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "origin/main")
    for i, name in enumerate(names[:150]):  # git got halfway
        (root / name).write_text(f"V = 'new {i}'\n" * 50, encoding="utf-8", newline="")
    (root / names[-1]).write_text("user edit\n", encoding="utf-8", newline="")
    marker = er.interrupted_pull_marker(root)
    marker.write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\n", encoding="utf-8", newline="")

    repo = os.path.realpath(Path(er.__file__).parent.parent)
    launches = [subprocess.Popen([sys.executable, "-c", _RACER, str(root)], cwd=repo, text=True, encoding="utf-8",
                                 env={**os.environ, "PYTHONPATH": repo}, stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(3)]
    for launch in launches:
        assert launch.stdout.readline().strip() == "ready"
    for launch in launches:  # release them together
        launch.stdin.write("go\n")
        launch.stdin.flush()
    results = [(launch.communicate(timeout=120), launch.returncode) for launch in launches]

    for (out, err), code in results:
        assert code == 0 and out.split()[-1:] == ["True"], (out, err)
        assert "Could not" not in err and "reset --hard" not in err, err
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == f"M {names[-1]}"
    assert (root / names[-1]).read_text(encoding="utf-8") == "user edit\n"
    assert not marker.exists()
