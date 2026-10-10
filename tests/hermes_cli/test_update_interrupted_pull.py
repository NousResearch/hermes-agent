"""A `hermes update` killed while git writes the new tree must leave a recoverable install.

Git rewrites the checkout file by file and moves HEAD last, so a kill in between leaves HEAD on the
old commit with some files already new — a mix that fails at import in every entry point. The
updater brackets the move with a marker; the next launch (``_early_recovery``, before any other
checkout import) puts the old tree back so ``hermes update`` can simply run again.
"""

from __future__ import annotations

import os
import shutil
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


def test_restore_runs_the_installers_store_git_when_path_has_none(checkout, monkeypatch, tmp_path):
    """Windows installs whose only git is the copy install.ps1 staged in PM's store: the launch-time
    restore must find it like the updater does, release the dead git's index.lock and put the tree back
    (a bare ``git`` died with WinError 2 there and every update for 10 minutes refused on the lock)."""
    import shutil

    import pm
    import pm.paths

    root, a, b = checkout
    (root / "utils.py").write_text("NEW = 1\n", encoding="utf-8", newline="")  # the killed ff's first write
    (root / ".git" / "index.lock").write_bytes(b"")  # left by the killed git
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    er.interrupted_pull_marker(root).write_text(f"pid={dead.pid}\npre={a}\ntarget={b}\nstash=\n", encoding="utf-8")

    real_git = shutil.which("git")
    store = tmp_path / "tools"
    version = pm.Lockfile(pm.paths.lockfile_path()).version("git")
    staged = store / f"git-{version}-win32-x64" / "cmd" / "git.exe"
    staged.parent.mkdir(parents=True)
    calls = tmp_path / "staged-git-calls"
    staged.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\nexec "{real_git}" "$@"\n', encoding="utf-8")
    staged.chmod(0o755)
    with monkeypatch.context() as windows:
        windows.setattr(pm.paths, "store_root", lambda: store)
        windows.setattr(pm, "current_target", lambda: "win32-x64")
        windows.setenv("PATH", str(tmp_path / "no-git-here"))
        assert er.restore_interrupted_pull(root) is True
    assert calls.read_text(encoding="utf-8-sig").count("rev-parse HEAD") >= 1, "the store's git ran the restore"
    assert _git(root, "rev-parse", "HEAD") == a and _git(root, "status", "--porcelain") == ""
    assert not (root / ".git" / "index.lock").exists() and not er.interrupted_pull_marker(root).exists()
    _pull(root)  # the next `hermes update` is not refused on the lock
    assert _git(root, "rev-parse", "HEAD") == b

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
    # A target git no longer knows (gc, re-clone) can never be compared against: the marker goes
    # only over a clean tracked tree, since the dirty bytes may be the update's (review G2).
    marker.write_text(stale.replace(b, "0" * 40), encoding="utf-8", newline="")
    capsys.readouterr()
    assert er.restore_interrupted_pull(root) is False and marker.exists()
    assert "the marker was kept" in capsys.readouterr().err
    assert (root / "utils.py").read_text(encoding="utf-8") == "OLD = 1  # my stash, re-applied\n"
    _git(root, "stash", "-q")
    assert er.restore_interrupted_pull(root) is False and not marker.exists()
    _git(root, "stash", "pop", "-q")

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


def _broken_release(tmp_path: Path, nfiles: int) -> tuple[Path, str, str]:
    """A checkout whose HEAD is a release with an uncompilable module (``target``) over ``pre``."""
    root = tmp_path / "install"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "module.py").write_text("good = True\n", encoding="utf-8", newline="")
    (root / "bulk").mkdir()
    for i in range(nfiles):  # enough index work that git holds index.lock long enough to be killed
        (root / "bulk" / f"f{i}.txt").write_text(f"{i}\n", encoding="utf-8", newline="")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "pre")
    pre = _git(root, "rev-parse", "HEAD")
    (root / "module.py").write_text("def broken(:\n", encoding="utf-8", newline="")
    (root / "added.py").write_text("X = 1\n", encoding="utf-8", newline="")
    for i in range(0, nfiles, 2):
        (root / "bulk" / f"f{i}.txt").write_text(f"{i} new\n", encoding="utf-8", newline="")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "broken target")
    return root, pre, _git(root, "rev-parse", "HEAD")


def _rollback_marker(root: Path, pre: str, target: str) -> Path:
    marker = er.interrupted_pull_marker(root)
    marker.write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\nrollback=branch\nref=refs/heads/main\n",
                      encoding="utf-8", newline="")
    return marker


def _sigkill_git_once_index_lock_exists(root: Path, *args: str) -> int:
    """Run a real git and SIGKILL it the moment inotify reports ``.git/index.lock`` created."""
    import ctypes
    import signal
    import struct

    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(0)
    assert fd >= 0 and libc.inotify_add_watch(fd, str(root / ".git").encode(), 0x100) >= 0  # IN_CREATE
    git = subprocess.Popen(["git", "-C", str(root), *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        while git.poll() is None:
            buf, i = os.read(fd, 4096), 0
            while i < len(buf):
                length = struct.unpack_from("iIII", buf, i)[3]
                name, i = buf[i + 16:i + 16 + length].rstrip(b"\0"), i + 16 + length
                if name == b"index.lock":
                    os.kill(git.pid, signal.SIGKILL)
                    return git.wait()
        return git.wait()
    finally:
        os.close(fd)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="inotify kill cell")
def test_a_rollback_killed_inside_its_reset_is_finished_by_the_next_launch(tmp_path):
    """The syntax rollback's own ``git reset -q pre`` is SIGKILLed holding ``index.lock``: HEAD is still the
    broken release. The next launch must reclaim that dead lock, redo the rollback and land on ``pre``
    whole; the marker (the rollback's only record) may go only then."""
    for attempt in range(10):
        cell = tmp_path / f"try{attempt}"
        cell.mkdir()
        root, pre, target = _broken_release(cell, 3000)
        marker = _rollback_marker(root, pre, target)
        _sigkill_git_once_index_lock_exists(root, "reset", "-q", pre)
        if (root / ".git" / "index.lock").exists() and _git(root, "rev-parse", "HEAD") == target:
            break
    else:
        pytest.fail("harness: the SIGKILL never landed while git held index.lock")

    assert er.restore_interrupted_pull(root) is True, "the broken files were put back: the caller relaunches"
    assert _git(root, "rev-parse", "HEAD") == pre
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    assert (root / "module.py").read_text(encoding="utf-8") == "good = True\n"
    assert not (root / ".git" / "index.lock").exists() and not marker.exists()


def test_a_rollback_marker_outlives_a_lock_that_may_still_be_live(tmp_path):
    """While a live process holds ``index.lock`` the rollback cannot resume: HEAD stays on the broken
    release, so the marker must stay too (never 'git finished'), and the lock is not stolen. Once the
    holder is gone the next launch finishes the rollback."""
    root, pre, target = _broken_release(tmp_path, 20)
    marker = _rollback_marker(root, pre, target)
    lock = root / ".git" / "index.lock"
    holder = subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'w'); "
                               "print('held', flush=True); time.sleep(120)", str(lock)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert er.restore_interrupted_pull(root) is False
        assert marker.exists(), "a rollback that could not run erased its own recovery record"
        assert lock.exists() and _git(root, "rev-parse", "HEAD") == target
    finally:
        holder.kill()
        holder.wait()
    if sys.platform == "darwin" and not shutil.which("lsof"):
        pytest.skip("no lsof: the dead lock cannot be proven dead here")
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists() and not lock.exists()
    assert _git(root, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_rollback_settles_beside_an_unrelated_tracked_edit_and_keeps_it(tmp_path, capsys):
    """HEAD and the index are back on ``pre`` but the broken files are still on disk, and the user has
    since edited a file the update never touched: the rollback finishes on its own paths, the edit stays."""
    root, pre, target = _broken_release(tmp_path, 4)
    marker = _rollback_marker(root, pre, target)
    _git(root, "reset", "-q", pre)  # the rollback's reset landed; its file restore was killed
    (root / "bulk" / "f1.txt").write_text("my edit\n", encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is True
    assert "reset --hard" not in capsys.readouterr().err
    assert not marker.exists() and (root / "module.py").read_text(encoding="utf-8-sig") == "good = True\n"
    assert _git(root, "status", "--porcelain", "--untracked-files=no") == "M bulk/f1.txt"


def test_an_index_lock_judged_foreign_after_our_git_exited_is_never_reclaimed_later(tmp_path):
    """Our git exited mid-move and an index.lock was there (another git's, e.g. `git commit` in the editor
    with its fd closed): no later launch may delete THAT lock; a new lock generation is judged afresh."""
    root, pre, target = _broken_release(tmp_path, 2)
    _git(root, "reset", "-q", "--hard", pre)
    (root / "module.py").write_text("def broken(:\n", encoding="utf-8", newline="")  # git's write landed
    marker = er.interrupted_pull_marker(root)
    marker.write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\n", encoding="utf-8", newline="")
    lock = root / ".git" / "index.lock"
    lock.write_bytes(b"")
    assert er.restore_interrupted_pull(root, after_failure=True) is False
    assert er.restore_interrupted_pull(root) is False
    assert lock.exists() and marker.exists(), "a lock judged foreign was deleted on the next launch"
    if sys.platform == "darwin" and not shutil.which("lsof"):
        pytest.skip("no lsof: a fresh dead lock cannot be proven dead here")
    lock.unlink()
    lock.write_bytes(b"")  # a later generation (our own killed git's): the dead-lock proof applies
    os.utime(lock, ns=(1_000_000_000, 1_000_000_000))
    assert er.restore_interrupted_pull(root) is True
    assert not lock.exists() and not marker.exists()
    assert (root / "module.py").read_text(encoding="utf-8-sig") == "good = True\n"


def test_a_launch_that_cannot_get_the_repair_claim_never_continues_from_the_torn_tree(tmp_path, monkeypatch):
    """Another launch holds the restore claim past the wait while the marker says the tree is torn:
    this launch must stop (fail closed) instead of reporting 'nothing to repair' and importing it."""
    root, pre, target = _broken_release(tmp_path, 5)
    marker = _rollback_marker(root, pre, target)
    monkeypatch.setattr(er, "_RESTORE_CLAIM_WAIT_SECONDS", 0.2)
    fd = os.open(marker.parent / er._RESTORE_CLAIM, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        assert er._lock_fd(fd, True)
        with pytest.raises(SystemExit, match="launch again"):
            er.restore_interrupted_pull(root)
        # A real launch sees one line and exit 1, never a traceback (review C14).
        launch = subprocess.run(
            [sys.executable, "-c", "import sys; from pathlib import Path; from hermes_cli import _early_recovery as er; "
             "er._RESTORE_CLAIM_WAIT_SECONDS = 0.2; er.restore_interrupted_pull(Path(sys.argv[1]))", str(root)],
            cwd=Path(er.__file__).resolve().parent.parent, capture_output=True, text=True, encoding="utf-8",
            timeout=60)
    finally:
        er._lock_fd(fd, False)
        os.close(fd)
    assert launch.returncode == 1 and "Traceback" not in launch.stderr
    assert launch.stderr.strip().count("\n") == 0 and "launch again" in launch.stderr
    assert marker.exists() and _git(root, "rev-parse", "HEAD") == target
    assert er.restore_interrupted_pull(root) is True  # once the claim is free the repair runs
    assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists()


# --- R2: the launch-time repair holds the checkout kernel lock -------------------------------

_HOLD_CHECKOUT = """
import sys, time
from pathlib import Path
from hermes_cli import update_lock
assert update_lock._acquire_checkout(Path(sys.argv[1])) is None
print("held", flush=True)
time.sleep(120)
"""


def test_launch_repair_leaves_the_tree_to_a_live_update_holding_the_checkout(tmp_path, capsys):
    """A live update tree (here: a process holding the checkout kernel lock, as a running
    completion/build/git does after its updater died) owns the checkout: the launch-time repair
    must not touch it, and must finish the job once the tree is gone."""
    root, pre, target = _broken_release(tmp_path, 20)
    marker = _rollback_marker(root, pre, target)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(er.__file__).resolve().parents[1]),
                                                       os.environ.get("PYTHONPATH", "")]))
    holder = subprocess.Popen([sys.executable, "-c", _HOLD_CHECKOUT, str(root)], stdout=subprocess.PIPE,
                              text=True, env=env)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert er.restore_interrupted_pull(root) is False
        assert "Not repairing the checkout now" in capsys.readouterr().err
        assert marker.exists() and _git(root, "rev-parse", "HEAD") == target, "the repair raced a live update"
    finally:
        holder.kill()
        holder.wait()
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists()


def test_a_dead_index_lock_is_released_even_when_git_cannot_run(tmp_path, monkeypatch):
    """The dead ``index.lock`` goes before any git runs: an unresolvable git must not strand it."""
    root, pre, target = _broken_release(tmp_path, 5)
    _rollback_marker(root, pre, target)
    lock = root / ".git" / "index.lock"
    lock.write_bytes(b"")
    if sys.platform == "darwin" and not shutil.which("lsof"):
        pytest.skip("no lsof: the dead lock cannot be proven dead here")
    monkeypatch.setattr(er, "_git_executable", lambda recorded="": str(tmp_path / "no-such-git"))
    assert er.restore_interrupted_pull(root) is False
    assert not lock.exists(), "a missing git stranded a dead index.lock"


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="Linux /proc holder scan")
def test_a_reader_git_in_the_tree_is_not_an_index_lock_holder_and_a_holder_is_named(tmp_path, capsys):
    """A reader git whose cwd is the checkout (a paged ``git log``, ``cat-file --batch``) never takes
    ``index.lock``: it must not keep a dead lock forever. A process with the lock open is named (pid + name) in the message."""
    root, pre, target = _broken_release(tmp_path, 5)
    marker = _rollback_marker(root, pre, target)
    lock = root / ".git" / "index.lock"
    lock.write_bytes(b"")
    # A long-lived reader git in the tree (what IDEs and gitstatusd keep): `git cat-file --batch`
    # waiting on stdin, the same shape as a `git log` parked in its pager.
    pager = subprocess.Popen(["git", "cat-file", "--batch"], cwd=root, stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert er.restore_interrupted_pull(root) is True, "a reader git kept a dead index.lock"
        assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists() and not lock.exists()
    finally:
        pager.kill()
        pager.wait()

    (tmp_path / "named").mkdir()
    root2, pre2, target2 = _broken_release(tmp_path / "named", 5)
    _rollback_marker(root2, pre2, target2)
    lock2 = root2 / ".git" / "index.lock"
    holder = subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'w'); "
                               "print('held', flush=True); time.sleep(120)", str(lock2)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert er.restore_interrupted_pull(root2) is False
        assert f"pid {holder.pid} (" in capsys.readouterr().err
    finally:
        holder.kill()
        holder.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell stubs stand in for a decayed git")
@pytest.mark.parametrize("stub", ["#!/bin/sh\nexit 0\n", "not a program\n"], ids=["silent-exit-0", "not-executable"])
def test_a_recorded_git_that_is_no_longer_git_falls_back_and_still_repairs(tmp_path, stub):
    """m5: the marker's recorded ``git=`` was trusted on ``os.path.isfile`` alone. A stub that exits 0
    silently made ``rev-parse HEAD`` print nothing, which read as "HEAD moved": the marker was
    deleted over the torn tree. A non-executable file bricked every launch. The recorded git must
    answer ``--version`` as git, else the resolver's git repairs."""
    root, pre, target = _broken_release(tmp_path, 5)
    stub_path = tmp_path / "decayed-git"
    stub_path.write_text(stub, encoding="utf-8", newline="")
    if stub.startswith("#!"):
        stub_path.chmod(0o755)
    marker = er.interrupted_pull_marker(root)
    marker.write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\nrollback=branch\nref=refs/heads/main\ngit={stub_path}\n",
                      encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not marker.exists()
    assert er._git_executable(str(stub_path)) != str(stub_path)


def test_an_empty_head_answer_keeps_the_marker(tmp_path, monkeypatch, capsys):
    """m5: whatever git answers, only a full object name counts as HEAD; an empty one keeps the
    marker for the next launch instead of reading as "the update finished"."""
    root, pre, target = _broken_release(tmp_path, 5)
    marker = _rollback_marker(root, pre, target)
    stub = tmp_path / "silent-git"
    stub.write_text("", encoding="utf-8")
    monkeypatch.setattr(er, "_git_executable", lambda recorded="": str(stub))
    real_run = subprocess.run

    def silent(argv, *args, **kwargs):  # every git call "succeeds" with no output
        if argv and argv[0] == str(stub):
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", silent)
    assert er.restore_interrupted_pull(root) is False
    assert marker.exists(), "an empty `rev-parse HEAD` deleted the marker over a torn tree"
    assert "Could not read HEAD" in capsys.readouterr().err


def test_a_refusal_whose_holder_exits_before_the_probe_retries_the_lock(tmp_path, monkeypatch):
    """m10: the acquire was refused, then the follow-up probe found the lock free (its holder had
    just exited) and the repair ran unguarded. It takes the lock in that case."""
    from hermes_cli import update_lock

    root, _pre, _target = _broken_release(tmp_path, 1)
    real = update_lock._acquire_checkout
    calls = []

    def refused_once(install_root):
        calls.append(install_root)
        if len(calls) == 1:  # the refusal; by the probe its holder is gone
            return update_lock.UpdateHolder(pid=2 ** 22 + 7, age_seconds=0.0)
        return real(install_root)

    monkeypatch.setattr(update_lock, "_acquire_checkout", refused_once)
    with er._checkout_custody(root) as busy:
        assert busy == ""
        held = update_lock._HELD
        assert held is not None and held["path"] == str(update_lock.checkout_lock_path(root)), \
            "the repair ran without the checkout lock"
    assert update_lock._HELD is None and len(calls) == 2


@pytest.fixture
def commit_point():
    from hermes_cli import update_cmd_commit as commit

    commit.begin_update_attempt()
    yield commit
    commit.begin_update_attempt()


def _unmovable(root: Path, a: str) -> None:
    assert _git(root, "rev-parse", "HEAD") == a
    assert _git(root, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_pull_whose_marker_cannot_be_written_never_moves_the_tree(checkout, commit_point):
    """No recovery marker, no move: a kill inside an unmarked merge leaves a torn tree no launch
    can identify (F23, F17)."""
    root, a, _b = checkout
    er.interrupted_pull_marker(root).mkdir()  # EISDIR stands in for ENOSPC/EROFS/a sharing violation

    with pytest.raises(SystemExit):
        _pull(root)

    _unmovable(root, a)
    from hermes_cli.update_host_obligation import read_host_obligation
    from hermes_cli.venv_sync import completion_pending_path

    assert read_host_obligation() is None and not completion_pending_path(root).exists(), "the refused move left a debt"


def test_a_pull_whose_target_does_not_resolve_never_moves_the_tree(checkout, commit_point, monkeypatch):
    """The marker must name the commit git moves to; an unresolved target is refused (F17)."""
    root, a, _b = checkout
    real = update_cmd._git_run

    def no_rev_parse(git_cmd, args, *rest, **kw):
        if args[:1] == ["rev-parse"] and any("origin/main" in arg for arg in args):
            return subprocess.CompletedProcess(args, 128, "", "fatal: ambiguous argument")
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", no_rev_parse)
    with pytest.raises(SystemExit):
        _pull(root)

    _unmovable(root, a)


def test_an_arm_failure_after_the_autostash_still_names_the_stash(checkout, commit_point, monkeypatch, capsys):
    """An unwritable install state refuses the pull; the user is still told where their work is (F29)."""
    _root, _a, _b = checkout

    def unwritable(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(commit_point, "arm_commit_obligations", unwritable)
    with pytest.raises((SystemExit, OSError)):
        update_cmd._pull_updates(["git"], "main", "stash@{0}", prompt_for_restore=False, gw_input_fn=None,
                                 discard_local_changes=False, keep_stash=False)

    assert "stash@{0}" in capsys.readouterr().out


def test_a_branch_switch_whose_marker_cannot_be_written_never_starts(checkout, commit_point):
    """CP0 is a tree move like the pull: no marker, no checkout (F22)."""
    root, a, _b = checkout
    _git(root, "checkout", "-q", "-b", "feat")
    er.interrupted_pull_marker(root).mkdir()

    result = update_cmd._switch_branch_at_commit_point(["git"], "main", "origin/main", pre=a, stash=None)

    assert result.returncode != 0
    _unmovable(root, a)
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "feat"


def test_a_branch_switch_owes_the_restart_for_the_commit_it_lands_on(checkout, commit_point):
    """Local main (A) trails origin/main (B): CP0 lands on A, so a stop before CP1 must leave debt
    a checkout at A can discharge, never B's (N16)."""
    from hermes_cli.update_host_obligation import read_host_obligation

    root, a, b = checkout
    _git(root, "checkout", "-q", "-b", "feat")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", "parked")
    parked = _git(root, "rev-parse", "HEAD")

    result = update_cmd._switch_branch_at_commit_point(["git"], "main", "origin/main", pre=parked, stash=None)

    assert result.returncode == 0 and _git(root, "rev-parse", "HEAD") == a
    assert (read_host_obligation() or {}).get("expected_sha") == a != b


def test_an_unmarked_rollback_that_fails_never_reports_the_install_unchanged(checkout, commit_point, monkeypatch, capsys):
    """The syntax rollback cannot refuse (the tree is already broken), but without a marker only a
    tree verified whole at pre may be reported restored (F25)."""
    root, a, b = checkout
    _git(root, "reset", "-q", "--hard", b)
    er.interrupted_pull_marker(root).mkdir()
    monkeypatch.setattr(update_cmd, "_validate_critical_files_syntax", lambda _root: (False, "utils.py", "SyntaxError"))
    real = update_cmd._git_run

    def hard_reset_fails(git_cmd, args, *rest, **kw):
        if args[:2] == ["reset", "--hard"]:
            return subprocess.CompletedProcess(args, 1, "", "error: unable to unlink utils.py")
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd, "_git_run", hard_reset_fails)
    with pytest.raises(SystemExit):
        update_cmd._rollback_if_pulled_syntax_error(["git"], a)

    out = capsys.readouterr().out
    assert "Rollback complete" not in out and "Recover manually" in out


def test_a_rollback_never_rewinds_a_branch_checked_out_at_its_target_since(tmp_path, commit_point):
    """The user ran `git checkout -b feature` on the broken release before relaunching: the redo must
    not reset *feature*, and main must not be left behind with the marker gone (F18)."""
    root, pre, target = _broken_release(tmp_path, 3)
    commit_point.arm_tree_move(["git"], root, pre=pre, target=target, stash=None, rollback="branch")
    _git(root, "checkout", "-q", "-b", "feature")

    er.restore_interrupted_pull(root)

    assert _git(root, "rev-parse", "feature") == target
    assert er.interrupted_pull_marker(root).exists()

def _killed_move(tmp_path: Path, change, base=lambda root: None) -> tuple[Path, str, str]:
    """A checkout back on ``pre`` (a module + ``base``) with a dead updater's marker for ``target`` = ``change(pre)``."""
    root = tmp_path / "install"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.invalid")
    _git(root, "config", "user.name", "t")
    (root / "core.py").write_text("OLD = 1\n", encoding="utf-8", newline="")
    base(root)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "pre")
    pre = _git(root, "rev-parse", "HEAD")
    change(root)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "target")
    target = _git(root, "rev-parse", "HEAD")
    _git(root, "reset", "-q", "--hard", pre)
    er.interrupted_pull_marker(root).write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\n", encoding="utf-8")
    return root, pre, target


def test_a_users_file_at_a_path_the_update_adds_is_kept_aside_never_deleted(tmp_path):
    """The update adds files; after the kill the user creates their own at two of those paths (a first
    line that starts git's blob, a ``touch``). Git's whole file goes; theirs leave the tree, bytes intact."""
    def adds(root: Path) -> None:
        for name in ("full.py", "started.py", "touched.py"):
            (root / name).write_text(f"{name} = 'new content'\n", encoding="utf-8", newline="")

    root, pre, _target = _killed_move(tmp_path, adds)
    (root / "full.py").write_text("full.py = 'new content'\n", encoding="utf-8", newline="")  # git's own write
    (root / "started.py").write_bytes(b"started")
    (root / "touched.py").write_bytes(b"")
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not er.interrupted_pull_marker(root).exists()
    assert not (root / "full.py").exists()
    assert (root / "started.py.hermes-update-kept").read_bytes() == b"started"
    assert (root / "touched.py.hermes-update-kept").read_bytes() == b""
    assert not (root / "started.py").exists() and not (root / "touched.py").exists()


def _regular(root: Path) -> None:
    (root / "L").unlink(missing_ok=True)
    (root / "L").write_text("L = 'regular'\n", encoding="utf-8", newline="")


def _link(root: Path) -> None:
    (root / "L").unlink(missing_ok=True)
    os.symlink("core.py", root / "L")


def _what_is_at(path: Path):
    """A link's target, a file's bytes, or None: never following the link."""
    if os.path.islink(path):
        return ("link", os.readlink(path))
    return ("file", path.read_bytes()) if path.exists() else None


@pytest.mark.platforms("posix")  # creating symlinks needs Developer Mode/admin on Windows
@pytest.mark.parametrize(("base", "change"), [
    (_regular, _link), (lambda root: None, _link), (_link, lambda root: (root / "L").unlink()), (_link, _regular),
], ids=["regular-to-symlink", "added-symlink", "deleted-symlink", "symlink-to-regular"])
def test_a_move_killed_after_git_wrote_a_symlink_change_is_restored(tmp_path, base, change):
    """Symlinks are blobs git moves like files: a kill after git wrote the link change (added, deleted,
    or a type change either way) is put back to ``pre``, never left with the marker spent."""
    root, pre, _target = _killed_move(tmp_path, change, base)
    before = _what_is_at(root / "L")
    change(root)  # git wrote this path, then the kill came before HEAD moved
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not er.interrupted_pull_marker(root).exists()
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    assert _what_is_at(root / "L") == before


def _regular_naming_core(root: Path) -> None:
    """A regular file holding ``core.py``: the same blob as a symlink to ``core.py``."""
    (root / "L").unlink(missing_ok=True)
    (root / "L").write_bytes(b"core.py")


@pytest.mark.platforms("posix")  # creating symlinks needs Developer Mode/admin on Windows
@pytest.mark.parametrize("kill", ["tree-written", "tree-and-index-written", "core-symlinks-false"])
@pytest.mark.parametrize(("base", "change"), [(_regular_naming_core, _link), (_link, _regular_naming_core)],
                         ids=["regular-to-symlink", "symlink-to-regular"])
def test_an_equal_blob_type_change_git_wrote_is_restored(tmp_path, base, change, kill):
    """A regular file holding ``core.py`` and a symlink to ``core.py`` are one blob id: only the entry's
    mode tells that git wrote it. Killed with the tree (and index) written but HEAD still ``pre``, the
    move is put back, never left as a type change with its marker spent (review N07). Under
    ``core.symlinks=false`` git checks the link out as that very file: only the index shows the change."""
    root, pre, target = _killed_move(tmp_path, change, base)
    if kill == "core-symlinks-false":
        _git(root, "config", "core.symlinks", "false")
        (root / "L").unlink()
        _git(root, "checkout", "--", "L")  # the link as git checks it out here: a plain file
    before = _what_is_at(root / "L")
    if kill == "tree-written":
        change(root)
    else:  # git's index/tree update, before its HEAD update
        _git(root, "read-tree", "-u", "-m", pre, target)
        assert _git(root, "status", "--porcelain") == "T  L"
    assert er.restore_interrupted_pull(root) is True
    assert _git(root, "rev-parse", "HEAD") == pre and not er.interrupted_pull_marker(root).exists()
    assert _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    assert _what_is_at(root / "L") == before


def test_the_tree_move_marker_is_durable_before_it_appears_under_its_name(checkout, commit_point, monkeypatch):
    """The marker is the restore's only record: it is written to a temp file, fsynced, then renamed
    over its name. An in-place write could leave a power cut with an empty or half marker over a
    torn tree, which no launch can identify (review C3)."""
    from hermes_cli import update_cmd_commit

    root, _a, b = checkout
    marker = er.interrupted_pull_marker(root)
    marker.write_text("pid=1\npre=older\n", encoding="utf-8")
    synced, replaced = [], []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        synced.append(fd)
        return real_fsync(fd)

    def replace(src, dst):
        assert synced, "renamed into place before its bytes were fsynced"
        assert marker.read_text(encoding="utf-8") == "pid=1\npre=older\n"  # never rewritten in place
        replaced.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    update_cmd_commit.arm_tree_move(["git"], root, pre=None, target=b, stash=None)

    assert [dst for _src, dst in replaced] == [marker]
    assert f"target={b}" in marker.read_text(encoding="utf-8")
    assert not list(marker.parent.glob(marker.name + "*.tmp"))


def test_a_rollback_whose_branch_was_switched_away_never_advises_a_reset(tmp_path, capsys):
    """The user checked out another branch after the rollback was killed: the restore must not
    resume, and its advice must never be `git reset --hard` (F73: it wipes the user's work); it
    names the way back so the next launch can finish (review C9)."""
    root, pre, target = _broken_release(tmp_path, 2)
    marker = _rollback_marker(root, pre, target)
    _git(root, "checkout", "-q", "-b", "elsewhere")

    assert er.restore_interrupted_pull(root) is False

    err = capsys.readouterr().err
    assert "was not resumed" in err and "reset --hard" not in err
    assert "checkout main" in err and marker.exists()


def test_a_lock_our_killed_rollback_git_left_is_reclaimed_by_the_next_launch(tmp_path):
    """C15: the syntax rollback's own `reset -q` is SIGKILLed holding index.lock and the updater's
    in-process settle (after_failure) recorded THAT lock as foreign, so every later launch printed
    "cannot resume yet" forever. Only the lock generation that predates the move is foreign."""
    if sys.platform == "darwin" and not shutil.which("lsof"):
        pytest.skip("no lsof: a fresh dead lock cannot be proven dead here")
    from hermes_cli import update_cmd_commit

    root, pre, target = _broken_release(tmp_path, 2)
    marker = update_cmd_commit.arm_tree_move(["git"], root, pre=pre, target=target, stash=None,
                                             rollback="branch")
    lock = root / ".git" / "index.lock"
    lock.write_bytes(b"")  # our killed git's: it appeared after the marker was armed
    os.utime(lock, ns=(1_000_000_000, 1_000_000_000))

    assert er.restore_interrupted_pull(root, after_failure=True) is False  # the updater: never judges it
    assert "foreign_lock=" not in marker.read_text(encoding="utf-8")
    assert er.restore_interrupted_pull(root) is True  # the next launch: proven dead, reclaimed
    assert _git(root, "rev-parse", "HEAD") == pre and not lock.exists() and not marker.exists()


def test_a_lock_that_predates_the_move_stays_foreign_after_our_git_exited(tmp_path):
    """F05/F06 unchanged: the lock generation already there when the move was armed is another
    git's (e.g. `git commit` waiting in the editor, fd closed); no later launch deletes it."""
    from hermes_cli import update_cmd_commit

    root, pre, target = _broken_release(tmp_path, 2)
    lock = root / ".git" / "index.lock"
    lock.write_bytes(b"")
    marker = update_cmd_commit.arm_tree_move(["git"], root, pre=pre, target=target, stash=None,
                                             rollback="branch")

    assert er.restore_interrupted_pull(root, after_failure=True) is False
    assert er.restore_interrupted_pull(root) is False
    assert lock.exists() and marker.exists(), "a lock that predates the move was deleted"
    assert _git(root, "rev-parse", "HEAD") == target


@pytest.mark.parametrize("torn", [True, False])
def test_a_gone_target_retires_its_marker_only_over_a_clean_pre_tree(tmp_path, torn):
    """A killed move left HEAD on ``pre`` with the target's bytes in a tracked file, then git pruned
    the target commit: recovery cannot attribute those bytes, and it dropped the only record over
    the dirty tree (review G2). The marker now goes only when the tracked tree is clean at pre."""
    root, pre, target = _broken_release(tmp_path, 2)
    _git(root, "reset", "-q", "--hard", pre)
    _git(root, "reflog", "expire", "--expire=now", "--all")
    _git(root, "gc", "-q", "--prune=now")
    assert subprocess.run(["git", "-C", str(root), "cat-file", "-e", target]).returncode != 0  # gone
    if torn:
        (root / "module.py").write_text("def broken(:\n", encoding="utf-8", newline="")
    marker = er.interrupted_pull_marker(root)
    marker.write_text(f"pid=0\npre={pre}\ntarget={target}\nstash=\n", encoding="utf-8", newline="")
    assert er.restore_interrupted_pull(root) is False
    assert marker.exists() is torn
    if torn:
        assert (root / "module.py").read_text(encoding="utf-8") == "def broken(:\n"  # never guessed back
