"""A killed tree move that tears the launch repair's own code is still repaired at the next launch.

The repair (``hermes_bootstrap`` -> ``_early_recovery`` under the checkout lock) lives in the tree git
rewrites. ``arm_tree_move`` publishes that closure beside the marker before git writes; the minted
launcher runs it (or the same files read from git's objects) when the checkout's copy cannot import. Real git, really killed mid-write (a
smudge-filter barrier), the production marker writer in a child that exits, the production launcher;
the application entry is an inert receipt.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import zlib
from pathlib import Path

import pytest

from hermes_cli import _launchers

SOURCE = Path(__file__).resolve().parents[2]
# What a minted launcher imports before the application entry.
BOOT = ("hermes_bootstrap.py", "hermes_constants.py", "hermes_cli/__init__.py", "hermes_cli/_launchers.py",
        "hermes_cli/_early_recovery.py", "hermes_cli/_parser.py", "hermes_cli/runtime_state.py",
        "hermes_cli/venv_sync.py", "hermes_cli/steward.py", "hermes_cli/stderr_timestamp.py",
        "hermes_cli/update_lock.py", "hermes_cli/update_custody.py", "pm/environments.py",
        "pm/filesystem.py", "pm/paths.py")
_HOLDER = ("import sys, time\nfrom pathlib import Path\nsys.path.insert(0, sys.argv[1])\n"
           "from hermes_cli.update_lock import UpdateLock\n"
           "lock = UpdateLock(path=Path(sys.argv[2]) / 'owner', install_root=Path(sys.argv[2]))\n"
           "assert lock.acquire()\nprint('HELD', flush=True)\ntime.sleep(120)\n")


def _killed_mid_write(tmp_path: Path, rel: str, committed: dict[str, str] | None = None):
    """A checkout whose update git was SIGKILLed after unlinking ``rel``, before writing it."""
    root, home = tmp_path / "checkout", tmp_path / "home"
    home.mkdir()
    for f in BOOT:
        if (SOURCE / f).is_file():
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / f, root / f)
    for f, text in (committed or {}).items():
        (root / f).write_text(text, encoding="utf-8")
    (root / "hermes_cli/main.py").write_text("def main():\n    print('APP_REACHED')\n    return 0\n", encoding="utf-8")
    env = {"HOME": str(home), "HERMES_HOME": str(home / ".hermes"), "PATH": os.environ.get("PATH", ""),
           "LANG": "C.UTF-8", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    git_exe = shutil.which("git") or "git"

    def git(*args):
        return subprocess.check_output([git_exe, "-C", str(root), *args], env=env, text=True, encoding="utf-8").strip()

    git("init", "-q", "-b", "main")
    git("config", "user.name", "t")
    git("config", "user.email", "t@example.invalid")
    git("add", "-A")
    git("commit", "-qm", "before")
    pre = git("rev-parse", "HEAD")
    original = (root / rel).read_bytes()
    (root / rel).write_bytes(original + b"\nNEXT_VERSION = True\n")
    git("commit", "-qam", "after")
    target = git("rev-parse", "HEAD")
    git("reset", "-q", "--hard", pre)
    armed = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); from pathlib import Path; "
         "from hermes_cli.update_cmd_commit import arm_tree_move; "
         "arm_tree_move([sys.argv[2]], Path(sys.argv[3]), pre=sys.argv[4], target=sys.argv[5], stash=None)",
         str(SOURCE), git_exe, str(root), pre, target], env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert armed.returncode == 0, armed.stderr  # the updater that armed the move is gone

    gate = tmp_path / "filter-ready"
    (tmp_path / "hold.py").write_text(f"from pathlib import Path\nimport time\nPath({str(gate)!r}).touch()\n"
                                      "time.sleep(120)\n", encoding="utf-8")
    (root / ".git/info/attributes").write_text(f"{rel} filter=hold\n", encoding="utf-8")
    git("config", "filter.hold.smudge", shlex.join([sys.executable, str(tmp_path / "hold.py")]))
    git("config", "filter.hold.clean", "cat")
    git("config", "filter.hold.required", "true")
    merge = subprocess.Popen([git_exe, "-C", str(root), "merge", "--ff-only", target], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not gate.exists():
            assert merge.poll() is None and time.monotonic() < deadline, merge.communicate()
            time.sleep(0.01)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(merge.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test (platforms marker)
        merge.wait()
    assert not (root / rel).exists()
    git("config", "--unset", "filter.hold.smudge")
    (root / ".git/info/attributes").unlink()

    (tmp_path / "bin").mkdir()
    launcher = _launchers._mint_shell_launcher("hermes", tmp_path / "bin", Path(sys.executable),
                                               _launchers._launcher_script("hermes", root, None))
    assert launcher is not None
    return root, env, original, launcher


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(("rel", "torn", "published"), [
    ("hermes_bootstrap.py", False, True),
    ("hermes_cli/_early_recovery.py", True, True),
    ("hermes_cli/__init__.py", True, True),
    ("hermes_cli/update_lock.py", True, True),
    # An updater that predates publication: the launcher reads the same files from git's objects.
    ("hermes_bootstrap.py", True, False),
])
def test_a_launch_repairs_a_move_killed_while_writing_the_repairs_own_code(tmp_path, rel, torn, published):
    root, env, original, launcher = _killed_mid_write(tmp_path, rel)
    if torn:  # git's write cut short: a prefix of the new blob
        (root / rel).write_bytes(original[:12])
    if not published:
        shutil.rmtree(root / ".git/hermes-update-recovery")
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert (root / rel).read_bytes() == original
    assert not (root / ".git/hermes-update-pull").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("published", [True, False])
def test_the_published_repair_leaves_the_tree_to_a_live_writer_holding_the_checkout(tmp_path, published):
    root, env, original, launcher = _killed_mid_write(tmp_path, "hermes_bootstrap.py")
    if not published:
        # The closure comes from git's objects, and no git was recorded: the launcher's git is an
        # absolute PATH entry's, never a ``git`` in the current directory an empty entry names (m4).
        shutil.rmtree(root / ".git/hermes-update-recovery")
        marker = root / ".git/hermes-update-pull"
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
        marker.write_text("".join(line + "\n" for line in lines if not line.startswith("git=")), encoding="utf-8")
        (tmp_path / "git").write_text(f"#!/bin/sh\ntouch {shlex.quote(str(tmp_path / 'CWD_GIT_RAN'))}\nexit 1\n",
                                      encoding="utf-8")
        (tmp_path / "git").chmod(0o755)
    holder = subprocess.Popen([sys.executable, "-I", "-c", _HOLDER, str(SOURCE), str(root)], env=env,
                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "HELD"
        launch = subprocess.run([str(launcher)], cwd=tmp_path, env={**env, "PATH": os.pathsep + env["PATH"]},
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False)
        assert launch.returncode != 0 and "Not repairing the checkout now" in launch.stderr, launch.stderr
        # One reason, not a "repairing it" banner followed by the import's traceback (m3).
        assert "Traceback" not in launch.stderr and "repairing it" not in launch.stderr, launch.stderr
        assert not (root / "hermes_bootstrap.py").exists(), "repaired under a live writer"
        assert (root / ".git/hermes-update-pull").exists()
        assert not (tmp_path / "CWD_GIT_RAN").exists()
        assert (root / ".git/hermes-update-recovery").is_dir(), "closure not rebuilt from git's objects"
    finally:
        holder.kill()
        holder.wait()
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert (root / "hermes_bootstrap.py").read_bytes() == original



@pytest.mark.platforms("posix")
@pytest.mark.parametrize("damage", ["truncated", "foreign"])
def test_a_damaged_published_closure_is_rebuilt_from_git_objects_not_trusted(tmp_path, damage):
    """A power loss can leave the published closure with bytes git never had: they never run, and the
    launch rebuilds the closure from ``pre``'s objects before repairing (M1)."""
    root, env, original, launcher = _killed_mid_write(tmp_path, "hermes_bootstrap.py")
    (closure,) = (root / ".git/hermes-update-recovery").iterdir()
    module = closure / "hermes_cli/_early_recovery.py"
    module.write_bytes(b"" if damage == "truncated" else
                       module.read_bytes() + b"\nprint('NOT_PRE_CODE_RAN', file=__import__('sys').stderr)\n")
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert "NOT_PRE_CODE_RAN" not in launch.stderr, launch.stderr
    assert (root / "hermes_bootstrap.py").read_bytes() == original
    assert not (root / ".git/hermes-update-pull").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("corrupt", [False, True])
def test_git_object_bytes_that_do_not_hash_to_pres_blob_never_run(tmp_path, corrupt):
    """No published closure, and ``pre``'s loose object of the repair inflates to other bytes: they
    never run and nothing is published; the marker stays for a later launch or update (N02)."""
    ran = tmp_path / "FOREIGN_RAN"
    root, env, _original, launcher = _killed_mid_write(tmp_path, "hermes_bootstrap.py")
    (closure,) = (root / ".git/hermes-update-recovery").iterdir()  # named for ``pre``
    shutil.rmtree(closure.parent)
    if corrupt:
        oid = subprocess.check_output(["git", "-C", str(root), "rev-parse", f"{closure.name}:hermes_cli/_early_recovery.py"],
                                      env=env, text=True).strip()
        obj = root / ".git/objects" / oid[:2] / oid[2:]
        foreign = f"open({str(ran)!r}, 'w').close()\n".encode()
        obj.chmod(0o644)
        obj.write_bytes(zlib.compress(b"blob %d\0" % len(foreign) + foreign))
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert not ran.exists(), launch.stderr
    assert (launch.returncode == 0) is not corrupt, launch.stderr
    assert (root / "hermes_bootstrap.py").exists() is not corrupt
    assert (root / ".git/hermes-update-pull").exists() is corrupt
    assert not (corrupt and (root / ".git/hermes-update-recovery").exists()), "published foreign bytes"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("shadowed", [False, True])
def test_a_stdlib_named_file_in_the_tree_never_runs_before_the_repair(tmp_path, shadowed):
    """The repair is stdlib plus ``pre``'s closure only: a root ``shutil.py`` in the checkout is not
    imported in its place, and the torn tree is still repaired (N03)."""
    ran = tmp_path / "SHADOW_RAN"
    shadow = {"shutil.py": f"open({str(ran)!r}, 'w').close()\nraise ImportError('shadow')\n"} if shadowed else {}
    root, env, original, launcher = _killed_mid_write(tmp_path, "hermes_bootstrap.py", shadow)
    launch = subprocess.run([str(launcher)], cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=60, check=False)
    assert not ran.exists(), launch.stderr
    assert launch.returncode == 0 and "APP_REACHED" in launch.stdout, launch.stderr
    assert (root / "hermes_bootstrap.py").read_bytes() == original
    assert not (root / ".git/hermes-update-pull").exists()


def test_the_closure_is_pres_committed_bytes_read_in_one_cat_file_spawn(tmp_path, monkeypatch):
    """The published repair is ``pre``'s committed bytes, not the working tree's, read with one
    ``git cat-file --batch`` (the preflight's batched reader), not a ``cat-file`` per file."""
    from hermes_cli import update_cmd_commit as commit
    from hermes_cli._early_recovery import RECOVERY_CLOSURE

    root = tmp_path / "checkout"
    for rel in RECOVERY_CLOSURE:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(f"# {rel} at pre\n".encode())
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    for args in (("init", "-q", "-b", "main"), ("add", "-A"),
                 ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "pre")):
        subprocess.run(["git", "-C", str(root), *args], env=env, check=True)
    pre = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    (root / RECOVERY_CLOSURE[0]).write_bytes(b"# a working-tree edit, never published\n")
    spawns = []
    real = commit.run_git
    monkeypatch.setattr(commit, "run_git", lambda git_cmd, args, **kw: spawns.append(args[0]) or real(git_cmd, args, **kw))

    closure = commit.publish_recovery_closure(["git"], root, pre)

    for rel in RECOVERY_CLOSURE:
        assert (closure / rel).read_bytes() == f"# {rel} at pre\n".encode()
    assert spawns == ["ls-tree", "cat-file"]
