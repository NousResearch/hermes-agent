"""Windows live cells for the checkout lock (contract C1.7): msvcrt byte lock + kill-on-close job.

Invariant under test: the checkout lock is free => no process of the update tree is alive.
On Windows a child cannot inherit an msvcrt lock, so the owner binds every update-tree child
into a kill-on-close job: killing the owner (taskkill /F) kills the child and frees the lock.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli.update_lock import UpdateLock, update_in_progress

pytestmark = pytest.mark.platforms("windows")

REPO_ROOT = Path(__file__).resolve().parents[2]

_OWNER = r"""
import subprocess, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock, bind_child_to_update_tree
lock = UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2])
assert lock.acquire()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                         creationflags=subprocess.CREATE_NO_WINDOW)
bind_child_to_update_tree(child)
print(child.pid, flush=True)
time.sleep(300)
"""


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], text=True, encoding="utf-8", errors="replace",
                         capture_output=True).stdout
    return str(pid) in out


def test_killed_owner_takes_its_tree_down_and_frees_the_lock(tmp_path):
    install = tmp_path / "checkout"
    install.mkdir()
    owner = subprocess.Popen([sys.executable, "-c", _OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m")],
                             stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
    try:
        child = int(owner.stdout.readline().strip())
        assert _alive(child)
        assert update_in_progress(install), "the owner's msvcrt lock is not visible"
        refused = UpdateLock(path=tmp_path / "other-home-marker", install_root=install)
        assert refused.acquire() is False, "a second update of one checkout was not refused"

        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        deadline = time.time() + 15
        while _alive(child) and time.time() < deadline:
            time.sleep(0.2)
        assert not _alive(child), "the update-tree child outlived its killed owner"
        assert not update_in_progress(install), "the lock stayed held after the whole tree died"
        fresh = UpdateLock(path=tmp_path / "other-home-marker", install_root=install)
        assert fresh.acquire() is True
        fresh.release()
    finally:
        if owner.poll() is None:
            owner.kill()



# --- R5b: a child the job refused (it runs outside the job) holds a lease of its own ----------

_JOINING_CHILD = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2], checkout_first=False).acquire_checkout(sys.argv[2])
print("joined", flush=True)
time.sleep(300)
"""

_UNBOUND_OWNER = r"""
import subprocess, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
assert UpdateLock(path=Path(sys.argv[3]), install_root=sys.argv[2]).acquire()
# The job refused it: a plain child, outside any kill-on-close job (a refused bind still runs).
child = subprocess.Popen([sys.executable, "-c", sys.argv[4], *sys.argv[1:4]], stdout=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, text=True, encoding="utf-8")
assert child.stdout.readline().strip() == "joined"
print(child.pid, flush=True)
child.wait()
"""


def test_a_child_the_job_refused_keeps_the_checkout_busy_after_its_owner_dies(tmp_path):
    """R5b: never two writers. The owner is killed while its unbound child still runs: the
    checkout must read held (the child's lease) until that child exits, then free."""
    install = tmp_path / "checkout"
    install.mkdir()
    owner = subprocess.Popen([sys.executable, "-c", _UNBOUND_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m"),
                              _JOINING_CHILD], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True,
                             encoding="utf-8")
    child = None
    try:
        child = int(owner.stdout.readline().strip())
        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        assert _alive(child), "fixture: the unbound child must outlive its owner"
        assert update_in_progress(install), "the checkout reads free while the refused child still runs"
        assert UpdateLock(path=tmp_path / "other-home-marker", install_root=install).acquire() is False
    finally:
        if child is not None:
            subprocess.run(["taskkill", "/F", "/PID", str(child)], capture_output=True, check=False)
        if owner.poll() is None:
            owner.kill()
    deadline = time.time() + 15
    while _alive(child) and time.time() < deadline:
        time.sleep(0.2)
    fresh = UpdateLock(path=tmp_path / "other-home-marker", install_root=install)
    assert fresh.acquire() is True, "the lease outlived the child that held it"
    fresh.release()


# --- R2 on Windows: the updater's git and Node children join the owner's kill-on-close job -----

_GIT_OWNER = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.update_cmd import _git_run
root = Path(sys.argv[2])
assert UpdateLock(path=Path(sys.argv[3]), install_root=root).acquire()
_git_run(["git"], ["stash", "push", "-m", "custody"], cwd=root)
"""

_BUILD_OWNER = r"""
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hermes_cli.update_lock import UpdateLock
from hermes_cli.source_build import run_source_script
from hermes_cli.update_custody import run_git
root = Path(sys.argv[2])
assert UpdateLock(path=Path(sys.argv[3]), install_root=root).acquire()
# A git child first, as in every install/update: the job then already holds a process when the
# build's launcher joins it (a venv redirector's child was refused here: ERROR_ACCESS_DENIED).
run_git(["git"], ["--version"], capture_output=True, check=True)
env = {**os.environ, "PATH": sys.argv[4] + os.pathsep + os.environ["PATH"]}
run_source_script(root, "build.mjs", env=env, label="probe build")
"""


def _blocker(tmp_path: Path) -> str:
    """Python source that records its pid and blocks until ``go`` exists."""
    pid_file, go = (tmp_path / "blocker.pid").as_posix(), (tmp_path / "go").as_posix()
    return (f"import os, time\nopen({pid_file!r}, 'w').write(str(os.getpid()))\n"
            f"while not os.path.exists({go!r}): time.sleep(0.05)\nprint('cleaned')\n")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"})


def _tree_dies_with_its_owner(tmp_path: Path, install: Path, owner_args: list[str]) -> None:
    import psutil

    owner = subprocess.Popen([sys.executable, "-c", *owner_args], stdin=subprocess.DEVNULL)
    pid_file, tree = tmp_path / "blocker.pid", []
    try:
        deadline = time.time() + 60
        while not (pid_file.exists() and pid_file.read_text(encoding="utf-8-sig").strip()):
            assert owner.poll() is None, f"owner exited {owner.returncode} before its child blocked"
            assert time.time() < deadline, "the blocking child never started"
            time.sleep(0.1)
        blocker = psutil.Process(int(pid_file.read_text(encoding="utf-8-sig")))
        tree = [blocker, *(p for p in blocker.parents() if p.pid != owner.pid and owner.pid in
                           {q.pid for q in p.parents()})]
        assert UpdateLock(path=tmp_path / "other-marker", install_root=install).acquire() is False

        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        deadline = time.time() + 15
        while any(p.is_running() for p in tree) and time.time() < deadline:
            time.sleep(0.2)
        survivors = [f"{p.pid}:{p.name()}" for p in tree if p.is_running()]
        assert not survivors, f"update children outlived their killed owner: {survivors}"
        fresh = UpdateLock(path=tmp_path / "other-marker", install_root=install)
        assert fresh.acquire() is True, "the checkout stayed locked after the whole tree died"
        fresh.release()
    finally:
        (tmp_path / "go").touch()
        for proc in tree:
            with contextlib.suppress(Exception):
                proc.kill()
        if owner.poll() is None:
            owner.kill()


def _git_install(tmp_path: Path) -> Path:
    """A checkout whose ``f.txt`` clean filter is the blocker: a ``stash push`` blocks in git."""
    exe = Path(sys.executable).as_posix()
    if any(ch in exe for ch in " =~%#'\"&;|<>()$`*?["):
        pytest.skip("the clean filter must run without a shell: interpreter path has shell metacharacters")
    install = tmp_path / "checkout"
    install.mkdir()
    _git(install, "init", "-q")
    _git(install, "config", "user.name", "probe")
    _git(install, "config", "user.email", "probe@invalid.local")
    (install / "f.txt").write_text("before\n", encoding="utf-8")
    _git(install, "add", "f.txt")
    _git(install, "commit", "-qm", "before")
    # The clean filter is the bare interpreter: it runs the file's content (stdin) as a program.
    _git(install, "config", "filter.block.clean", exe)
    (install / ".git" / "info" / "attributes").write_text("f.txt filter=block\n", encoding="utf-8")
    (install / "f.txt").write_text(_blocker(tmp_path), encoding="utf-8")
    return install


def _build_install(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout whose ``node`` (a .bat on the build PATH) runs the blocker."""
    install = tmp_path / "checkout"
    install.mkdir()
    (install / "build.py").write_text(_blocker(tmp_path), encoding="utf-8")
    (install / "build.mjs").write_text("// stands in for a build script\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "node.bat").write_text(f'@"{sys.executable}" "{install / "build.py"}"\r\n', encoding="utf-8")
    return install, bin_dir


def test_killed_owner_takes_its_git_child_down(tmp_path):
    install = _git_install(tmp_path)
    _tree_dies_with_its_owner(tmp_path, install, [_GIT_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m")])


def test_killed_owner_takes_its_node_build_down(tmp_path):
    install, bin_dir = _build_install(tmp_path)
    _tree_dies_with_its_owner(tmp_path, install,
                              [_BUILD_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m"), str(bin_dir)])


# --- D2: a child the job refuses is fenced or never runs ---------------------------------------
#
# Negative control: the update's job handle is replaced by an event handle right before the
# writer starts, so the REAL AssignProcessToJobObject refuses it (ERROR_INVALID_HANDLE) the way it
# refuses a process it cannot nest. Invariant: once the owner is killed, an update writer still
# alive means the checkout lock is still held — or the writer never ran and the owner refused it
# with a clear message. Never a live writer behind a free lock.
_REFUSE_JOBS = (
    "import ctypes\n"
    "from hermes_cli import update_lock as _ul\n"
    "_k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
    "_k.CreateEventW.restype = ctypes.c_void_p\n"
    "_ul._JOBS[:] = [_k.CreateEventW(None, True, False, None)]\n"
)
_GIT_REFUSED_OWNER = _GIT_OWNER.replace("_git_run([", _REFUSE_JOBS + "_git_run([")
_BUILD_REFUSED_OWNER = _BUILD_OWNER.replace("env = {", _REFUSE_JOBS + "env = {")
REFUSED = "so it was not run"


def _writer_fenced_or_refused(tmp_path: Path, install: Path, owner_args: list[str]) -> str:
    import psutil

    # Output to a file, not a pipe: nobody drains a pipe while we poll, and a refusal's traceback
    # (it quotes the launcher argv) outgrows the pipe buffer and blocks the owner on write.
    log = tmp_path / "owner.log"
    with open(log, "wb") as sink:
        owner = subprocess.Popen([sys.executable, "-c", *owner_args], stdin=subprocess.DEVNULL,
                                 stdout=sink, stderr=subprocess.STDOUT)
    pid_file, tree = tmp_path / "blocker.pid", []

    def started() -> bool:
        return pid_file.exists() and bool(pid_file.read_text(encoding="utf-8-sig").strip())

    try:
        deadline = time.time() + 60
        while not started() and owner.poll() is None:
            assert time.time() < deadline, "the owner neither ran nor refused its writer"
            time.sleep(0.1)
        time.sleep(0.5)  # an exiting owner's writer may still be starting
        if not started():
            owner.wait(timeout=30)
            out = log.read_text(encoding="utf-8-sig", errors="replace")
            assert owner.returncode != 0 and REFUSED in out, f"the owner neither ran nor refused its writer:\n{out}"
            return "refused"
        blocker = psutil.Process(int(pid_file.read_text(encoding="utf-8-sig")))
        tree = [blocker, *(p for p in blocker.parents() if p.pid != owner.pid and owner.pid in
                           {q.pid for q in p.parents()})]
        subprocess.run(["taskkill", "/F", "/PID", str(owner.pid)], capture_output=True, check=False)
        owner.wait(timeout=30)
        deadline = time.time() + 15
        while any(p.is_running() for p in tree) and time.time() < deadline:
            time.sleep(0.2)
        survivors = [f"{p.pid}:{p.name()}" for p in tree if p.is_running()]
        assert not survivors or update_in_progress(install), \
            f"update writer(s) {survivors} outlived the killed owner while the checkout lock is free"
        return "fenced"
    finally:
        (tmp_path / "go").touch()
        for proc in tree:
            with contextlib.suppress(Exception):
                proc.kill()
        if owner.poll() is None:
            owner.kill()


def test_a_git_child_the_job_refuses_never_runs_unfenced(tmp_path):
    install = _git_install(tmp_path)
    outcome = _writer_fenced_or_refused(
        tmp_path, install, [_GIT_REFUSED_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m")])
    if outcome == "refused":  # refused before it ran: the checkout is untouched
        stash = subprocess.run(["git", "-C", str(install), "stash", "list"], capture_output=True)
        assert stash.stdout == b"", stash


def test_a_node_build_the_job_refuses_never_runs_unfenced(tmp_path):
    install, bin_dir = _build_install(tmp_path)
    _writer_fenced_or_refused(
        tmp_path, install, [_BUILD_REFUSED_OWNER, str(REPO_ROOT), str(install), str(tmp_path / "m"), str(bin_dir)])


def test_a_refused_job_join_never_runs_the_command(tmp_path):
    """D2: a launcher whose join is refused exits without running its command (m1: and says so)."""
    from hermes_cli.update_custody import _CUSTODY_UNAVAILABLE, _JOIN_JOB, _REFUSED_EXIT

    child = "import sys; print('built'); sys.exit(3)"
    launcher = tmp_path / "join_job.py"  # a file, not -c: the guard reads argv text as a command line
    launcher.write_text(_JOIN_JOB, encoding="utf-8")
    report = tmp_path / "report.txt"
    out = subprocess.run([sys.executable, "-I", "-S", str(launcher), "0", str(report), sys.executable, "-c", child],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert out.returncode == _REFUSED_EXIT, out
    assert "built" not in out.stdout
    assert _CUSTODY_UNAVAILABLE in out.stderr
    # m1: the notice also lands in the report the updater turns into a warning + receipt step
    assert _CUSTODY_UNAVAILABLE in report.read_text(encoding="utf-8-sig")



def test_a_command_that_would_start_outside_the_job_never_runs(tmp_path):
    """F54: a launcher interpreter that joins while what it starts escapes the job (a Store
    Python alias) must not run the command. Control: a job with SILENT_BREAKAWAY_OK, which the
    launcher joins but whose children start outside it."""
    import ctypes

    from hermes_cli.update_custody import _CUSTODY_UNAVAILABLE, _JOIN_JOB, _REFUSED_EXIT

    class _Basic(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [("user_limits", ctypes.c_int64 * 2), ("LimitFlags", ctypes.c_uint32),
                    ("working_set", ctypes.c_size_t * 2), ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.c_size_t), ("classes", ctypes.c_uint32 * 2)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    job = kernel32.CreateJobObjectW(None, None)
    limits = _Basic(LimitFlags=0x1000)  # JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
    assert kernel32.SetInformationJobObject(job, 2, ctypes.byref(limits), ctypes.sizeof(limits))
    os.set_handle_inheritable(job, True)
    launcher = tmp_path / "join_job.py"
    launcher.write_text(_JOIN_JOB, encoding="utf-8")
    info = subprocess.STARTUPINFO()
    info.lpAttributeList = {"handle_list": [job]}
    out = subprocess.run([sys.executable, "-I", "-S", str(launcher), str(job), str(tmp_path / "report.txt"),
                          sys.executable, "-c", "print('built')"], startupinfo=info, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=60)
    assert out.returncode == _REFUSED_EXIT, out
    assert "built" not in out.stdout and _CUSTODY_UNAVAILABLE in out.stderr, out



# --- R8 m1: the completion child is bound before it runs; a refused bind reaches the receipt ---

_COMPLETION_CHILD = (
    "import json, os, sys\n"
    "from pathlib import Path\n"
    "Path(os.environ['HERMES_PROBE_STARTED']).touch()\n"  # its first instruction
    "request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))\n"
    "receipt = dict(request['receipt'], finished_at='now', outcome='success')\n"
    "Path(sys.argv[2]).write_text(json.dumps({'schema': 1, 'update_id': receipt['update_id'], 'exit_code': 0,\n"
    "                                         'receipt': receipt, 'windows_resume': None}), encoding='utf-8')\n"
)


def test_a_completion_child_runs_only_after_its_bind_and_a_refusal_is_receipted(tmp_path, monkeypatch):
    """The completion child is a checkout writer (dependency sync, product builds). It is created
    suspended and bound before it runs one instruction; when the real AssignProcessToJobObject
    refuses it (handed an event handle), it still runs (post-commit, ruling) but the refusal is a
    failed ``update_custody`` step in the receipt it finishes."""
    import ctypes
    import copy

    from hermes_cli import update_completion, update_lock, update_receipt

    root = tmp_path / "checkout"
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "update_completion.py").write_text(_COMPLETION_CHILD, encoding="utf-8")
    started = tmp_path / "started"
    monkeypatch.setenv("HERMES_PROBE_STARTED", str(started))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateEventW.restype = ctypes.c_void_p
    event = kernel32.CreateEventW(None, True, False, None)
    ran_before_bind = []

    def refusing_job() -> int:
        time.sleep(2.0)  # a child already running has touched its marker by now
        ran_before_bind.append(started.exists())
        return event

    monkeypatch.setattr(update_lock, "update_tree_job", refusing_job)
    with update_receipt.update_receipt_scope():
        update_receipt.begin_update_receipt()
        request = {"source": str(root), "home": str(tmp_path / "home"),
                   "receipt": copy.deepcopy(update_receipt._current.get().data)}
        result = update_completion.run_completion(request)
    assert ran_before_bind == [False], "the completion child ran before its job bind"
    assert result["exit_code"] == 0 and started.exists(), result
    steps = [s for s in result["receipt"].get("steps", []) if s["name"] == "update_custody"]
    assert steps and steps[0]["ok"] is False and "completion child" in steps[0]["detail"], result["receipt"]


# --- R8 m2: a refusal a reader swallows is what `hermes update` reports -----------------------

_SWALLOWING_OWNER = r"""
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from hermes_cli import main, update_cmd, update_owning_install
from hermes_cli.update_cmd_git import _git_stdout
root = Path(sys.argv[2])
main.PROJECT_ROOT = root
update_owning_install.retarget_to_owning_install = lambda project_root: None
main._update_preflight_handled = lambda args: False
main._install_hangup_protection = lambda **kw: None
main._finalize_update_output = lambda state: None

def impl(args, gateway_mode):
    from hermes_cli.update_receipt import begin_update_receipt
    begin_update_receipt()
    REFUSE_JOBS
    # The real readers swallow the refusal: HEAD reads as unknown.
    pre, head = update_cmd._capture_head_sha(["git"], root), _git_stdout(["git"], ["rev-parse", "HEAD"], root)
    print(f"✗ Could not resolve the checkout's HEAD ({pre!r}, {head!r})", flush=True)
    sys.exit(1)

update_cmd._cmd_update_impl = impl
main.cmd_update(SimpleNamespace(gateway=False))
"""


def test_a_refusal_the_readers_swallow_is_what_the_update_reports(tmp_path):
    install = tmp_path / "checkout"
    install.mkdir()
    _git(install, "init", "-q")
    refuse = "\n    ".join(_REFUSE_JOBS.strip().splitlines())
    home = tmp_path / "home"
    out = subprocess.run([sys.executable, "-c", _SWALLOWING_OWNER.replace("REFUSE_JOBS", refuse), str(REPO_ROOT),
                          str(install)], stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=120,
                         env={**os.environ, "HERMES_HOME": str(home), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    text = out.stdout + out.stderr
    assert out.returncode == 1, text
    assert "`hermes update` stopped: Windows would not put `git` in this update's process job" in text, text
    assert "Nothing was changed" in text and "Run `hermes update` again from a regular terminal" in text, text


# --- F03/N11: both historical takeover hops are bound to the job (or leased and receipted) -----

def _refusing_job(monkeypatch):
    import ctypes

    from hermes_cli import update_lock

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateEventW.restype = ctypes.c_void_p
    event = kernel32.CreateEventW(None, True, False, None)
    monkeypatch.setattr(update_lock, "update_tree_job", lambda: event)


def test_the_takeover_finish_child_is_never_silently_outside_the_job(tmp_path, monkeypatch):
    """F03: the takeover's update_finish child builds the checkout. It goes through the job bind;
    a refused bind still runs it (the update committed) but is receipted."""
    import json

    from hermes_cli import _update_takeover, update_receipt

    root = tmp_path / "checkout"
    (root / "hermes_cli").mkdir(parents=True)
    started = tmp_path / "started"
    (root / "hermes_cli" / "update_finish.py").write_text(
        f"import sys\nopen({str(started)!r}, 'w').close()\nopen(sys.argv[2], 'w').write('{{}}')\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(_update_takeover, "prepare", lambda request: (Path(sys.executable), dict(os.environ)))
    _refusing_job(monkeypatch)
    context = tmp_path / "request.json"
    context.write_text(json.dumps({"root": str(root), "receipt": {}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["takeover", str(context), str(tmp_path / "result.json")])
    with update_receipt.update_receipt_scope():
        code = _update_takeover.main()
        steps = update_receipt._current.get().data.get("steps", [])
    assert code == 0 and started.exists()
    assert any(s["name"] == "update_custody" and s["ok"] is False for s in steps), steps


def test_the_old_updater_runs_its_takeover_in_the_job(tmp_path, monkeypatch, capsys):
    """N11: the old updater's takeover child syncs and builds the checkout. With this updater
    holding the checkout lock it is bound before it runs; a refused bind is said out loud."""
    from hermes_cli import _old_updater

    started = tmp_path / "started"
    lock = UpdateLock(path=tmp_path / "m", install_root=tmp_path)
    assert lock.acquire()
    try:
        _refusing_job(monkeypatch)
        code = _old_updater._run_in_custody(
            [sys.executable, "-c", f"open({str(started)!r}, 'w').close()"], tmp_path, cwd=tmp_path)
    finally:
        lock.release()
    assert code == 0 and started.exists()
    assert "would not take the takeover" in capsys.readouterr().out
