"""Checkout custody for the updater's children (R2): the ONE spawn policy for every git (and the
Node build) an update starts.

The checkout kernel lock (``update_lock``) must stay held while any process of the update tree
can still write the checkout, and must NOT leak into processes that outlive the update:

* Every updater git call carries ``-c gc.autoDetach=false -c maintenance.auto=false``: git never
  forks a detached gc/maintenance child that would inherit (POSIX) or outlive (Windows) the lock.
* POSIX: the lock fd is inherited ONLY by git commands that mutate the worktree, index or refs
  locally (:data:`LOCAL_MUTATORS`, run with ``core.fsmonitor=false`` so no fsmonitor daemon
  starts under them). Network/credential commands (fetch, ls-remote, credential) and readers run
  without it: a ``git credential-cache--daemon`` they start never holds the checkout. A killed
  fetch that leaves ``*.lock`` ref files is recovered by the stale-lock rules
  (``gitlock.clear_stale_git_locks``).
* Windows has no fd inheritance: while this process holds (or joined) the checkout lock, every
  child started here is created SUSPENDED, assigned to the update's kill-on-close job and only
  then resumed, so it and everything it spawns die with the lock owner. The Node build goes
  through ``pm.progress.run_contained`` (whose Popen it never sees): :func:`contained_command`
  wraps it in a stdlib launcher that joins the job before it starts node.

Every updater git runner (``update_cmd._git_run``, ``update_cmd_git._git_run``,
``update_cmd_stash``, ``update_cmd_check``, ``gitlock``, ``update_cmd_commit``) calls
:func:`run_git`; nothing else spawns updater git.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# No detached child: `git gc --auto` / `maintenance run --auto --detach` would otherwise fork a
# daemonized repack after any command that writes objects (commit, merge, fetch, stash).
GIT_NO_DETACH = ("-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")
# Local mutators only (they hold the lock fd): never start an fsmonitor daemon under it.
_MUTATOR_CONFIG = ("-c", "core.fsmonitor=false")

# git subcommands that write the worktree, the index or refs on THIS machine. Only these inherit
# the checkout lock fd: if the updater dies mid-command, the checkout stays locked until git exits.
LOCAL_MUTATORS = frozenset({
    "add", "am", "apply", "checkout", "checkout-index", "cherry-pick", "clean", "commit", "merge",
    "mv", "pull", "read-tree", "rebase", "reset", "restore", "revert", "rm", "stash", "switch",
    "update-index", "update-ref", "symbolic-ref", "tag", "branch", "worktree",
})

# Global options before the subcommand that take a separate value argument.
_GLOBAL_WITH_VALUE = frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path",
                                "--config-env", "--super-prefix", "--attr-source", "--list-cmds"})

_CREATE_SUSPENDED = 0x00000004


def git_subcommand(args: Sequence[str]) -> str | None:
    """The subcommand in ``args`` (everything after the git executable), skipping global options."""
    it = iter(args)
    for arg in it:
        if arg in _GLOBAL_WITH_VALUE:
            next(it, None)
        elif not arg.startswith("-"):
            return arg
    return None


def is_local_mutator(args: Sequence[str]) -> bool:
    return git_subcommand(args) in LOCAL_MUTATORS


def git_argv(git_cmd: Sequence[str], args: Sequence[str]) -> list[str]:
    """``git_cmd + args`` with the custody config inserted before the subcommand."""
    git_cmd, args = list(git_cmd), list(args)
    extra = GIT_NO_DETACH + (_MUTATOR_CONFIG if is_local_mutator(git_cmd[1:] + args) else ())
    return [*git_cmd, *extra, *args]


def _held() -> dict | None:
    try:
        from hermes_cli import update_lock
    except Exception:  # noqa: BLE001 - a torn tree's launch repair: no updater runs from it
        return None
    return update_lock._HELD


def _death_signal_preexec():
    """Linux: a ``preexec_fn`` making the child die (SIGKILL) with the updater thread that waits
    on it. For the git children that do NOT hold the lock fd (fetch, ls-remote, readers): a
    killed owner must not leave a fetch rewriting refs under the next lock owner. The ruling's
    stale-lock rules recover a fetch killed mid-write. ``None`` elsewhere (Windows: the job;
    macOS: no parent-death signal — see NOT_COVERED)."""
    if not sys.platform.startswith("linux"):
        return None
    import ctypes
    import os

    prctl = ctypes.CDLL(None, use_errno=True).prctl  # resolved before fork: the child only calls
    parent = os.getpid()

    def _arm():
        prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
        if os.getppid() != parent:  # the owner died between fork and prctl
            os._exit(137)

    return _arm


def _custody_kwargs(inherit_lock: bool, kwargs: dict) -> dict:
    fds = _lock_fds(inherit_lock)
    if fds:
        kwargs["pass_fds"] = tuple(dict.fromkeys((*kwargs.get("pass_fds", ()), *fds)))
    elif not inherit_lock and _held() is not None and "preexec_fn" not in kwargs:
        arm = _death_signal_preexec()
        if arm is not None:
            kwargs["preexec_fn"] = arm
    return kwargs


def _lock_fds(inherit_lock: bool) -> tuple[int, ...]:
    if not inherit_lock or sys.platform == "win32":
        return ()
    from hermes_cli.update_lock import custody_spawn_kwargs

    return tuple(custody_spawn_kwargs().get("pass_fds", ()))


def _bind_suspended(proc: subprocess.Popen) -> None:
    """Assign a CREATE_SUSPENDED child to the update's kill-on-close job, then resume it.

    Suspended until bound, so nothing it spawns can escape the job. A failed bind still resumes
    (the update must not hang on a weaker lock); a failed resume kills the child."""
    import ctypes

    from hermes_cli.update_lock import bind_child_to_update_tree

    try:
        bind_child_to_update_tree(proc)
    finally:
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtResumeProcess.argtypes = [ctypes.c_void_p]
        ntdll.NtResumeProcess.restype = ctypes.c_long
        if ntdll.NtResumeProcess(int(proc._handle)) != 0:
            proc.kill()
            raise OSError(f"could not resume update child {proc.pid}")


def popen(argv: Sequence[str], *, inherit_lock: bool = False, **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` in the update tree's custody (see the module doc)."""
    if sys.platform == "win32" and _held() is not None:
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_SUSPENDED
        proc = subprocess.Popen(list(argv), **kwargs)
        try:
            _bind_suspended(proc)
        except BaseException:
            with contextlib.suppress(OSError):
                proc.kill()
            raise
        return proc
    return subprocess.Popen(list(argv), **_custody_kwargs(inherit_lock, kwargs))


def run(argv: Sequence[str], *, inherit_lock: bool = False, **kwargs) -> subprocess.CompletedProcess:
    """``subprocess.run`` in the update tree's custody. Off Windows (or outside an update) it IS
    ``subprocess.run`` with the caller's kwargs untouched, plus the lock fd for mutators (or the
    parent-death signal for the rest); on Windows inside an update it is the same contract over
    :func:`popen`, so the child is job-bound before it runs."""
    if not (sys.platform == "win32" and _held() is not None):
        return subprocess.run(list(argv), **_custody_kwargs(inherit_lock, kwargs))
    input, timeout = kwargs.pop("input", None), kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with popen(argv, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            exc.stdout, exc.stderr = proc.communicate()
            raise
        except BaseException:
            proc.kill()
            raise
        code = proc.poll()
    if check and code:
        raise subprocess.CalledProcessError(code, proc.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, code, stdout, stderr)


def run_git(git_cmd: Sequence[str], args: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """THE updater git runner: custody config in argv, the lock fd only into local mutators,
    Windows job binding inside an update. ``kwargs`` are ``subprocess.run``'s."""
    argv = git_argv(git_cmd, args)
    return run(argv, inherit_lock=is_local_mutator(argv[1:]), **kwargs)


# A stdlib launcher for children whose Popen the updater never sees (``run_contained``): it
# joins the job named by an inherited handle, drops that handle (only the owner's handle may
# keep the job open) and runs the real command with the same stdio. A refused join never fails
# the build: the command runs outside the job (the pre-custody behavior) and says so on stderr.
_CUSTODY_UNAVAILABLE = "hermes: update custody unavailable"
_JOIN_JOB = (
    "import ctypes, subprocess, sys\n"
    "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
    "k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]\n"
    "k.GetCurrentProcess.restype = ctypes.c_void_p\n"
    "k.CloseHandle.argtypes = [ctypes.c_void_p]\n"
    "h = ctypes.c_void_p(int(sys.argv[1]))\n"
    "if not k.AssignProcessToJobObject(h, k.GetCurrentProcess()):\n"
    f"    sys.stderr.write('{_CUSTODY_UNAVAILABLE} (could not join the update job: %d); '\n"
    "                     'this child runs outside it\\n' % ctypes.get_last_error())\n"
    "    sys.stderr.flush()\n"
    "k.CloseHandle(h)\n"
    "sys.exit(subprocess.call(sys.argv[2:], stdin=subprocess.DEVNULL))\n"
)


def _join_launcher_python() -> str:
    """The interpreter for the job-joining launcher: the real one, never a venv redirector.

    A venv's ``Scripts\\python.exe`` is a redirector: it starts the base interpreter as its own
    child, inside a job of its own. That child cannot join the update's job once the job holds a
    process from another job hierarchy (the updater's git children): ``AssignProcessToJobObject``
    fails with ERROR_ACCESS_DENIED (5). And the redirector, never joined, keeps its inherited copy
    of the job handle open, so the job would not close when the owner dies. The launcher is
    stdlib-only (``-I -S``), so the base interpreter runs it as is.
    """
    import os

    base = getattr(sys, "_base_executable", None)
    if base and os.path.isfile(base):
        return base
    return sys.executable


@contextlib.contextmanager
def contained_command(argv: Sequence[str], *, inherit_lock: bool = True):
    """``(argv, kwargs)`` for a checkout writer started by a runner that hides its Popen (the Node
    build in ``pm.progress.run_contained``). POSIX: the lock fd. Windows inside an update: the
    command runs under a launcher that joins the update's kill-on-close job first."""
    argv = list(argv)
    if not (sys.platform == "win32" and _held() is not None):
        fds = _lock_fds(inherit_lock)
        yield argv, ({"pass_fds": fds} if fds else {})
        return
    handle = _inheritable_job_handle()
    if handle is None:
        yield argv, {}
        return
    try:
        info = subprocess.STARTUPINFO()
        info.lpAttributeList = {"handle_list": [handle]}
        yield [_join_launcher_python(), "-I", "-S", "-c", _JOIN_JOB, str(handle), *argv], {"startupinfo": info}
    finally:
        import ctypes

        ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))


def _inheritable_job_handle() -> int | None:
    """An inheritable duplicate of the update's kill-on-close job handle (Windows), or None."""
    import ctypes

    from hermes_cli.update_lock import update_tree_job

    try:
        job = update_tree_job()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.DuplicateHandle.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.POINTER(ctypes.c_void_p), ctypes.c_ulong, ctypes.c_int,
                                             ctypes.c_ulong]
        me, dup = kernel32.GetCurrentProcess(), ctypes.c_void_p()
        # DUPLICATE_SAME_ACCESS, inheritable
        if not kernel32.DuplicateHandle(me, job, me, ctypes.byref(dup), 0, True, 0x2):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(dup.value)
    except OSError as exc:
        logger.warning("Could not hand the update's job to a build child: %s", exc)
        return None
