"""Checkout custody for the updater's children (R2): the ONE spawn policy for every git (and the
Node build) an update starts.

The checkout kernel lock (``update_lock``) must stay held while any process of the update tree
can still write the checkout, and must NOT leak into processes that outlive the update:

* Every updater git call carries ``-c gc.autoDetach=false -c maintenance.auto=false``: git never
  forks a detached gc/maintenance child that would inherit (POSIX) or outlive (Windows) the lock.
* POSIX: the lock fd is inherited ONLY by git commands that mutate the worktree, index or refs
  locally (:data:`LOCAL_MUTATORS`, run with ``core.fsmonitor=false`` so no fsmonitor daemon
  starts under them, with no credential helper and no repository hooks). Network/credential commands (fetch,
  ls-remote, credential) and readers run without it: a ``git credential-cache--daemon`` they
  start never holds the checkout. A partial clone's mutator would lazily fetch the objects a move
  needs as a child holding the fd (and start the daemon under it), so before a move the objects
  are fetched without the fd and with the user's helpers (:func:`_prefetch_for_move`); a lazy
  fetch the prefetch missed still runs, just without a helper. A killed fetch that leaves
  ``*.lock`` ref files is recovered by the stale-lock rules (``gitlock.clear_stale_git_locks``).
* Windows has no fd inheritance: while this process holds (or joined) the checkout lock, every
  child started here is created SUSPENDED, assigned to the update's kill-on-close job and only
  then resumed, so it and everything it spawns die with the lock owner. The Node build goes
  through ``pm.progress.run_contained`` (whose Popen it never sees): :func:`contained_command`
  wraps it in a stdlib launcher that joins the job before it starts node.
* Windows fails CLOSED (D2): a child the job refuses is never run. The msvcrt byte lock belongs
  to the process that took it — an inherited handle does not keep it, the kernel unlocks it when
  the owner exits — so nothing else could fence a writer outside the job: after the owner's
  death the lock would be free while it still wrote. A refused bind kills the still-suspended
  child; a refused launcher join exits before it starts node; either raises
  :class:`CustodyRefused`, a clear refusal of that step instead of an unfenced writer.

Every updater git runner (``update_cmd._git_run``, ``update_cmd_git._git_run``,
``update_cmd_stash``, ``update_cmd_check``, ``gitlock``, ``update_cmd_commit``,
``_early_recovery``'s restore) calls :func:`run_git`.
Not covered, on purpose: the read-only release/check readers (``source_releases``,
``source_check``: ``ls-remote``, ``cherry``, ``rev-parse``) run their own probes; they never get
the lock fd and write nothing, so they cannot leak or outlive custody of the checkout.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)

# No detached child: `git gc --auto` / `maintenance run --auto --detach` would otherwise fork a
# daemonized repack after any command that writes objects (commit, merge, fetch, stash).
GIT_NO_DETACH = ("-c", "gc.autoDetach=false", "-c", "maintenance.auto=false")
# Local mutators only (they hold the lock fd): never start an fsmonitor daemon under it, and no
# credential helper — a promisor lazy fetch under a mutator would start `git
# credential-cache--daemon` holding the fd for its lifetime (900 s by default, m3). No repository
# hooks either (F1): a hook inherits the fd, and one that backgrounds a process (a post-merge
# daemon) kept a completed update's checkout locked until it exited. Per command only: the
# repository's own hook configuration is untouched.
_MUTATOR_CONFIG = ("-c", "core.fsmonitor=false", "-c", "credential.helper=", "-c", f"core.hooksPath={os.devnull}")

# git subcommands that write the worktree, the index or refs on THIS machine. Only these inherit
# the checkout lock fd: if the updater dies mid-command, the checkout stays locked until git exits.
# No `pull` (fetch + merge: its network half must not hold the fd; the updater fetches, then
# merges). `gc` packs refs (and its repack writes the object store); `pack-objects` is the pack
# tidy's merge, whose output the update moves into the object store.
LOCAL_MUTATORS = frozenset({
    "add", "am", "apply", "checkout", "checkout-index", "cherry-pick", "clean", "commit", "gc",
    "merge", "mv", "pack-objects", "read-tree", "rebase", "reset", "restore", "revert", "rm", "stash",
    "switch", "update-index", "update-ref", "symbolic-ref", "tag", "branch", "worktree",
})
# Mutators that move the checkout to another commit: what they read from a promisor remote is
# fetched first, without the fd (_prefetch_for_move).
_MOVES = frozenset({"checkout", "merge", "read-tree", "reset", "switch"})

# Global options before the subcommand that take a separate value argument.
_GLOBAL_WITH_VALUE = frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path",
                                "--config-env", "--super-prefix", "--attr-source", "--list-cmds"})

_CREATE_SUSPENDED = 0x00000004


class CustodyRefused(OSError):
    """Windows would not put an update child in the update's kill-on-close job, so the child was
    never run (D2): outside the job it could outlive a killed update and write the checkout after
    the checkout lock is free."""

    def __init__(self, argv: Sequence[str], cause: object) -> None:
        import os

        name = os.path.basename(str(argv[0])) if argv else "child"
        self.reason = (
            f"Windows would not put `{name}` in this update's process job ({cause}), so it was not "
            "run: a checkout writer outside the job could outlive a killed update and write the "
            "checkout after its lock is released.")
        super().__init__(f"{self.reason} {_RETRY.format(command='the command')}")


_RETRY = ("Run {command} again from a regular terminal, outside any sandbox or tool that confines "
          "the processes it starts.")


# This run's custody (m2): whether an update child already ran in custody, and the first refusal.
# Many readers swallow an OSError (a reader's None, a best-effort probe), so a refusal can end
# the update as a misleading downstream error; the command's failure path prints
# :func:`refusal_notice` instead.
_RUN: dict = {"ran": False, "refused": None}


def _refuse(argv: Sequence[str], cause: object, *, receipted: bool = False) -> CustodyRefused:
    """The :class:`CustodyRefused` for ``argv``: logged, noted in the receipt (unless the caller
    already did) and kept as this run's first refusal."""
    argv = list(argv) if isinstance(argv, (list, tuple)) else [argv]
    exc = CustodyRefused(argv, cause)
    detail = f"{exc.reason}: {' '.join(str(arg) for arg in argv[:2])}"
    if not receipted:
        logger.warning("Refused update child: %s", detail)
        with contextlib.suppress(Exception):
            from hermes_cli.update_receipt import record_step

            record_step("update_custody", False, detail)
    if _RUN["refused"] is None:
        _RUN["refused"] = (exc, not _RUN["ran"] and not _past_commit())
    return exc


def _past_commit() -> bool:
    """The update reached its commit point: its receipt records the ``apply`` stage (the
    completion child resumes that receipt). No receipt is open before an update begins, nor in
    the parent after its completion child returned, which starts no update child."""
    try:
        from hermes_cli.update_receipt import _current

        current = _current.get()
    except Exception:  # health: allow BLE001 -- fail closed: unknown = past commit, never "nothing changed"
        return True
    return current is not None and any(stage.get("name") == "apply" for stage in current.data.get("stages") or ())


def refusal_notice(command: str = "hermes update") -> str | None:
    """What the command's failure path prints when this run refused an update child (m2), in
    place of whatever generic error the refusal turned into downstream; ``None`` otherwise."""
    if _RUN["refused"] is None:
        return None
    exc, unchanged = _RUN["refused"]
    outcome = ("Nothing was changed: no update step had run yet." if unchanged else
               "Update steps before it had already run.")
    return f"✗ `{command}` stopped: {exc.reason}\n  {outcome}\n  {_RETRY.format(command=f'`{command}`')}"


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
    except Exception:  # health: allow BLE001 -- a torn tree's half-written module raises anything; no updater runs
        return None
    return update_lock._HELD


def _death_signal_preexec():
    """Linux: a ``preexec_fn`` making the child die (SIGKILL) with the updater thread that waits
    on it. For the git children that do NOT hold the lock fd (fetch, ls-remote, readers): a
    killed owner must not leave a fetch rewriting refs under the next lock owner. The ruling's
    stale-lock rules recover a fetch killed mid-write. ``None`` elsewhere (Windows: the job;
    macOS has no parent-death signal: :func:`run_git` puts a fetch under :func:`_owner_watch`)."""
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

    Suspended until bound, so nothing it spawns can escape the job. A refused bind kills the
    child before it ran a single instruction and raises :class:`CustodyRefused` (D2: never an
    unfenced writer); a failed resume kills the child."""
    from hermes_cli.update_lock import _bind_to_kill_on_close_job, resume_suspended_child

    try:
        _bind_to_kill_on_close_job(proc)
    except OSError as exc:
        proc.kill()
        proc.wait()
        raise _refuse(proc.args, exc) from exc
    resume_suspended_child(proc)
    _RUN["ran"] = True


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
            # git.exe's git-remote-https inherits the pipes: kill the tree, not git.exe alone (and
            # never the job: it also holds the update's other children).
            from hermes_cli._subprocess_compat import kill_and_drain

            drained = kill_and_drain(proc, _DRAIN_SECONDS)
            if drained is not None:
                exc.stdout, exc.stderr = drained
            raise
        except BaseException:
            proc.kill()
            raise
        code = proc.poll()
    if check and code:
        raise subprocess.CalledProcessError(code, proc.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, code, stdout, stderr)


def popen_post_commit(argv: Sequence[str], *, label: str, **kwargs) -> subprocess.Popen:
    """``subprocess.Popen`` for a checkout writer that runs after the update committed (the
    historical takeover's ``update_finish``). POSIX: the lock fd. Windows inside an update:
    created suspended and bound to the update's kill-on-close job before it runs. A refused bind
    must not fail a committed update, so the child still runs, never silently (printed and
    receipted) and never unfenced: it joins the checkout lock holding its own lease (R5b)."""
    if not (sys.platform == "win32" and _held() is not None):
        return subprocess.Popen(list(argv), **_custody_kwargs(True, kwargs))
    from hermes_cli.update_lock import bind_child_to_update_tree, resume_suspended_child

    kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_SUSPENDED
    proc = subprocess.Popen(list(argv), **kwargs)
    try:
        refusal = bind_child_to_update_tree(proc)
        if refusal is not None:
            detail = (f"the update's job would not take the {label} ({refusal}), so it runs outside "
                      "the job, holding its own checkout lease")
            print(f"  ⚠ {detail}", flush=True)
            with contextlib.suppress(Exception):
                from hermes_cli.update_receipt import record_step

                record_step("update_custody", False, detail)
        resume_suspended_child(proc)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    return proc


# How long a timed-out child's output may still drain after its tree was killed.
_DRAIN_SECONDS = 5


def run_git(git_cmd: Sequence[str], args: Sequence[str], **kwargs) -> subprocess.CompletedProcess:
    """THE updater git runner: custody config in argv, the lock fd only into local mutators,
    Windows job binding inside an update. ``kwargs`` are ``subprocess.run``'s."""
    argv = git_argv(git_cmd, args)
    mutator = is_local_mutator(argv[1:])
    if mutator:
        _prefetch_for_move(list(git_cmd), list(args), kwargs)
    elif git_subcommand(argv[1:]) in _FD_LESS_REF_WRITERS and sys.platform != "win32" \
            and _held() is not None and "preexec_fn" not in kwargs and _death_signal_preexec() is None:
        return _run_owner_watched(argv, kwargs)
    return run(argv, inherit_lock=mutator, **kwargs)


# git commands that write refs WITHOUT the lock fd (m3: their network half must not hold it). On
# Linux the parent-death signal ends one whose owner died; where there is none (macOS) a watchdog
# does (R3), or it would go on rewriting refs and leave live `*.lock` files under the next owner.
_FD_LESS_REF_WRITERS = frozenset({"fetch"})

# The watchdog: stdin is a pipe only the owner writes. A byte = the child finished; EOF with no
# byte = the owner died (the kernel closed its end), so SIGKILL the child, as PR_SET_PDEATHSIG
# would. It holds no lock fd and runs in its own session (a terminal's ^C/hang-up is the owner's).
_OWNER_WATCH = (
    "import os, signal, sys\n"
    "if not os.read(0, 1):\n"
    "    try:\n"
    "        os.kill(int(sys.argv[1]), signal.SIGKILL)\n"  # windows-footgun: ok -- POSIX only (run_git gates win32 out)
    "    except OSError:\n"
    "        pass\n"
)


def _owner_watch(pid: int):
    """Start the parent-death stand-in for child ``pid``; returns the callable that dismisses it
    once the child is done, or ``None`` when it could not start (logged: the fetch still runs, as
    it did before; only the owner's death mid-fetch is then unfenced)."""
    import os

    read_end, write_end = os.pipe()  # non-inheritable: no other child keeps the owner's end open
    try:
        watcher = subprocess.Popen([sys.executable, "-I", "-S", "-c", _OWNER_WATCH, str(pid)], stdin=read_end,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError as exc:
        os.close(write_end)
        logger.warning("Could not watch update child %s for its owner's death: %s", pid, exc)
        return None
    finally:
        os.close(read_end)

    def dismiss() -> None:
        with contextlib.suppress(OSError):  # the watcher already gone: nothing left to tell it
            os.write(write_end, b"x")
        os.close(write_end)
        with contextlib.suppress(subprocess.TimeoutExpired):  # it exits on the byte; never block on it
            watcher.wait(timeout=5)

    return dismiss


def _run_owner_watched(argv: Sequence[str], kwargs: dict) -> subprocess.CompletedProcess:
    """``subprocess.run`` for an fd-less ref writer, under :func:`_owner_watch` while it runs."""
    input, timeout = kwargs.pop("input", None), kwargs.pop("timeout", None)
    check = kwargs.pop("check", False)
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    if kwargs.pop("capture_output", False):
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    with subprocess.Popen(list(argv), **kwargs) as proc:
        dismiss = _owner_watch(proc.pid)
        try:
            stdout, stderr = proc.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            try:
                exc.stdout, exc.stderr = proc.communicate(timeout=_DRAIN_SECONDS)
            except subprocess.TimeoutExpired:
                # A helper git started still holds the pipes: leave them to the reader threads.
                proc.stdin = proc.stdout = proc.stderr = None
            raise
        except BaseException:
            proc.kill()
            raise
        finally:
            if dismiss is not None:
                dismiss()
        code = proc.poll()
    if check and code:
        raise subprocess.CalledProcessError(code, proc.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(proc.args, code, stdout, stderr)


def _partial_clone(git_cmd: Sequence[str], kwargs: dict) -> bool:
    """The repository has a promisor remote (``extensions.partialClone``), read from disk."""
    from pathlib import Path

    from hermes_cli.update_lock import _git_common_dir

    cwd = kwargs.get("cwd")
    for flag, value in zip(git_cmd, git_cmd[1:]):
        if flag == "-C":
            cwd = Path(cwd or ".") / value
    common = _git_common_dir(Path(cwd or "."))
    try:
        return common is not None and "partialclone" in \
            (common / "config").read_text(encoding="utf-8-sig", errors="replace").lower()
    except OSError:
        return False


def _prefetch_for_move(git_cmd: list[str], args: list[str], kwargs: dict) -> None:
    """Partial clones: fetch what a move to another commit will read — the target's changed trees
    and blobs — WITHOUT the lock fd and with the user's credential helpers, so the mutator (which
    has neither helper nor reason to fetch) never starts a lazy fetch or a credential daemon under
    the fd (m3). ``git diff HEAD <target>`` reads exactly the trees and blobs that differ from the
    checked-out tree; the rest is local. Best effort: a miss only leaves a helper-less lazy fetch."""
    sub = git_subcommand(args)
    if sub not in _MOVES or not _partial_clone(git_cmd, kwargs):
        return
    common = {key: kwargs[key] for key in ("cwd", "env") if key in kwargs}

    def resolve(*query: str, stdin: str | None = None) -> str | None:
        out = run(git_argv(git_cmd, list(query)), capture_output=True, text=True, encoding="utf-8",
                  errors="replace", input=stdin, **({} if stdin is not None else {"stdin": subprocess.DEVNULL}),
                  **common)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None

    try:
        targets = []
        for arg in args[args.index(sub) + 1:]:
            if arg == "--":
                break
            if not arg.startswith("-"):
                targets.append(resolve("rev-parse", "--verify", "-q", f"{arg}^{{commit}}"))
        targets = [target for target in dict.fromkeys(targets) if target]
        if not targets:
            return
        # An unborn HEAD (a fresh clone) has nothing checked out: diff from the empty tree.
        base = resolve("rev-parse", "--verify", "-q", "HEAD^{commit}") \
            or resolve("hash-object", "-t", "tree", "--stdin", stdin="")
        for target in targets:
            if base and target != base:
                run(git_argv(git_cmd, ["diff", "--binary", "--no-ext-diff", "--no-textconv", base, target]),
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **common)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        logger.debug("Could not prefetch the objects for git %s: %s", sub, exc)


# A stdlib launcher for children whose Popen the updater never sees (``run_contained``): it
# joins the job named by an inherited handle, starts the real command suspended with the same
# stdio, runs it only once it is in the job, and drops that handle (only the owner's handle may
# keep the job open). A refused join, or a command outside the job, never runs (D2): the launcher says so on stderr, writes the notice to the report file
# (argv[2]) — run_contained captures the child's stderr, so stderr alone is not seen (m1) — and
# exits; the updater logs the notice, notes it in the receipt and raises CustodyRefused.
_CUSTODY_UNAVAILABLE = "hermes: update custody unavailable"
_REFUSED_EXIT = 87
_JOIN_JOB = (
    "import ctypes, subprocess, sys\n"
    "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
    "k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]\n"
    "k.GetCurrentProcess.restype = ctypes.c_void_p\n"
    "k.CloseHandle.argtypes = [ctypes.c_void_p]\n"
    "h = ctypes.c_void_p(int(sys.argv[1]))\n"
    "def refuse(why):\n"
    f"    note = '{_CUSTODY_UNAVAILABLE} (%s); the command was not run' % why\n"
    "    sys.stderr.write(note + '\\n')\n"
    "    sys.stderr.flush()\n"
    "    try:\n"
    "        with open(sys.argv[2], 'w', encoding='utf-8') as report:\n"
    "            report.write(note)\n"
    "    except OSError:\n"
    "        pass\n"
    f"    sys.exit({_REFUSED_EXIT})\n"
    "if not k.AssignProcessToJobObject(h, k.GetCurrentProcess()):\n"
    "    refuse('could not join the update job: %d' % ctypes.get_last_error())\n"
    # Verify, never assume the topology: a redirecting interpreter (a Store Python alias) can
    # join while what it starts lands outside the job. Start the command suspended and run it
    # only once Windows says it is in the job (F54).
    "k.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]\n"
    "p = subprocess.Popen(sys.argv[3:], stdin=subprocess.DEVNULL, creationflags=0x4)\n"
    "inside = ctypes.c_int(0)\n"
    "if not k.IsProcessInJob(ctypes.c_void_p(int(p._handle)), h, ctypes.byref(inside)) or not inside.value:\n"
    "    p.kill()\n"
    "    p.wait()\n"
    "    refuse('the command would start outside the update job')\n"
    "k.CloseHandle(h)\n"
    "nt = ctypes.WinDLL('ntdll')\n"
    "nt.NtResumeProcess.argtypes = [ctypes.c_void_p]\n"
    "if nt.NtResumeProcess(ctypes.c_void_p(int(p._handle))) != 0:\n"
    "    p.kill()\n"
    "sys.exit(p.wait())\n"
)


def _report_refused_join(report: str, argv: Sequence[str]) -> str | None:
    """Make a refused job join visible: a warning (errors.log) and a receipt step. The notice,
    or None when the child joined (or never ran the launcher)."""
    import os

    try:
        with open(report, encoding="utf-8-sig") as fh:
            note = fh.read().strip()
    except OSError:
        note = ""
    finally:
        with contextlib.suppress(OSError):
            os.unlink(report)
    if not note:
        return None
    logger.warning("%s: %s", note, " ".join(str(arg) for arg in argv[:2]))
    with contextlib.suppress(Exception):
        from hermes_cli.update_receipt import record_step

        record_step("update_custody", False, f"{note}: {' '.join(str(arg) for arg in argv[:2])}")
    return note


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


# POSIX twin of the launcher (N13, review L1/L6). Node marks the donated lock fd close-on-exec,
# so nothing node starts (npm, esbuild, sh) keeps the checkout locked: custody of them rests on
# this stdlib parent, which holds the fd until no descendant of the command is left.
# * The command stays in the CALLER's process group: every group kill that stops a build (a
#   Ctrl-C'd completion child's ``killpg``, Desktop's ``kill(-pid)``) reaches node and everything
#   under it that did not start a session of its own. The custodian leaves that group (E): a
#   SIGKILL of the group would otherwise kill it too, and a descendant in a session of its own
#   would go on writing with the checkout lock free. It outlives the kill, settles the tree as
#   below, and only then exits (releasing the fd).
# * Linux: the custodian is a child subreaper, so a descendant orphaned by node's exit (or by any
#   intermediate's) is re-parented to it, never to init. Once node exits it SIGKILLs and reaps
#   its children until none is left; it only signals its own unreaped children, so no pid it
#   kills can have been reused (never a ``killpg`` of an already reaped leader's group).
# * Elsewhere (macOS has no subreaper) it records node's descendant tree from ``ps`` while node
#   runs and, once node exits, kills every recorded process whose start time still matches,
#   plus what they started. Best effort: a process born and orphaned between two samples escapes.
# * SIGINT/SIGTERM/SIGHUP are forwarded to node (a node still running 10 s later is killed); the
#   descendants are then killed as above.
# * The custodian is NOT the process the runner sees (R4): ``subprocess.run`` SIGKILLs its own
#   child when Ctrl-C reaches the caller, and no group separation survives a direct kill. The
#   runner's child only forks the custodian (own group) and relays signals to it and its exit
#   status back; it leaves the caller's group too, or a terminal's Ctrl-C would reach node twice. Killed, it leaves the custodian to settle: node gets the same 10 s it gets after
#   a forwarded signal (it is usually still cleaning up from the same Ctrl-C), then the tree is
#   killed as above and only then is the fd dropped.
_REAP_TREE = r"""
import os, signal, subprocess, sys, time
KILL = signal.SIGKILL  # windows-footgun: ok - POSIX-only launcher
fds = tuple(int(fd) for fd in sys.argv[1].split(','))
caller = os.getpgrp()
if caller != os.getpid():
    os.setpgid(0, 0)
stand_in = os.getpid()
custodian = os.fork()
if custodian:
    def relay(signum, frame):
        try:
            os.kill(custodian, signum)
        except OSError:
            pass
    for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
        signal.signal(getattr(signal, name), relay)
    code = os.waitstatus_to_exitcode(os.waitpid(custodian, 0)[1])
    sys.exit(code if code >= 0 else 128 - code)
os.setpgid(0, 0)
me = os.getpid()
reaper = False
if sys.platform.startswith('linux'):
    try:
        import ctypes
        reaper = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        reaper = False

def children():
    found = []
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            with open('/proc/%s/stat' % entry, 'rb') as fh:
                stat = fh.read()
            if int(stat[stat.rindex(b')') + 2:].split()[1]) == me:
                found.append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    return found

def table():
    try:
        out = subprocess.run(['ps', '-A', '-o', 'pid=,ppid=,stat=,lstart='], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, errors='replace', timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and not parts[2].startswith('Z'):
            rows[int(parts[0])] = (int(parts[1]), parts[3].strip())
    return rows

seen = {}

def sample(root):
    rows = table()
    tree = {pid for pid, start in seen.items() if rows.get(pid, (0, None))[1] == start}
    if root is not None:
        tree.add(root)
    grew = True
    while grew:
        grew = False
        for pid, (ppid, start) in rows.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                seen[pid] = start
                grew = True
    return [pid for pid, start in seen.items() if rows.get(pid, (0, None))[1] == start]

def kill(pids):
    for pid in pids:
        try:
            os.kill(pid, KILL)
        except OSError:
            pass

p = subprocess.Popen(sys.argv[2:], pass_fds=fds, process_group=caller)
got = []

def forward(signum, frame):
    got.append(signum)
    try:
        p.send_signal(signum)
    except OSError:
        pass

for name in ('SIGINT', 'SIGTERM', 'SIGHUP'):
    signal.signal(getattr(signal, name), forward)
deadline = None
while True:
    try:
        code = p.wait(timeout=0.5)
        break
    except subprocess.TimeoutExpired:
        pass
    if not reaper:
        sample(p.pid)
    if got or os.getppid() != stand_in:
        deadline = deadline or time.monotonic() + 10
        if time.monotonic() > deadline:
            p.kill()
end = time.monotonic() + 10
while time.monotonic() < end:
    live = children() if reaper else sample(None)
    if not live:
        break
    kill(live)
    while reaper:
        try:
            if not os.waitpid(-1, os.WNOHANG)[0]:
                break
        except ChildProcessError:
            break
    time.sleep(0.02)
sys.exit(code if code >= 0 else 128 - code)
"""


@contextlib.contextmanager
def contained_command(argv: Sequence[str], *, inherit_lock: bool = True, root=None):
    """``(argv, kwargs)`` for a checkout writer started by a runner that hides its Popen (the Node
    build in ``pm.progress.run_contained``). POSIX: the lock fd (this process's, or one it
    inherited for checkout ``root``), under a launcher that keeps the command in the caller's
    process group and kills every descendant left when it exits (:data:`_REAP_TREE`).
    Windows inside an update: the command runs under a launcher that joins the update's
    kill-on-close job first; when the job cannot be handed over or the join is refused, the
    command never runs and :class:`CustodyRefused` is raised (D2)."""
    argv = list(argv)
    if not (sys.platform == "win32" and _held() is not None):
        from hermes_cli.update_lock import checkout_lock_fds

        fds = tuple(checkout_lock_fds(root)) if inherit_lock and root is not None else _lock_fds(inherit_lock)
        if fds:
            argv = [sys.executable, "-I", "-S", "-c", _REAP_TREE, ",".join(map(str, fds)), *argv]
        yield argv, ({"pass_fds": fds} if fds else {})
        return
    try:
        handle = _inheritable_job_handle()
    except OSError as exc:
        raise _refuse(argv, exc) from exc
    import os
    import tempfile

    fd, report = tempfile.mkstemp(prefix="hermes-custody-", suffix=".txt")
    os.close(fd)
    try:
        info = subprocess.STARTUPINFO()
        info.lpAttributeList = {"handle_list": [handle]}
        yield ([_join_launcher_python(), "-I", "-S", "-c", _JOIN_JOB, str(handle), report, *argv],
               {"startupinfo": info})
    finally:
        import ctypes

        ctypes.WinDLL("kernel32").CloseHandle(ctypes.c_void_p(handle))
        note = _report_refused_join(report, argv)
        if note:
            cause = note.removeprefix(f"{_CUSTODY_UNAVAILABLE} (").partition(")")[0]
            raise _refuse(argv, cause, receipted=True)
        _RUN["ran"] = True


def _inheritable_job_handle() -> int:
    """An inheritable duplicate of the update's kill-on-close job handle (Windows). Raises
    ``OSError`` when the job cannot be created or duplicated."""
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
        raise
