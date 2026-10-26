"""Keep a checkout's commit-graph fresh: the ``hermes -w`` pass and every ``hermes update``.

Generation numbers in the commit-graph keep ``git tag --merged HEAD`` (the version lookup) and
ancestry checks off a full history walk: on hermes-agent a graph two months behind cost 0.33 s
per lookup, a fresh one 0.04 s. Partial clones have git's own graph writers switched off
(gitlock._TREE0_MAINTENANCE_OFF, #127711), so nothing else refreshes it there. Without Bloom data
the write reads commits only, never trees, so it cannot start that lazy-fetch loop, and it costs
~3 s instead of ~27 s when changed paths would be recomputed.

Also home of the per-clone maintenance slot and the bounded git runner the ``-w`` repack shares,
so the update path imports this module and not all of ``worktree_ops``.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

from hermes_cli._subprocess_compat import (
    IS_WINDOWS,
    NO_LAZY_FETCH_ENV,
    bounded_probe_run,
    kill_process_tree,
    noninteractive_git_env,
    windows_detach_flags_without_breakaway,
    windows_detach_popen_kwargs,
)

logger = logging.getLogger(__name__)

_WRITE = ("commit-graph", "write", "--reachable", "--split", "--no-changed-paths", "--no-progress")
_MAX_AGE = 24 * 3600
_TIMEOUT = 600
_MIN_INTERVAL = 6 * 3600
_LOCK = "hermes-commit-graph.lock"
# A write with no new commits leaves git's files untouched; this stamp records that it succeeded,
# so an idle repo is not rewritten on every slot.
_FRESH_STAMP = "hermes-commit-graph.fresh"
# git deletes its lock and tmp files on SIGTERM; a SIGKILLed writer strands commit-graph-chain.lock
# and tmp_graph_*, and that lock fails every later write until the stale-lock sweep removes it.
_KILL_GRACE = 5.0
_CHILD = ("import sys; sys.path.insert(0, sys.argv[1]); "
          "from hermes_cli.git_commit_graph import refresh_in_child; refresh_in_child(*sys.argv[2:])")


@contextmanager
def _locked_slot(lock: Path, *, wait: bool) -> Iterator[Optional[int]]:
    """The slot file's fd under an exclusive cross-process lock, or None while another process holds
    it (``wait=False``). The file is never unlinked: a waiter on a deleted inode would claim a slot
    nobody else can see. The kernel drops the lock with the process."""
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            if IS_WINDOWS:
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except OSError:
            yield None
            return
        try:
            yield fd
        finally:
            if IS_WINDOWS:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def claim_maintenance_slot(git_dir: Path, lock_name: str, interval: float) -> bool:
    """Exactly one process per clone gets to run a maintenance job per ``interval``.

    The slot file holds the claimant's pid and its mtime is the stamp: younger than the interval
    means another launch is running the job (or just tried and timed out) — skip. Check and claim
    happen under one file lock, so of N launches finding an expired or empty slot exactly one wins.
    """
    with _locked_slot(git_dir / lock_name, wait=False) as fd:
        if fd is None:
            return False
        st = os.fstat(fd)
        if st.st_size and time.time() - st.st_mtime < interval:
            return False
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()}\n".encode())
        return True


def release_maintenance_slot(git_dir: Path, lock_name: str) -> None:
    """Give back this process's claim (its job never started) so the next run retries; a claim
    another process made since is left alone."""
    with _locked_slot(git_dir / lock_name, wait=True) as fd:
        if fd is not None and os.read(fd, 32).strip() == str(os.getpid()).encode():
            os.ftruncate(fd, 0)


def run_bounded_git(repo_root: str, args: Sequence[str], timeout: float, env: Optional[dict] = None) -> Optional[int]:
    """Run a niced ``git *args`` whose whole process tree dies with the timeout or with us.

    Returns the exit code, or None when the timeout killed it; a failure is logged with git's last
    stderr line. ``repack`` forks ``pack-objects``; ``subprocess.run(timeout=)`` killed only the
    parent and left the grandchild packing for days, and a daemon thread's child outlived the CLI the
    same way. A new session/process group + ``atexit`` reaps both cases, SIGTERM first so git can
    delete its own lock and tmp files.
    """
    cmd = ["git", *args]
    if os.name == "posix":
        cmd = ["nice", "-n", "19", *cmd]
        group_kw: dict = {"process_group": 0}
    else:
        group_kw = {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, cwd=repo_root, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=err, env=env, **group_kw)

        def _reap() -> None:
            if proc.poll() is None:
                kill_process_tree(proc, grace=_KILL_GRACE)

        atexit.register(_reap)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _reap()
            logger.warning("git %s in %s exceeded %ss; killed", args[0], repo_root, timeout)
            return None
        finally:
            atexit.unregister(_reap)
        if rc:
            err.seek(0)
            lines = err.read()[-4096:].decode("utf-8", "replace").strip().splitlines()
            # git's "File exists" ends on advice; the fatal/error line names the cause.
            cause = next((ln for ln in reversed(lines) if ln.startswith(("fatal:", "error:"))), lines[-1] if lines else "no output")
            logger.warning("git %s in %s failed (exit %d): %s", args[0], repo_root, rc, cause)
        return rc


def commit_graph_stale(git_dir: Path) -> bool:
    """No commit-graph, or none written or confirmed current for a day."""
    info = git_dir / "objects" / "info"
    graphs = [p.stat().st_mtime for p in (info / "commit-graph", info / "commit-graphs" / "commit-graph-chain")
              if p.is_file()]
    if not graphs:
        return True
    stamp = git_dir / _FRESH_STAMP
    newest = max(graphs + ([stamp.stat().st_mtime] if stamp.is_file() else []))
    return time.time() - newest > _MAX_AGE


def claim_commit_graph_refresh(git_dir: Path) -> bool:
    return commit_graph_stale(git_dir) and claim_maintenance_slot(git_dir, _LOCK, _MIN_INTERVAL)


def _commit_graph_disabled(repo_root: str) -> bool:
    """``core.commitGraph=false`` in the repo, global or system config: never write a graph there.

    Read with the user's global/system config, which the writer's isolated env blanks."""
    env = noninteractive_git_env()
    for key in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_NOSYSTEM"):
        env.pop(key)
        if key in os.environ:
            env[key] = os.environ[key]
    probe = bounded_probe_run(["git", "config", "--type=bool", "--get", "core.commitGraph"],
                              timeout=10, cwd=repo_root, env=env)
    return probe is not None and probe.stdout.strip() == "false"


def common_git_dir(repo_root: str | Path) -> Optional[Path]:
    """The git dir holding the object store: ``.git`` itself, or the dir a linked worktree's or
    submodule's ``.git`` file leads to. None when ``repo_root`` is no checkout."""
    from hermes_cli.update_lock import _git_common_dir

    return _git_common_dir(Path(repo_root))


def refresh_commit_graph(repo_root: str, git_dir: Path) -> None:
    """Clear what a killed writer stranded, then write the graph (bounded, niced, never lazy-fetching).

    ``git_dir`` is the common git dir (:func:`common_git_dir`): objects, debris and stamp live there."""
    from hermes_cli.gitlock import clear_stale_object_store_locks, clear_stale_tmp_packs

    if _commit_graph_disabled(repo_root):
        logger.info("core.commitGraph=false in %s; not writing a commit-graph", repo_root)
        return
    # Object-store debris only: ``hermes -w`` runs this in whatever repo it was launched from, where
    # index.lock / HEAD.lock belong to the user's own git.
    clear_stale_object_store_locks(git_dir)
    clear_stale_tmp_packs(git_dir)
    logger.info("git commit-graph missing or stale — writing it in background")
    if run_bounded_git(repo_root, _WRITE, _TIMEOUT, env={**noninteractive_git_env(), **NO_LAZY_FETCH_ENV}) != 0:
        return
    # The stamp vouches for a graph on disk: a write another git undid (or whose files vanished)
    # exits 0 too, and a stamp then would hold the retry off for a day with no graph at all.
    if _graph_on_disk(git_dir):
        (git_dir / _FRESH_STAMP).touch()
    else:
        logger.info("git commit-graph write in %s exited 0 but left no graph (no commits yet?)", repo_root)


def _graph_on_disk(git_dir: Path) -> bool:
    """A split chain whose every layer file exists, or a monolithic graph."""
    info = git_dir / "objects" / "info"
    try:
        layers = (info / "commit-graphs" / "commit-graph-chain").read_text(encoding="ascii").split()
    except (OSError, UnicodeDecodeError):
        return (info / "commit-graph").is_file()
    return bool(layers) and all((info / "commit-graphs" / f"graph-{layer}.graph").is_file() for layer in layers)


class _LogDirOnFailureHandler(logging.FileHandler):
    """Opens (``delay``) only when there is a failure to record, creating ``logs/`` then: a home
    whose logs were never set up still gets the line; a deleted profile is not resurrected. A log
    that cannot be opened drops the line (``handleError``) instead of raising out of the logger."""

    def emit(self, record: logging.LogRecord) -> None:
        if self.stream is None:
            from hermes_constants import mkdir_under_hermes_home

            try:
                mkdir_under_hermes_home(Path(self.baseFilename).parent)
                self.stream = self._open()
            except OSError:  # a missing/deleted named profile is FileNotFoundError
                self.handleError(record)
                return
        super().emit(record)


def refresh_in_child(repo_root: str, git_dir: str, log_file: str) -> None:
    """Body of the detached child: a failure is appended to the home's agent.log; never raises.

    A plain handler, not ``setup_logging``: that reads config.yaml and seeds the home's layout,
    side effects a git maintenance child has no business having in a profile mid-update.
    """
    handler = _LogDirOnFailureHandler(log_file, encoding="utf-8", delay=True)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        refresh_commit_graph(repo_root, Path(git_dir))
    except Exception:
        logger.warning("Background commit-graph refresh of %s failed", repo_root, exc_info=True)


def spawn_commit_graph_refresh(repo_root: Path) -> bool:
    """Start the bounded commit-graph refresh in a detached child when it is due; never waits.

    ``hermes update`` is the one regular git pass an install checkout gets (``hermes -w`` is
    rarely launched from it), so without this its version lookups never get a graph. Detached
    because the write takes seconds and must neither delay the update nor die with it. A spawn that
    fails gives the slot back, so the next update retries instead of waiting out the interval.
    """
    from hermes_constants import get_hermes_home

    git_dir = common_git_dir(repo_root)
    if git_dir is None or not (git_dir / "objects").is_dir() or not claim_commit_graph_refresh(git_dir):
        return False
    code_root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, "-c", _CHILD, str(code_root), str(repo_root), str(git_dir),
            str(get_hermes_home() / "logs" / "agent.log")]
    kwargs: dict = dict(cwd=str(repo_root), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, close_fds=True)
    try:
        try:
            subprocess.Popen(argv, **kwargs, **windows_detach_popen_kwargs())
        except OSError:
            if not IS_WINDOWS:
                raise
            # A job object without BREAKAWAY_OK refuses CREATE_BREAKAWAY_FROM_JOB; retry without it.
            subprocess.Popen(argv, **kwargs, creationflags=windows_detach_flags_without_breakaway())
    except OSError:
        release_maintenance_slot(git_dir, _LOCK)
        raise
    return True
