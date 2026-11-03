"""The hourly scratch prune runs in a detached child process, not in the process that is due it.

A pass can take seconds to minutes (the idle scan of a large tree, the ``rmtree`` of a multi-GB
entry). Inline, it held every process start; on a daemon thread it died with every short-lived
``hermes`` command, the hourly stamp already spent. A detached child outlives the command that
launched it, so each due pass runs to the end, with the same rules as an inline one.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import hermes_constants_scratch as rules

logger = logging.getLogger(__name__)

# ``-I``: neither the launcher's cwd (a project ``utils/`` would shadow ours) nor PYTHON* env vars
# decide what the child imports; it gets the launcher's own ``sys.path`` (repo root, PM/venv
# site-packages), not ``hermes_bootstrap``, whose boot hook would schedule another prune.
_CHILD = ("import sys; sys.path[:] = sys.argv[3:]; "
          "import hermes_constants_scratch_detach as d; d.run(*sys.argv[1:3])")


def spawn_prune(scratch: Path) -> None:
    """Start one detached prune of *scratch*, the path the caller resolved (a child never
    re-resolves the profile home). A failed spawn is logged and dropped: the stamp is already
    spent, so the next pass is the next process past the hourly mark.

    This process's ancestry is captured here, before the spawn: once this command and its
    parents exit, the child can no longer see that the user's shell was the caller's."""
    argv = [sys.executable, "-I"]
    if sys.dont_write_bytecode:
        argv.append("-B")
    if sys.pycache_prefix:  # a sealed payload must never get __pycache__ writes
        argv += ["-X", f"pycache_prefix={sys.pycache_prefix}"]
    argv += ["-c", _CHILD, os.fspath(scratch), _encode_lineage(rules.lineage_with_start_times()), *sys.path]
    from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway, windows_detach_popen_kwargs

    stdio = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        try:
            subprocess.Popen(argv, close_fds=True, **stdio, **windows_detach_popen_kwargs())
        except PermissionError:
            if sys.platform != "win32":
                raise
            # A job object without BREAKAWAY_OK rejects CREATE_BREAKAWAY_FROM_JOB.
            subprocess.Popen(argv, close_fds=True, **stdio, creationflags=windows_detach_flags_without_breakaway())
    except (OSError, RuntimeError) as exc:
        logger.debug("scratch prune: could not start the detached pass for %s: %s", scratch, exc)


def _encode_lineage(chain: list[tuple[int, float]] | None) -> str:
    # repr() round-trips a float exactly; "-" is "could not compute" (the child fails closed).
    return "-" if chain is None else ",".join(f"{pid}:{started!r}" for pid, started in chain)


def run(scratch: str, lineage: str) -> None:
    """Child entry: one prune of *scratch*, sparing the launcher's ancestry. Any failure is one
    line in ``scratch-prune.log``; nothing is raised (no one reads this process's stderr)."""
    try:
        from hermes_constants import prune_scratch_dir

        rules._launcher_lineage = None if lineage == "-" else frozenset(
            (int(pid), float(started)) for pid, started in (item.split(":") for item in lineage.split(",")))
        prune_scratch_dir(Path(scratch))
    # health: allow BLE001 -- the detached child's top level: its stderr is /dev/null, so the audit log is the record
    except Exception as exc:
        audit = rules._open_prune_log(Path(scratch).parent.parent / "logs" / "scratch-prune.log")
        audit.info("scratch prune: detached pass failed (%s: %r)", type(exc).__name__, str(exc))
        if audit is not rules.logger:
            for handler in audit.handlers:
                handler.close()
