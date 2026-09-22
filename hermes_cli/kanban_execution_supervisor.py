"""Linux native worker owner: adopt/reap detached children before settling a run.

The supervisor stays alive through cleanup. SIGKILL/loss without its durable
receipt leaves capacity occupied; a process scan cannot manufacture that proof.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def _subreaper():
    if sys.platform != 'linux':
        raise RuntimeError('execution supervision requires Linux child subreaping')
    libc = ctypes.CDLL(None, use_errno=True)
    enabled = ctypes.c_int()
    if libc.prctl(36, 1, 0, 0, 0) or libc.prctl(37, ctypes.byref(enabled), 0, 0, 0) or enabled.value != 1:
        raise RuntimeError('kernel child subreaping unavailable')


def _reap():
    """ECHILD is proof; WNOHANG=0 means at least one live child still exists."""
    exited = {}
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return exited, True
        if pid == 0:
            return exited, False
        exited[pid] = os.waitstatus_to_exitcode(status)


def _signal_children(sig):
    # Single-threaded owner: no reaping between this kernel child list and
    # signalling. Exited children remain zombies, so these PIDs cannot be reused.
    path = Path(f'/proc/self/task/{os.getpid()}/children')
    for value in path.read_text().split():
        try:
            os.kill(int(value), sig)
        except ProcessLookupError:
            pass


def supervise(db_path, task_id, run_id, claim_lock, scope_id, argv):
    from hermes_cli import kanban_db_connect as kbc, kanban_execution_scope as scopes
    from hermes_cli.kanban_db_dispatch import _process_fingerprint
    _subreaper()
    stopping = False
    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    pid = os.getpid()
    fingerprint = _process_fingerprint(pid)
    with kbc.connect_closing(Path(db_path)) as conn:
        scope = scopes.activate(conn, task_id, run_id, claim_lock, scope_id, pid, fingerprint)
        deadline = scope['deadline']
        monotonic_cutoff = None if deadline is None else time.monotonic()+max(0, deadline-time.time())
        def expired():
            return deadline is not None and (time.time() >= deadline or time.monotonic() >= monotonic_cutoff)
        proc, root_code, stopping_at = None, None, None
        reason = 'root_exit'
        try:
            # The immutable cutoff is checked at the actual process boundary,
            # after startup/DB waits, not only when the dispatcher planned it.
            if stopping or expired():
                reason = 'stopped_before_launch' if stopping else 'deadline_before_launch'
            else:
                proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL)
                while True:
                    exits, empty = _reap()
                    if proc.pid in exits:
                        root_code = exits[proc.pid]
                        proc.returncode = root_code
                    if empty:
                        break
                    if expired():
                        reason, stopping = 'deadline', True
                    if stopping or root_code is not None:
                        if stopping_at is None:
                            stopping_at = time.monotonic()
                            if stopping and reason == 'root_exit':
                                reason = 'signal'
                        _signal_children(signal.SIGKILL if time.monotonic()-stopping_at >= 2 else signal.SIGTERM)
                    time.sleep(.02)
            # Even failed Popen/prelaunch must establish absence, not assume it.
            _, empty = _reap()
            if not empty:
                raise RuntimeError('execution still owns children')
            scopes.finish(conn, run_id, scope_id, pid, fingerprint, reason=reason, returncode=root_code)
            return root_code if root_code is not None and root_code >= 0 else 1
        finally:
            # Unexpected failures must not publish cleanup. Attempt to kill
            # owned children; a lost/failed supervisor remains unresolved.
            _, empty = _reap()
            if not empty:
                _signal_children(signal.SIGKILL)


if __name__ == '__main__':
    # Script execution works from an arbitrary native workspace without relying
    # on PYTHONPATH or a globally installed copy of Hermes.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    db_path, task_id, run_id, claim_lock, scope_id, *command = sys.argv[1:]
    if not command:
        raise SystemExit('native command is required')
    raise SystemExit(supervise(db_path, task_id, int(run_id), claim_lock, scope_id, command))
