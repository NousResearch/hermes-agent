"""Start one test file's pytest process for run_tests_parallel.py.

``start_pytest`` returns the process and the process group the runner kills
on a timeout. The process has the subprocess.Popen surface the runner uses:
``pid``, ``returncode``, ``communicate(timeout)`` and ``kill()``.

* By default it is a fresh ``python -m pytest <file>`` in a session of its own.
* With HERMES_TEST_FORKSERVER=1 on Linux it is a child of one multiprocessing
  fork server, which imports pytest and the common third-party modules once
  for the whole run (see run_tests_fork_child for what each child resets).

Stdlib only, like the runner: it runs before any test environment exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, NamedTuple

# Third-party modules nearly every test file imports. The fork server imports
# them once; product code and tests/conftest.py stay out so each file imports
# them fresh. pytest plugins (anyio, pytest_asyncio) stay out too: pytest must
# import them itself to rewrite their asserts.
_FORK_PRELOAD = (
    "pytest", "_pytest.python", "_pytest.fixtures", "_pytest.assertion.rewrite",
    "asyncio", "email.message", "importlib.metadata", "psutil", "pygments",
    "pydantic", "httpx", "urllib3", "rich", "ruamel.yaml",
)
# Process.start() is not thread-safe, and it also polls every live child
# (multiprocessing.process._cleanup). A child's exit status is 8 bytes the fork
# server writes once: if that poll and the owning thread's join() read it at the
# same time, the loser reads EOF and multiprocessing reports exit status 255.
# Starting children and reading their statuses both hold this lock.
_fork_lock = threading.Lock()


class _ForkServer(NamedTuple):
    """The run's fork-server context plus what each child sets up (run_tests_fork_child)."""

    ctx: Any
    sys_path: list[str]
    argv0: str


_fork_server: _ForkServer | None = None  # started by _fork_server_for_run() on first use
_fork_server_checked = False


def _start_fork_server(repo_root: Path) -> _ForkServer:
    """Linux only: one fork server for the whole run."""
    import multiprocessing

    ctx = multiprocessing.get_context("forkserver")
    # "__main__" (the runner) too: otherwise every child re-runs it as __mp_main__.
    ctx.set_forkserver_preload(["__main__", *_FORK_PRELOAD])
    # What ``python -m pytest`` would put on sys.path: the cwd first, then the
    # interpreter's own entries (never the runner's directory).
    base = json.loads(subprocess.run(
        [sys.executable, "-c", "import json, sys; print(json.dumps(sys.path[1:]))"],
        cwd=repo_root, capture_output=True, text=True, check=True, timeout=60,
    ).stdout)
    import pytest

    return _ForkServer(ctx, [str(repo_root), *base], str(Path(pytest.__file__).with_name("__main__.py")))


def _fork_server_for_run(repo_root: Path) -> _ForkServer | None:
    """The run's fork server when HERMES_TEST_FORKSERVER=1 on Linux, started by the first file.

    The first file calls this before it copies os.environ, so popping the knob
    here still keeps it out of every test's environment.
    """
    global _fork_server, _fork_server_checked
    with _fork_lock:
        if not _fork_server_checked:
            _fork_server_checked = True
            if os.environ.pop("HERMES_TEST_FORKSERVER", "") == "1" and sys.platform == "linux":
                _fork_server = _start_fork_server(repo_root)
                print("  fork server: on (third-party imports shared; each file still a fresh process)", flush=True)
        return _fork_server


def start_forked(proc) -> None:
    with _fork_lock:
        proc.start()


def forked_exit_status(proc, timeout: float | None) -> int | None:
    """Wait for a fork-server child, then read its status under _fork_lock (None on timeout)."""
    from multiprocessing.connection import wait

    if not wait([proc.sentinel], timeout):
        return None
    with _fork_lock:
        proc.join()
        return proc.exitcode


class _ForkedPytest:
    """The subprocess.Popen surface the runner uses, for a fork-server child."""

    def __init__(self, server: _ForkServer, argv: list[str], repo_root: Path, env: dict[str, str], scratch_dir: str):
        from scripts import run_tests_fork_child

        fd, self._out = tempfile.mkstemp(prefix="o-", dir=scratch_dir)
        os.close(fd)
        self._proc = server.ctx.Process(
            target=run_tests_fork_child.run,
            kwargs=dict(argv=argv, cwd=str(repo_root), env=env, sys_path=server.sys_path, out_path=self._out),
        )
        try:
            start_forked(self._proc)
        except BaseException:
            os.unlink(self._out)
            raise
        self.pid = self._proc.pid
        self.returncode: int | None = None

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        rc = forked_exit_status(self._proc, timeout)
        if rc is None:
            raise subprocess.TimeoutExpired(str(self._out), timeout or 0)
        self.returncode = rc
        try:
            with open(self._out, encoding="utf-8-sig", errors="replace") as fh:
                return fh.read(), None
        finally:
            os.unlink(self._out)

    def kill(self) -> None:
        import signal as _signal

        if self.returncode is not None:
            return  # reaped: the pid may already belong to another process
        try:
            os.kill(self.pid, _signal.SIGKILL)  # windows-footgun: ok (Linux only)
        except (ProcessLookupError, OSError):
            pass


def start_pytest(
    file: Path, pytest_args: list[str], repo_root: Path, env: dict[str, str], scratch_dir: str,
) -> tuple[Any, int | None]:
    """Start ``python -m pytest <file> <pytest_args>``; return it and its process group.

    The group is None on Windows, where the runner kills the tree another way.
    *scratch_dir* holds a forked child's output file until the runner reads it.
    """
    server = _fork_server_for_run(repo_root)
    if server is not None:
        proc = _ForkedPytest(server, [server.argv0, str(file), *pytest_args], repo_root, env, scratch_dir)
        # The child's first act is setsid(), so its pid is its group. Never ask
        # getpgid here: until setsid runs it reports the fork server's group,
        # which is the runner's own.
        return proc, proc.pid
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", str(file), *pytest_args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        env=env,
        # POSIX: place the child at the head of its own process group so
        # _kill_tree can SIGKILL the group atomically.
        # Windows: this maps to CREATE_NEW_PROCESS_GROUP in CPython 3.12+;
        # _kill_tree handles the Windows path via taskkill /F /T.
        start_new_session=True,
    )
    # Capture the pgid NOW, before the leader can exit and be reaped. Once
    # the leader is reaped, os.getpgid(proc.pid) raises ProcessLookupError
    # even though grandchildren in that group are still alive — defeating
    # the whole cleanup. None on Windows where the pgid concept doesn't apply.
    if sys.platform == "win32":
        return proc, None
    try:
        return proc, os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return proc, None
