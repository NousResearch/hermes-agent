"""Child side of run_tests_parallel.py's fork server (HERMES_TEST_FORKSERVER=1, Linux).

A fresh ``python -m pytest <file>`` spends about half a second starting the
interpreter and importing pytest and the third-party libraries every test file
loads. The runner's fork server pays that once: it imports those modules, then
forks one child per test file. Each child gets the state a fresh process
would have: its own session and process group, the per-file environment, the
``python -m pytest`` sys.path and argv, stdout and stderr in one file, and the
exit code pytest returns. Product code and tests/conftest.py are never
preloaded, so every file still imports them fresh.

This module must stay importable on its own (the fork server unpickles ``run``
by name) and import nothing from the product.
"""

from __future__ import annotations

import atexit
import os
import sys
import tempfile


def run(
    file: str,
    pytest_args: list[str],
    repo_root: str,
    env: dict[str, str],
    out_path: str,
    sys_path: list[str],
    argv0: str,
) -> None:
    # First, before anything can spawn: a session of our own, so the runner's
    # killpg(pid) takes out every process this file starts and nothing else.
    os.setsid()
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    os.chdir(repo_root)
    os.environ.clear()
    os.environ.update(env)
    # gettempdir() caches its answer; this file's TMPDIR must win.
    tempfile.tempdir = None
    sys.path[:] = sys_path
    sys.argv[:] = [argv0, file, *pytest_args]

    import pytest

    try:
        rc = pytest.main([file, *pytest_args])
    finally:
        # A real interpreter exit runs these; multiprocessing's child exit does not.
        atexit._run_exitfuncs()
        sys.stdout.flush()
        sys.stderr.flush()
    sys.exit(int(rc))
