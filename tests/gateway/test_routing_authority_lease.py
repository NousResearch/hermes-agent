from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import signal
import time

import pytest

_START_METHODS = ["spawn"] + (
    ["fork"] if "fork" in multiprocessing.get_all_start_methods() else []
)


def _passive_fork_owner(path: str, queue) -> None:
    from gateway.session import _acquire_routing_authority_lease

    _acquire_routing_authority_lease(Path(path))
    child_pid = os.fork()  # windows-footgun: ok - test is skip-gated on os.fork
    if child_pid == 0:
        time.sleep(10)
        os._exit(0)
    queue.put(child_pid)
    queue.close()
    queue.join_thread()


def _try_acquire(path: str, queue) -> None:
    from gateway.session import _acquire_routing_authority_lease

    try:
        _acquire_routing_authority_lease(Path(path))
    except RuntimeError:
        queue.put("blocked")
    else:
        queue.put("acquired")


@pytest.mark.parametrize("start_method", _START_METHODS)
def test_routing_authority_lease_fails_closed_across_processes(
    tmp_path, start_method
):
    from gateway.session import _acquire_routing_authority_lease

    sessions_dir = tmp_path / "sessions"
    _acquire_routing_authority_lease(sessions_dir)

    context = multiprocessing.get_context(start_method)
    queue = context.Queue()
    process = context.Process(target=_try_acquire, args=(str(sessions_dir), queue))
    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    assert queue.get(timeout=1) == "blocked"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork only")
def test_passive_fork_child_does_not_retain_routing_lease(tmp_path):
    sessions_dir = tmp_path / "sessions"
    context = multiprocessing.get_context("spawn")
    owner_queue = context.Queue()
    owner = context.Process(
        target=_passive_fork_owner,
        args=(str(sessions_dir), owner_queue),
    )
    owner.start()
    passive_child_pid = owner_queue.get(timeout=10)
    owner.join(timeout=10)
    assert owner.exitcode == 0

    contender_queue = context.Queue()
    contender = context.Process(
        target=_try_acquire,
        args=(str(sessions_dir), contender_queue),
    )
    contender.start()
    contender.join(timeout=10)
    try:
        assert contender.exitcode == 0
        assert contender_queue.get(timeout=1) == "acquired"
    finally:
        try:
            os.kill(passive_child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
