"""Superseded and aborted PM runtime generations are collected; running workers are not."""
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from pm.runtime import collect_runtime_generations

# The lease protocol differs per host (staged rename on POSIX, in-place lock on
# Windows), so these run on every OS lane instead of only the Linux suite.
_ALL_HOSTS = pytest.mark.platforms("linux", "macos", "windows")


def _generation(root: Path, name: str, *, published: bool = True, leased: bool = True) -> Path:
    generation = root / "generations" / name
    generation.mkdir(parents=True)
    if leased:
        (generation / ".lease-managed").touch()
    if published:
        (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}), encoding="utf-8")
    return generation


def _hold_lock(path: Path, ready, done) -> None:
    """Child-process body: hold a lease lock until the parent releases us."""
    from pm.filesystem import lock_fd

    fd = os.open(path, os.O_RDWR)
    try:
        assert lock_fd(fd, wait=True)
        ready.set()
        done.wait(30)
    finally:
        os.close(fd)


@_ALL_HOSTS
def test_collector_keeps_selected_leased_and_pre_lease_generations(tmp_path):
    from hermes_cli.runtime_state import lease_directory

    root = tmp_path / "pm-runtime"
    selected = _generation(root, "selected")
    busy = _generation(root, "busy")
    idle = _generation(root, "idle")
    legacy = _generation(root, "legacy", leased=False)
    aborted = _generation(root, "aborted", published=False)
    (root / "selected.json").write_text(json.dumps({"generation": "generations/selected"}), encoding="utf-8")
    release = lease_directory(busy)
    lease_directory(idle)()

    removed = collect_runtime_generations(root)

    assert set(removed) == {idle, aborted}
    assert selected.is_dir() and busy.is_dir() and legacy.is_dir()
    release()
    assert collect_runtime_generations(root) == [busy]


@_ALL_HOSTS
def test_collector_yields_to_an_in_flight_stage(tmp_path):
    from pm.filesystem import lock_fd

    root = tmp_path / "pm-runtime"
    aborted = _generation(root, "aborted", published=False)
    root.mkdir(exist_ok=True)
    with (root / ".prepare.lock").open("a+b") as lock:
        assert lock_fd(lock.fileno(), wait=False)
        assert collect_runtime_generations(root) == []
    assert aborted.is_dir()
    assert collect_runtime_generations(root) == [aborted]


@_ALL_HOSTS
def test_new_lease_sweeps_files_left_by_dead_holders(tmp_path):
    """A holder exiting via os.execv / os._exit skips atexit and leaves its lease file
    behind (#125609); the next lease taken in the generation removes the backlog."""
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    dead = generation / ".leases" / "dead-holder"
    dead.parent.mkdir(parents=True)
    dead.write_bytes(b"")

    release = lease_directory(generation)

    assert not dead.exists()
    assert len(list((generation / ".leases").glob("*"))) == 1  # only this process's lease
    release()


@_ALL_HOSTS
def test_new_lease_keeps_a_live_foreign_lease(tmp_path):
    """A foreign lease held by another live process is never swept.

    Held in a child process, not this one: Windows byte locks are per-process
    re-entrant, so a same-process holder would let the sweep's probe succeed and
    the test would pass for the wrong reason (the unlink failing on the handle).
    """
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    live = generation / ".leases" / "live-foreign"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"")

    ctx = multiprocessing.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(live, ready, done), daemon=True)
    holder.start()
    assert ready.wait(30), "holder child failed to take the lease lock"
    try:
        release = lease_directory(generation)
        assert live.exists()  # a held lease is never swept
        release()
    finally:
        done.set()
        holder.join(30)
        if holder.is_alive():
            holder.terminate()
            holder.join(30)


@_ALL_HOSTS
def test_release_is_idempotent(tmp_path):
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    release = lease_directory(generation)
    lease_file = next((generation / ".leases").glob("*"))

    release()
    release()  # an explicit call after atexit must not close the fd twice

    assert not lease_file.exists()


@_ALL_HOSTS
def test_sweep_leaves_staged_names_alone(tmp_path):
    """A concurrent creator's staged file may not be locked yet; it is never swept."""
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    staging = generation / ".leases" / ".concurrent-creator.staging"
    staging.parent.mkdir(parents=True)
    staging.write_bytes(b"")

    release = lease_directory(generation)

    assert staging.exists()
    release()
