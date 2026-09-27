"""Superseded and aborted PM runtime generations are collected; running workers are not."""
import json
import os
from pathlib import Path

from pm.runtime import collect_runtime_generations


def _generation(root: Path, name: str, *, published: bool = True, leased: bool = True) -> Path:
    generation = root / "generations" / name
    generation.mkdir(parents=True)
    if leased:
        (generation / ".lease-managed").touch()
    if published:
        (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}), encoding="utf-8")
    return generation


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


def test_new_lease_keeps_a_live_foreign_lease(tmp_path):
    from pm.filesystem import lock_fd
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    live = generation / ".leases" / "live-foreign"
    live.parent.mkdir(parents=True)
    live.write_bytes(b"")
    holder = os.open(live, os.O_RDWR)
    try:
        assert lock_fd(holder, wait=True)
        release = lease_directory(generation)
        assert live.exists()  # a held lease is never swept
        release()
    finally:
        os.close(holder)


def test_release_is_idempotent(tmp_path):
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path, "gen")
    release = lease_directory(generation)
    lease_file = next((generation / ".leases").glob("*"))

    release()
    release()  # an explicit call after atexit must not close the fd twice

    assert not lease_file.exists()


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
