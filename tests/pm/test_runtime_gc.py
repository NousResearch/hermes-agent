"""Superseded and aborted PM runtime generations are collected; running workers are not."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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


def test_next_reader_removes_lease_left_by_hard_exit(tmp_path):
    from hermes_cli.runtime_state import lease_directory

    generation = _generation(tmp_path / "pm-runtime", "selected")
    code = """
import os
import sys
from pathlib import Path
from hermes_cli.runtime_state import lease_directory

lease_directory(Path(sys.argv[1]))
os._exit(0)
"""
    subprocess.run([sys.executable, "-c", code, str(generation)], check=True, env=dict(os.environ))
    leases = generation / ".leases"
    stale = list(leases.iterdir())
    assert len(stale) == 1

    release = lease_directory(generation)
    try:
        active = list(leases.iterdir())
        assert len(active) == 1
        assert active[0] not in stale
    finally:
        release()

    assert list(leases.iterdir()) == []


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root opens mode-0 files")
def test_prune_keeps_unopenable_sibling_lease(tmp_path):
    """A lease this user cannot open (root-owned, `sudo hermes` on the same checkout) is HELD:
    boot never raises and GC never collects."""
    from hermes_cli.runtime_state import lease_directory, leases_held

    generation = _generation(tmp_path / "pm-runtime", "selected")
    leases = generation / ".leases"
    leases.mkdir()
    foreign = leases / "foreign"
    foreign.touch()
    foreign.chmod(0)
    try:
        lease_directory(generation)()
        assert foreign.exists()
        assert leases_held(generation)
    finally:
        foreign.chmod(0o600)


def test_lease_taken_while_a_peer_pruned_it_is_retaken(tmp_path, monkeypatch):
    """A peer's prune can lock-and-unlink our lease between create and flock; the writer must
    end up holding a lease the collectors can see, never an unlinked inode."""
    from hermes_cli import runtime_state
    from hermes_cli.runtime_state import lease_directory, leases_held

    generation = _generation(tmp_path / "pm-runtime", "selected")
    leases = generation / ".leases"
    real_lock = runtime_state._lock
    raced = []

    def racing_lock(fd, *, wait, **kw):
        if wait and not raced:  # the writer's first take: a peer pruned the file first
            for stray in leases.iterdir():
                stray.unlink()
            raced.append(fd)
        return real_lock(fd, wait=wait, **kw)

    monkeypatch.setattr(runtime_state, "_lock", racing_lock)
    release = lease_directory(generation)
    try:
        assert raced
        assert len(list(leases.iterdir())) == 1
        assert leases_held(generation)
    finally:
        release()
    assert list(leases.iterdir()) == []
