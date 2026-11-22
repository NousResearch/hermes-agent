"""Every entry point imports hermes_bootstrap first, whose activate_dependencies imports
hermes_cli.runtime_state at boot in a PM-managed install: a v0.23.3 process keeps THAT runtime_state
(no moved_lock / install_lock_paths / collect_lease_dirs, and a leases_held that only reads
``<generation>/.leases``) and pm modules it imports lazily after ``hermes update`` come from the new
tree (#158723)."""
from __future__ import annotations

import json
from pathlib import Path

import hermes_cli.runtime_state
import pm.filesystem


def _v0_23_3_leases_held(generation: Path) -> bool:
    """v0.23.3's leases_held: leases inside the generation only."""
    leases = generation / ".leases"
    return leases.is_dir() and any(leases.iterdir())


def _old_boot(monkeypatch):
    monkeypatch.delattr(pm.filesystem, "moved_lock")
    for name in ("moved_lock", "install_lock_paths", "collect_lease_dirs"):
        monkeypatch.delattr(hermes_cli.runtime_state, name, raising=False)
    monkeypatch.setattr(hermes_cli.runtime_state, "leases_held", _v0_23_3_leases_held)


def test_install_states_lock_with_v0_23_3_runtime_state_loaded_at_boot(tmp_path: Path, monkeypatch):
    from pm import install_states

    _old_boot(monkeypatch)
    with install_states._install_lock(tmp_path) as held:
        assert held


def test_runtime_prepare_lock_with_v0_23_3_runtime_state_loaded_at_boot(tmp_path: Path, monkeypatch):
    from pm import runtime

    _old_boot(monkeypatch)
    with runtime._prepare_lock(tmp_path / "runtime", timeout=0) as held:
        assert held


def test_runtime_collector_keeps_a_current_lease_with_v0_23_3_runtime_state(tmp_path: Path, monkeypatch):
    """A current worker leases outside the generation; the old process's collector must see it."""
    from pm.leases import lease_directory
    from pm.runtime import collect_runtime_generations

    root = tmp_path / "pm-runtime"
    generations = {}
    for name in ("selected", "busy"):
        generation = generations[name] = root / "generations" / name
        generation.mkdir(parents=True)
        (generation / ".lease-managed").touch()
        (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}), encoding="utf-8")
    (root / "selected.json").write_text(json.dumps({"generation": "generations/selected"}), encoding="utf-8")
    release = lease_directory(generations["busy"])
    _old_boot(monkeypatch)
    try:
        assert collect_runtime_generations(root) == []
        assert generations["busy"].is_dir()
    finally:
        release()
    assert collect_runtime_generations(root) == [generations["busy"]]
