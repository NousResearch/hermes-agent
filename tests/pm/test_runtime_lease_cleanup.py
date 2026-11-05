"""Lease pruning keeps its exclusion through deletion and retains uncertain owners."""

import json
import os
from pathlib import Path

import pytest

from hermes_cli import runtime_state


def _generation(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / ".lease-managed").touch()
    (root / ".leases").mkdir()
    return root


@pytest.mark.platforms("any")
def test_pruner_removes_abandoned_file_while_holding_its_lock(tmp_path, monkeypatch):
    generation = _generation(tmp_path / "generation")
    lease = generation / ".leases" / "abandoned"
    lease.touch()
    unlink = Path.unlink
    locked_at_delete = []

    def observe_unlink(path, *args, **kwargs):
        if path == lease:
            probe = os.open(path, os.O_RDWR)
            try:
                locked_at_delete.append(not runtime_state._lock(probe, wait=False))
            finally:
                os.close(probe)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", observe_unlink)
    assert not runtime_state.leases_held(generation)
    assert not lease.exists()
    assert locked_at_delete == [True]


@pytest.mark.platforms("windows")
def test_pruner_keeps_a_windows_creator_before_its_first_lock(tmp_path, monkeypatch):
    generation = _generation(tmp_path / "generation")
    lock = runtime_state._lock
    observed = []

    def interleave_prune(fd, *, wait, **kwargs):
        if wait and not observed:
            created = list((generation / ".leases").iterdir())
            observed.append((runtime_state.leases_held(generation), created))
        return lock(fd, wait=wait, **kwargs)

    monkeypatch.setattr(runtime_state, "_lock", interleave_prune)
    release = runtime_state.lease_directory(generation)
    try:
        held, created = observed[0]
        assert held, "An unlocked but still-open creator is not an abandoned owner"
        assert len(created) == 1 and created[0].exists()
        assert runtime_state.leases_held(generation)
    finally:
        release()
    assert list((generation / ".leases").iterdir()) == []


@pytest.mark.platforms("any")
@pytest.mark.parametrize("collector", ["runtime", "dependencies"])
def test_unremovable_lease_keeps_its_generation(tmp_path, monkeypatch, collector):
    from pm.environments import install_state_dir, runtime_facts_path
    from pm.runtime import collect_runtime_generations

    if collector == "runtime":
        root = tmp_path / "runtime"
        generation = _generation(root / "generations" / "obsolete")
        (generation / "pm-runtime.json").write_text("{}", encoding="utf-8")
        collect = lambda: collect_runtime_generations(root)
    else:
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        state = install_state_dir(project)
        generation = _generation(state / "environments" / "obsolete")
        selected = _generation(state / "environments" / "selected") / "venv"
        selected.mkdir()
        (selected / "pyvenv.cfg").write_text("version = 3.14\n", encoding="utf-8")
        runtime_facts_path(project).write_text(
            json.dumps({"packages": {"venv": {"environment": str(selected)}}}),
            encoding="utf-8",
        )
        collect = lambda: runtime_state.collect_generations(project, min_age_seconds=0)
    lease = generation / ".leases" / "abandoned"
    lease.touch()
    unlink = Path.unlink

    def refuse_lease(path, *args, **kwargs):
        if path == lease:
            raise PermissionError("Synthetic lease deletion refusal")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", refuse_lease)
        assert runtime_state.leases_held(generation)
        assert collect() == []
        assert generation.is_dir() and lease.exists()
    assert collect() == [generation]
    assert not generation.exists()
