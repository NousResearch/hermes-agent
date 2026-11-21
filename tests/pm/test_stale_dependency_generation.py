"""stale_dependency_generation: the installed-but-predates-this-process check (#140620)."""

import json
import sys

from pm.environments import install_state_dir, stale_dependency_generation


def _generation(state_dir, name):
    venv = state_dir / "environments" / name
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /nonexistent\n", encoding="utf-8")
    return venv


def _record(path, venv):
    path.write_text(
        json.dumps({"packages": {"venv": {"environment": str(venv)}}}), encoding="utf-8"
    )


def _install_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    project = tmp_path / "repo"
    project.mkdir()
    return project, install_state_dir(project)


def test_process_on_the_committed_generation_is_not_stale(tmp_path, monkeypatch):
    project, state_dir = _install_state(tmp_path, monkeypatch)
    committed = _generation(state_dir, "gen-b")
    _record(state_dir / "facts.json", committed)
    monkeypatch.setattr(sys, "prefix", str(committed))

    assert stale_dependency_generation(project) is False


def test_process_on_an_older_generation_is_stale(tmp_path, monkeypatch):
    project, state_dir = _install_state(tmp_path, monkeypatch)
    running = _generation(state_dir, "gen-a")
    committed = _generation(state_dir, "gen-b")
    _record(state_dir / "facts.json", committed)
    monkeypatch.setattr(sys, "prefix", str(running))

    assert stale_dependency_generation(project) is True


def test_process_outside_pm_generations_is_not_stale(tmp_path, monkeypatch):
    """A dev checkout or bare interpreter keeps the plain install advice."""
    project, state_dir = _install_state(tmp_path, monkeypatch)
    committed = _generation(state_dir, "gen-b")
    _record(state_dir / "facts.json", committed)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "in-tree-venv"))

    assert stale_dependency_generation(project) is False


def test_unreadable_record_reads_not_stale(tmp_path, monkeypatch):
    project, state_dir = _install_state(tmp_path, monkeypatch)
    running = _generation(state_dir, "gen-a")
    (state_dir / "facts.json").write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(running))

    assert stale_dependency_generation(project) is False


def test_missing_record_reads_not_stale(tmp_path, monkeypatch):
    project, state_dir = _install_state(tmp_path, monkeypatch)
    running = _generation(state_dir, "gen-a")
    monkeypatch.setattr(sys, "prefix", str(running))

    assert stale_dependency_generation(project) is False
