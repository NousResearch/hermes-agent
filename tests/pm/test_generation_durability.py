"""A generation's selection record is its completion marker, so it may only be
written after the finished tree has been flushed to disk. Otherwise a power loss
mid-publish can persist the (fsynced) record while the venv's files are still in
the page cache, and the next boot selects a half-written environment.
"""
from __future__ import annotations

import os


def _flush_probe(monkeypatch, observe):
    """Replace the filesystem-wide flush with one that records what was on disk
    when it ran; ``raising=False`` so the contract is exercised on Windows too."""
    seen: list = []
    monkeypatch.setattr(os, "sync", lambda: seen.append(observe()), raising=False)
    return seen


def test_install_selection_is_committed_only_after_its_tree_is_flushed(tmp_path, monkeypatch):
    import pm.install as install
    from pm import paths
    from pm.environments import install_state_dir, runtime_facts_path, selected_venv

    store = tmp_path / "tools"
    store.mkdir()
    monkeypatch.setattr("pm.paths.store_root", lambda: store)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    venv = install.get_package("venv")
    monkeypatch.setattr(venv, "expected_stamp", lambda extras, **kwargs: "stamp")
    candidate = install_state_dir(repo) / "environments" / "fresh" / "venv"

    def apply(*args, **kwargs):
        candidate.mkdir(parents=True)
        (candidate / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        return {"environment": candidate}

    monkeypatch.setattr(venv, "apply", apply)

    def committed() -> bool:
        return runtime_facts_path(repo).is_file() and selected_venv(repo) == candidate.resolve()

    flushes = _flush_probe(monkeypatch, lambda: (candidate / "pyvenv.cfg").is_file() and not committed())
    install.sync_venv([], explicit=True)

    assert flushes == [True], "the tree must be flushed after it is complete and before it is selected"
    assert selected_venv(repo) == candidate.resolve()


def test_side_environment_selection_is_committed_only_after_its_tree_is_flushed(tmp_path, monkeypatch):
    from pm import _uv, operations

    base = tmp_path / "base" / "python"
    base.parent.mkdir()
    base.touch()
    monkeypatch.setattr(_uv, "_toolchain", lambda **kwargs: (None, base))
    root = tmp_path / "side"
    built: list = []

    def build(generation, base_python):
        environment = generation / "venv"
        python = operations._python(environment)
        python.parent.mkdir(parents=True)
        python.touch()
        (environment / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
        built.append(environment)
        return python

    flushes = _flush_probe(monkeypatch, lambda: bool(built) and not (root / "active.json").exists())
    python = operations._ensure_generation("proof", root, {}, build, record={}, explicit=True)

    assert flushes == [True], "the tree must be flushed after it is complete and before it is selected"
    assert operations.environment_python("proof", root=root) == python
