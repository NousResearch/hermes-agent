"""ABI guard for activate_dependencies (#122555).

A committed dependency environment built for another interpreter must not be
activated: its C extensions cannot load here, and pure-Python packages would
silently resolve against the wrong tree. Activation keeps the launch packages
instead.
"""
import json
import os
import sys

import pytest


def _commit(tmp_path, monkeypatch, cfg_text):
    from pm.environments import install_state_dir, runtime_facts_path, site_packages

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    environment = install_state_dir(repo) / "environments" / "gen1" / "venv"
    (environment / "pyvenv.cfg").parent.mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text(cfg_text, encoding="utf-8")
    site = site_packages(environment)
    site.mkdir(parents=True)
    (environment.parent / ".lease-managed").touch()
    runtime_facts_path(repo).write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
    return repo, environment, site


def _running_version():
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _other_version():
    # Deterministically not the running interpreter, on any host.
    return f"{sys.version_info.major}.{sys.version_info.minor + 1}"


def test_venv_python_version_reads_version_info_key(tmp_path):
    """uv writes `version_info`, CPython writes `version` — both declare the tree."""
    from pm.environments import venv_python_version

    venv = tmp_path / "venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = x\nversion_info = 3.11.15\n", encoding="utf-8")
    assert venv_python_version(venv) == (3, 11)


def test_mismatched_tree_keeps_launch_packages(tmp_path, monkeypatch, caplog):
    """The #122555 incident: a 3.14 tree must not rewire a 3.11 process."""
    import pm.environments as runtime_paths

    repo, environment, site = _commit(tmp_path, monkeypatch, f"version = {_other_version()}\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    before = list(sys.path)

    with caplog.at_level("ERROR", logger="pm.environments"):
        runtime_paths.activate_dependencies(repo)

    assert list(sys.path) == before
    assert str(site) not in sys.path
    assert str(site) not in os.environ.get("PYTHONPATH", "")
    assert any("was built for Python" in r.message for r in caplog.records)


def test_matching_tree_activates(tmp_path, monkeypatch):
    import pm.environments as runtime_paths

    repo, _environment, site = _commit(tmp_path, monkeypatch, f"version = {_running_version()}\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    runtime_paths.activate_dependencies(repo)

    assert sys.path[1] == str(site)


def test_undeclared_version_keeps_old_behavior(tmp_path, monkeypatch):
    """No declared version (legacy cfg): activation proceeds as before."""
    import pm.environments as runtime_paths

    repo, _environment, site = _commit(tmp_path, monkeypatch, "home = fixture\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    for key in ("PYTHONPATH", "PATH", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, os.environ.get(key, ""))

    runtime_paths.activate_dependencies(repo)

    assert str(site) in sys.path
