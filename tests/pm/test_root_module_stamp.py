"""The venv stamp must cover the root single-file modules.

The PEP 660 editable install maps root modules by name when it is built
(setup.py derives ``py_modules`` from the tree), so a module that a source
update adds (``hermes_yaml``) is invisible to an existing venv until the
editable is rebuilt. The stamp is the gate that decides whether a sync is
owed, so it has to track the module name set; module bodies are loaded from
source and must NOT be part of the stamp, or every edit would demand a
rebuild.
"""

from __future__ import annotations


def _repo(tmp_path, monkeypatch):
    from pm import paths

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    return repo


def test_added_root_module_changes_the_stamp(tmp_path, monkeypatch):
    from pm.packages import Venv

    repo = _repo(tmp_path, monkeypatch)
    (repo / "run_agent.py").write_text("value = 1\n")
    before = Venv().expected_stamp([])
    (repo / "hermes_yaml.py").write_text("value = 2\n")
    assert Venv().expected_stamp([]) != before


def test_removed_root_module_changes_the_stamp(tmp_path, monkeypatch):
    from pm.packages import Venv

    repo = _repo(tmp_path, monkeypatch)
    (repo / "run_agent.py").write_text("value = 1\n")
    before = Venv().expected_stamp([])
    (repo / "run_agent.py").unlink()
    assert Venv().expected_stamp([]) != before


def test_edited_module_body_does_not_change_the_stamp(tmp_path, monkeypatch):
    from pm.packages import Venv

    repo = _repo(tmp_path, monkeypatch)
    (repo / "hermes_yaml.py").write_text("value = 1\n")
    before = Venv().expected_stamp([])
    (repo / "hermes_yaml.py").write_text("value = 2\n")
    assert Venv().expected_stamp([]) == before


def test_setup_py_stays_outside_the_module_stamp(tmp_path, monkeypatch):
    from pm.packages import Venv

    repo = _repo(tmp_path, monkeypatch)
    before = Venv().expected_stamp([])
    (repo / "setup.py").write_text("value = 1\n")
    assert Venv().expected_stamp([]) == before
