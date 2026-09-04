"""`pm develop` must boot the pm STORE python — never the venv.

Under no-boot-through-venv (pm work item 3) the devshell's `python`
resolves through the store interpreter's bin dir, imports arrive via
PYTHONPATH=<repo>;<venv>/site-packages (repo first), and VIRTUAL_ENV plus
the venv bin dir are deliberately absent — the venv is only a uv sync
target and pyvenv.cfg is inert dead config. The env composition lives in
``pm.cli._develop_env`` so these are real input→output checks against a
fake store, not source-text assertions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import pm.cli as pm_cli
import pm.paths as paths
from pm.store import current_target


def _fake_store(tmp_path: Path, monkeypatch, *, with_python: bool = True):
    """A store laid out like pm's: facts.json + a python entry. The
    interpreter file's bytes don't matter — only the layout does."""
    store = tmp_path / "store"
    entry = store / "python-3.11.15+x20260807-win32-arm64"
    (entry / "bin").mkdir(parents=True)
    if with_python:
        (entry / "bin" / "python3").write_bytes(b"MZ")
        (entry / "python.exe").write_bytes(b"MZ")
    (store / "facts.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "packages": {
                    "python": {
                        "entry": entry.name,
                        "version": "3.11.15+x20260807",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(store))
    return store, entry


def _fake_repo(tmp_path: Path, monkeypatch) -> Path:
    """A fake repo root with a synced venv, substituted for pm.paths.repo_root."""
    repo = tmp_path / "repo"
    win = current_target().startswith("win32")
    site = repo / "venv" / ("Lib" if win else "lib/python3.11") / "site-packages"
    site.mkdir(parents=True)
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    return repo


def _develop_env(tmp_path, monkeypatch, *, with_python=True):
    _fake_store(tmp_path, monkeypatch, with_python=with_python)
    repo = _fake_repo(tmp_path, monkeypatch)
    win = current_target().startswith("win32")
    site = repo / "venv" / ("Lib" if win else "lib/python3.11") / "site-packages"
    return pm_cli._develop_env(["faketool"]), repo, site


def test_python_resolves_to_the_store_not_the_venv(tmp_path, monkeypatch):
    env, _repo, _site = _develop_env(tmp_path, monkeypatch)

    assert env is not None
    first_path_entry = Path(env["PATH"].split(os.pathsep)[0])
    # The store python's bin dir leads PATH — the venv bin dir never does.
    assert "python-" in str(first_path_entry)


def test_pythonpath_is_repo_first_then_venv_site_packages(tmp_path, monkeypatch):
    env, repo, site = _develop_env(tmp_path, monkeypatch)

    entries = env["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(repo)
    assert str(site) in entries[1:]


def test_venv_boot_markers_are_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "somewhere" / "venv"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "somewhere" / "python"))
    env, _repo, _site = _develop_env(tmp_path, monkeypatch)

    assert env is not None
    assert "VIRTUAL_ENV" not in env
    assert "PYTHONHOME" not in env


def test_venv_bin_dir_is_removed_from_path(tmp_path, monkeypatch):
    env, repo, _site = _develop_env(tmp_path, monkeypatch)
    win = current_target().startswith("win32")
    venv_bin = str(repo / "venv" / ("Scripts" if win else "bin"))

    assert venv_bin not in env["PATH"].split(os.pathsep)


def test_no_store_python_yet_means_no_develop_env(tmp_path, monkeypatch):
    env, _repo, _site = _develop_env(tmp_path, monkeypatch, with_python=False)

    assert env is None
