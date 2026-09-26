"""Windows gateway must not mix PM's Python ABI with the old in-tree venv."""

import os
import site
import sys

import pytest

from gateway import run
import pm.environments


@pytest.mark.skipif(sys.platform != "win32", reason="Windows venv overlay only")
def test_committed_pm_environment_is_not_shadowed_by_legacy_venv(monkeypatch, tmp_path):
    """PM selected packages stay first, and gateway leaves its boot environment alone."""
    legacy = tmp_path / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    committed = tmp_path / "committed"
    (committed / "Lib" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(run, "__file__", str(tmp_path / "gateway" / "run.py"))
    monkeypatch.setattr(pm.environments, "committed_venv", lambda root: committed)
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.setenv("PYTHONPATH", "original")
    original_path = list(sys.path)
    added = []
    monkeypatch.setattr(site, "addsitedir", lambda path: added.append(path))

    run._ensure_windows_gateway_venv_imports()

    assert added == []
    assert sys.path == original_path
    assert os.environ["VIRTUAL_ENV"] == str(legacy)
    assert os.environ["PYTHONPATH"] == "original"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows venv overlay only")
def test_legacy_venv_remains_available_without_pm_generation(monkeypatch, tmp_path):
    legacy = tmp_path / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    monkeypatch.setattr(run, "__file__", str(tmp_path / "gateway" / "run.py"))
    monkeypatch.setattr(pm.environments, "committed_venv", lambda root: None)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    added = []
    monkeypatch.setattr(site, "addsitedir", lambda path: added.append(path))

    run._ensure_windows_gateway_venv_imports()

    assert added == [str(legacy.resolve() / "Lib" / "site-packages")]
    assert str(legacy.resolve() / "Lib" / "site-packages") in sys.path
