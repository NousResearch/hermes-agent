"""Regression for #122183: a legacy venv must not shadow PM's selected wheels."""

import os
import sys
import pytest

from gateway import run as gateway_run


@pytest.mark.platforms("windows")
def test_pm_gateway_keeps_committed_imports_ahead_of_legacy_venv(tmp_path, monkeypatch):
    project = tmp_path / "project"
    monkeypatch.setattr(gateway_run, "__file__", str(project / "gateway" / "run.py"))
    legacy = project / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    selected = tmp_path / "generation" / "Lib" / "site-packages"
    selected.mkdir(parents=True)
    monkeypatch.setattr(gateway_run.sys, "path", [str(project), str(selected), *sys.path])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project), str(selected)]))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root: tmp_path / "python.exe")

    before_path = list(sys.path)
    before_pythonpath = os.environ["PYTHONPATH"]
    gateway_run._ensure_windows_gateway_venv_imports()

    assert sys.path == before_path
    assert os.environ["PYTHONPATH"] == before_pythonpath
    assert "VIRTUAL_ENV" not in os.environ
    assert str(legacy / "Lib" / "site-packages") not in sys.path


@pytest.mark.platforms("windows")
def test_unmanaged_gateway_still_activates_existing_venv(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda root: None)
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setattr(gateway_run.sys, "path", list(sys.path))

    gateway_run._ensure_windows_gateway_venv_imports()

    assert str(site_packages) in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv.resolve())
