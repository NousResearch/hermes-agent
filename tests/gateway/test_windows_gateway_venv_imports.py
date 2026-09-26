"""Regression: the Windows gateway must import from PM's committed venv, not the pre-PM tree.

``gateway/run.py::_ensure_windows_gateway_venv_imports`` fixes up ``sys.path`` for detached
Windows gateway runs. It used to unconditionally append the hardcoded in-tree ``<repo>/venv``
before consulting PM, so when PM had committed a generation elsewhere that pre-migration tree
(``Lib/site-packages``) was prepended instead and its compiled modules — ``pydantic_core`` and
the pywin32 DLLs — shadowed the PM-installed ones. The contract asserted here is that the
environment PM commits wins: it becomes ``VIRTUAL_ENV``, its site-packages is importable
(``.pth`` processing included, which pywin32 needs), and it is advertised in ``PYTHONPATH``.
"""

import os
import sys

import pytest

import gateway.run as gateway_run

pytestmark = pytest.mark.platforms("windows")


def test_committed_venv_wins_over_pre_pm_in_tree_venv(tmp_path, monkeypatch):
    committed_venv = tmp_path / "generations" / "abc" / "venv"
    site_packages = committed_venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    # addsitedir must run .pth files: pywin32 registers pywintypes that way on Windows.
    pth_target = tmp_path / "pth-target"
    pth_target.mkdir()
    (site_packages / "_hermes_probe.pth").write_text(f"{pth_target}\n", encoding="utf-8")

    # No inherited activation: the committed venv is the first and only candidate.
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    import pm.environments as pm_environments

    # Production late-imports ``committed_venv`` inside the function, so the seam is the module attr.
    monkeypatch.setattr(pm_environments, "committed_venv", lambda root, _v=committed_venv: _v)

    saved_path = list(sys.path)
    try:
        gateway_run._ensure_windows_gateway_venv_imports()

        assert os.environ.get("VIRTUAL_ENV") == str(committed_venv)
        assert str(site_packages) in sys.path
        assert str(pth_target) in sys.path, "addsitedir must process .pth entries"
        assert str(committed_venv) in os.environ.get("PYTHONPATH", "")
    finally:
        sys.path[:] = saved_path
        os.environ.pop("PYTHONPATH", None)
        os.environ.pop("VIRTUAL_ENV", None)
