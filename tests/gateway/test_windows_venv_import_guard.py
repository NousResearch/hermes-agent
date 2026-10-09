"""Gateway Windows venv import guards (#122183, #57467)."""

import os
import sys

import pytest

pytestmark = pytest.mark.platforms("windows")


def test_committed_generation_blocks_the_legacy_venv_overlay(tmp_path, monkeypatch):
    """A PM-managed Windows gateway never overlays the leftover pre-PM venv.

    hermes_bootstrap already activated the store Python onto the committed
    generation; the late ``_ensure_windows_gateway_venv_imports`` overlay used
    to prepend ``<root>/venv`` (or ``VIRTUAL_ENV``) regardless, loading a cp311
    ``pydantic_core`` into 3.14.
    """
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    legacy = root / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    generation = tmp_path / "gen"
    (generation / "Lib" / "site-packages").mkdir(parents=True)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: generation)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    legacy_site = str((legacy / "Lib" / "site-packages").resolve())
    assert legacy_site not in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(legacy)
    assert "PYTHONPATH" not in os.environ


def test_venv_import_setup_scopes_to_sys_path_not_global_environ(
    tmp_path, monkeypatch
):
    """The venv import setup must not leak into the global environment (#57467).

    ``_ensure_windows_gateway_venv_imports`` patches the CURRENT process via
    ``sys.path``/``site.addsitedir`` only. Mutating ``os.environ["PYTHONPATH"]``
    there poisoned every child process spawned from the gateway session —
    non-Hermes Python tools resolved Hermes' venv packages ahead of their own.
    """
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv = root / "venv"
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("PYTHONPATH", "already-there")

    gateway_run._ensure_windows_gateway_venv_imports()

    # In-process import surface: project root + venv site-packages on sys.path.
    assert str(root.resolve()) in sys.path
    assert str(site_packages) in sys.path
    # VIRTUAL_ENV stays — standard, scoped, no cross-process poisoning.
    assert os.environ["VIRTUAL_ENV"] == str(venv.resolve())
    # The inherited PYTHONPATH must be untouched — no Hermes paths prepended.
    assert os.environ["PYTHONPATH"] == "already-there"
    assert str(site_packages) not in os.environ["PYTHONPATH"]
    assert str(root) not in os.environ["PYTHONPATH"]
