"""A PM-managed Windows gateway never overlays the leftover pre-PM venv (#122183).

hermes_bootstrap already activated the store Python onto the committed generation; the
late ``_ensure_windows_gateway_venv_imports`` overlay used to prepend ``<root>/venv`` (or
``VIRTUAL_ENV``) regardless, loading a cp311 ``pydantic_core`` into 3.14.
"""

import os
import sys

import pytest

pytestmark = pytest.mark.platforms("windows")


def test_committed_generation_blocks_the_legacy_venv_overlay(tmp_path, monkeypatch):
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
