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


def _write_pyvenv_cfg(venv_dir, version_line):
    (venv_dir / "pyvenv.cfg").write_text(
        f"home = /elsewhere\n{version_line}\n", encoding="utf-8")


def _current_minor():
    return f"{sys.version_info[0]}.{sys.version_info[1]}"


@pytest.mark.parametrize("version_line", [f"version = {_current_minor()}.5",
                                          f"version-info = {_current_minor()}"],
                         ids=["version", "version-info"])
def test_matching_abi_venv_still_overlays(tmp_path, monkeypatch, version_line):
    """A venv pinned to the RUNNING interpreter's ABI keeps the overlay (fail-open only
    applies to unreadable/foreign pins)."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv = root / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    _write_pyvenv_cfg(venv, version_line)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    site = str((venv / "Lib" / "site-packages").resolve())
    assert site in sys.path


def test_foreign_abi_venv_is_never_overlaid(tmp_path, monkeypatch):
    """#123185: the pinned cp311 tree overlaid under a newer system Python (PATH race)
    crashes every boot on the first compiled import. The overlay must refuse a venv
    whose pyvenv.cfg pins a different major.minor than the running interpreter."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    legacy = root / "venv"
    (legacy / "Lib" / "site-packages").mkdir(parents=True)
    foreign = "3.7" if sys.version_info[:2] != (3, 7) else "3.9"
    _write_pyvenv_cfg(legacy, f"version = {foreign}.3")

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(legacy))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    legacy_site = str((legacy / "Lib" / "site-packages").resolve())
    assert legacy_site not in sys.path
    # VIRTUAL_ENV is only rewritten for an overlay that actually happened.
    assert os.environ["VIRTUAL_ENV"] == str(legacy)
    assert "PYTHONPATH" not in os.environ


def test_unreadable_pyvenv_cfg_fails_open(tmp_path, monkeypatch):
    """A legacy venv without a readable pin keeps today's overlay behavior."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv = root / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)  # no pyvenv.cfg at all

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    site = str((venv / "Lib" / "site-packages").resolve())
    assert site in sys.path
