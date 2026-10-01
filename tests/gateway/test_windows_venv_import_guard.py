"""A PM-managed Windows gateway never overlays the leftover pre-PM venv (#122183).

hermes_bootstrap already activated the store Python onto the committed generation; the
late ``_ensure_windows_gateway_venv_imports`` overlay used to prepend ``<root>/venv`` (or
``VIRTUAL_ENV``) regardless, loading a cp311 ``pydantic_core`` into 3.14. With no
committed generation the overlay is still ABI-checked against the running interpreter
(#122556): a leftover venv built for another Python must not shadow the boot environment.
"""

import os
import sys

import pytest

from gateway.run import _venv_pyver

pytestmark = pytest.mark.platforms("windows")


CURRENT = (sys.version_info.major, sys.version_info.minor)
OTHER = (3, 11) if CURRENT != (3, 11) else (3, 12)


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


def _write_pyvenv_cfg(venv_dir, version):
    (venv_dir / "pyvenv.cfg").write_text(
        "home = C:\\py\\bin\nversion-info = %d.%d.7\n" % version, encoding="utf-8")


def test_matching_venv_is_still_overlaid(tmp_path, monkeypatch):
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv_dir = root / "venv"
    (venv_dir / "Lib" / "site-packages").mkdir(parents=True)
    _write_pyvenv_cfg(venv_dir, CURRENT)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    site_entry = str((venv_dir / "Lib" / "site-packages").resolve())
    assert site_entry in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv_dir.resolve())


def test_foreign_abi_venv_is_refused(tmp_path, monkeypatch, caplog):
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    venv_dir = root / "venv"
    (venv_dir / "Lib" / "site-packages").mkdir(parents=True)
    _write_pyvenv_cfg(venv_dir, OTHER)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)

    with caplog.at_level("WARNING", logger="gateway.run"):
        gateway_run._ensure_windows_gateway_venv_imports()

    site_entry = str((venv_dir / "Lib" / "site-packages").resolve())
    assert site_entry not in sys.path
    assert "VIRTUAL_ENV" not in os.environ
    assert "PYTHONPATH" not in os.environ
    assert any("Refusing venv overlay" in rec.message for rec in caplog.records)


def test_virtualenv_candidate_checked_before_project_venv(tmp_path, monkeypatch):
    """VIRTUAL_ENV is the first candidate: when it carries a foreign ABI the
    project venv behind it must not be overlaid either."""
    import gateway.run as gateway_run

    root = tmp_path / "root"
    (root / "gateway").mkdir(parents=True)
    active = tmp_path / "active"
    (active / "Lib" / "site-packages").mkdir(parents=True)
    _write_pyvenv_cfg(active, OTHER)
    project = root / "venv"
    (project / "Lib" / "site-packages").mkdir(parents=True)
    _write_pyvenv_cfg(project, OTHER)

    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr(gateway_run, "__file__", str(root / "gateway" / "run.py"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("VIRTUAL_ENV", str(active))
    monkeypatch.delenv("PYTHONPATH", raising=False)

    gateway_run._ensure_windows_gateway_venv_imports()

    for venv_dir in (active, project):
        site_entry = str((venv_dir / "Lib" / "site-packages").resolve())
        assert site_entry not in sys.path
    # Both candidates refused: the environment is left exactly as the launcher set it.
    assert os.environ["VIRTUAL_ENV"] == str(active)
    assert "PYTHONPATH" not in os.environ


class TestVenvPyver:
    def _cfg(self, tmp_path, body):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        (venv_dir / "pyvenv.cfg").write_text(body, encoding="utf-8")
        return venv_dir

    def test_version_info_key_wins(self, tmp_path):
        assert _venv_pyver(self._cfg(tmp_path, "version = 3.9.0\nversion-info = 3.14.7\n")) == (3, 14)

    def test_legacy_version_key(self, tmp_path):
        assert _venv_pyver(self._cfg(tmp_path, "version = 3.11.15\n")) == (3, 11)

    def test_missing_cfg_is_unknown(self, tmp_path):
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        assert _venv_pyver(venv_dir) is None

    def test_unparseable_version_is_unknown(self, tmp_path):
        assert _venv_pyver(self._cfg(tmp_path, "version = banana\n")) is None
