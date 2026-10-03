"""ABI-mismatch guard for the Windows gateway legacy-venv injection.

``gateway/run.py::_ensure_windows_gateway_venv_imports`` adds the legacy
``<repo>/venv`` site-packages to sys.path so detached Windows gateway runs see
Hermes packages. PM-era installs lease a store-Python generation instead, and
a stale legacy venv built for another Python ABI injected its site-packages
AHEAD of the leased one — compiled extensions (``pydantic_core``) then failed
to load and crash-looped the gateway's hosted room worker. The injection now
skips candidate venvs whose ``pyvenv.cfg`` ABI does not match the running
interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _write_venv(root: Path, version: str) -> Path:
    venv = root / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(f"home = X\nversion_info = {version}\n", encoding="utf-8")
    return venv


def _running_abi() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _foreign_abi() -> str:
    return "3.11" if (sys.version_info.major, sys.version_info.minor) != (3, 11) else "3.12"


class TestVenvInterpreterMatches:
    def test_matching_abi_matches(self, tmp_path):
        from gateway.run import _venv_interpreter_matches

        assert _venv_interpreter_matches(_write_venv(tmp_path, _running_abi())) is True

    def test_mismatched_abi_does_not_match(self, tmp_path):
        from gateway.run import _venv_interpreter_matches

        assert _venv_interpreter_matches(_write_venv(tmp_path, _foreign_abi())) is False

    def test_missing_pyvenv_cfg_fails_open(self, tmp_path):
        from gateway.run import _venv_interpreter_matches

        venv = tmp_path / "venv"
        (venv / "Lib" / "site-packages").mkdir(parents=True)

        assert _venv_interpreter_matches(venv) is True


@pytest.fixture
def _restore_sys_path():
    saved = list(sys.path)
    yield
    sys.path[:] = saved


@pytest.mark.platforms("windows")
class TestEnsureWindowsGatewayVenvImports:
    def _fake_project(self, monkeypatch, tmp_path, version: str | None) -> Path:
        """Point gateway.run's project root at a fake repo with a legacy venv."""
        root = tmp_path / "repo"
        (root / "gateway").mkdir(parents=True)
        (root / "gateway" / "run.py").write_text("", encoding="utf-8")
        monkeypatch.setattr("gateway.run.__file__", str(root / "gateway" / "run.py"))
        if version is not None:
            _write_venv(root, version)
        return root

    def test_mismatched_legacy_venv_is_not_injected(
        self, tmp_path, monkeypatch, _restore_sys_path
    ):
        import gateway.run as run_mod

        before = {p for p in sys.path if "site-packages" in p}
        self._fake_project(monkeypatch, tmp_path, version=_foreign_abi())

        run_mod._ensure_windows_gateway_venv_imports()

        added = {p for p in sys.path if "site-packages" in p} - before
        assert not any(str(tmp_path) in p for p in added), added

    def test_matching_legacy_venv_is_injected(
        self, tmp_path, monkeypatch, _restore_sys_path
    ):
        import gateway.run as run_mod

        self._fake_project(monkeypatch, tmp_path, version=_running_abi())

        run_mod._ensure_windows_gateway_venv_imports()

        assert any(
            str(tmp_path) in p and "site-packages" in p for p in sys.path
        )
