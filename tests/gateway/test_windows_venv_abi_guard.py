"""A leftover pre-PM in-tree venv must never shadow the managed interpreter's dependencies.

``_ensure_windows_gateway_venv_imports`` adds the first ``<repo>/venv`` it can find to
``sys.path`` — correct in the pre-migration layout, harmful afterwards: run under the
managed store Python (3.14), a stale 3.11 tree won the sys.path race and its cp311
``pydantic_core`` extension became unimportable ("No module named
'pydantic_core._pydantic_core'"), which killed the supervised Group Chat worker on every
gateway start while the gateway itself kept reporting healthy.

The guard skips a candidate only when ``pyvenv.cfg`` *proves* another interpreter ABI, so
every payload that worked before (no metadata, unreadable metadata) keeps working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import gateway.run as gateway_run


def _write_venv(tmp_path: Path, cfg: str | None) -> Path:
    """A minimal venv-shaped directory: ``Lib/site-packages`` plus an optional pyvenv.cfg."""
    venv = tmp_path / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    if cfg is not None:
        (venv / "pyvenv.cfg").write_text(cfg, encoding="utf-8")
    return venv


def _running_abi() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _other_abi() -> str:
    """A major.minor that is definitely not the one running these tests."""
    return "3.11" if _running_abi() != "3.11" else "3.14"


def test_venv_for_this_interpreter_is_usable(tmp_path):
    venv = _write_venv(tmp_path, f"home = /opt/python\nversion_info = {_running_abi()}\n")
    assert gateway_run._venv_abi_mismatch(venv) is False


def test_foreign_abi_venv_is_a_mismatch(tmp_path):
    """The regression: this tree is what made the gateway's worker unimportable."""
    venv = _write_venv(tmp_path, f"home = /opt/python\nversion_info = {_other_abi()}\n")
    assert gateway_run._venv_abi_mismatch(venv) is True


def test_missing_pyvenv_cfg_stays_permissive(tmp_path):
    """No metadata (sealed payload venvs) must keep the previous behaviour."""
    assert gateway_run._venv_abi_mismatch(_write_venv(tmp_path, None)) is False


def test_pyvenv_cfg_without_version_info_stays_permissive(tmp_path):
    assert gateway_run._venv_abi_mismatch(_write_venv(tmp_path, "home = /opt/python\n")) is False


def test_unreadable_pyvenv_cfg_does_not_raise(tmp_path):
    venv = _write_venv(tmp_path, None)
    (venv / "pyvenv.cfg").mkdir()  # a directory where a file is expected -> OSError inside
    assert gateway_run._venv_abi_mismatch(venv) is False


@pytest.mark.skipif(sys.platform != "win32", reason="the import shim is Windows-only")
def test_foreign_venv_is_never_added_to_sys_path(tmp_path, monkeypatch):
    venv = _write_venv(tmp_path, f"home = /opt/python\nversion_info = {_other_abi()}\n")
    site_packages = str(venv / "Lib" / "site-packages")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    before = list(sys.path)
    try:
        gateway_run._ensure_windows_gateway_venv_imports()
        assert site_packages not in sys.path
    finally:
        sys.path[:] = before
