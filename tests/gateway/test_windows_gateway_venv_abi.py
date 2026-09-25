"""Windows gateway venv ABI guard (gateway/run.py).

Regression tests for #122325 / #122324: a stale pre-PM in-tree venv (3.11)
must never shadow the running (PM, 3.14) interpreter — its binary extensions
are the wrong ABI and crash every turn with
``No module named 'pydantic_core._pydantic_core'``.
"""
import os
import sys

import pytest


def _make_venv(root, version):
    (root / "Lib" / "site-packages").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text(f"version = {version}\n", encoding="utf-8")
    return root


@pytest.mark.platforms("windows")
def test_skips_abi_mismatched_venv(tmp_path, monkeypatch):
    """A venv built for another Python is skipped, never injected."""
    from gateway.run import _ensure_windows_gateway_venv_imports

    stale = _make_venv(tmp_path / "venv", "3.11.0")
    monkeypatch.setenv("VIRTUAL_ENV", str(stale))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    saved_path = list(sys.path)
    try:
        _ensure_windows_gateway_venv_imports()
        assert str(stale / "Lib" / "site-packages") not in sys.path
    finally:
        sys.path[:] = saved_path
        os.environ.pop("PYTHONPATH", None)


@pytest.mark.platforms("windows")
def test_matching_venv_still_injected(tmp_path, monkeypatch):
    """The guard only skips mismatches — a same-ABI venv still lands on the path."""
    from gateway.run import _ensure_windows_gateway_venv_imports

    running = f"{sys.version_info.major}.{sys.version_info.minor}.0"
    good = _make_venv(tmp_path / "goodvenv", running)
    monkeypatch.setenv("VIRTUAL_ENV", str(good))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    saved_path = list(sys.path)
    try:
        _ensure_windows_gateway_venv_imports()
        assert str(good / "Lib" / "site-packages") in sys.path
    finally:
        sys.path[:] = saved_path
        for var in ("VIRTUAL_ENV", "PYTHONPATH"):
            if var in os.environ:
                del os.environ[var]
