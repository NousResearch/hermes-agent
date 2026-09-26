"""Windows gateway startup must never put a foreign interpreter's site-packages on the path.

``_ensure_windows_gateway_venv_imports`` publishes ``VIRTUAL_ENV``/``PYTHONPATH`` for every gateway
child, so the tree it picks must belong to the interpreter that is running. Installs migrated from a
repo-local ``venv`` to the bundled store Python keep the old venv on disk (``hermes-agent/venv``
built by uv for 3.11 while the runtime is 3.14): adopting it handed wrong-ABI site-packages to
children, and ``pydantic_core._pydantic_core`` raised ModuleNotFoundError — the gateway's
``hosted_room_worker`` crash-looped five times and gave up, while that tree's pure-Python modules
imported fine.

Invariants: the venv's recorded version is read in both spellings real venvs use; a venv built for
another minor version is skipped, and a matching one is still adopted (the guard must not degrade
into "never adopt anything").
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from gateway.run import _ensure_windows_gateway_venv_imports, _windows_gateway_venv_python_version

pytestmark = pytest.mark.platforms("windows")


def _fake_venv(root: Path, version_line: str) -> Path:
    venv = root / version_line.replace(" ", "").replace("=", "-")
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(f"home = C:\\python\n{version_line}\n", encoding="utf-8")
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    return venv


def test_recorded_python_version_reads_both_pyvenv_spellings(tmp_path):
    assert _windows_gateway_venv_python_version(_fake_venv(tmp_path, "version = 3.11.15")) == (3, 11)
    assert _windows_gateway_venv_python_version(_fake_venv(tmp_path, "version_info = 3.14")) == (3, 14)
    # Unrecorded/unreadable metadata keeps the permissive legacy behaviour.
    assert _windows_gateway_venv_python_version(tmp_path / "absent") is None


def test_foreign_minor_venv_is_skipped_and_matching_venv_is_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr("pm.environments.committed_venv", lambda *_a, **_k: None)
    baseline_pythonpath = os.environ.get("PYTHONPATH", "")

    running = ".".join(str(part) for part in sys.version_info[:2])
    # Any minor version other than the running one; the guard keys on the comparison, not a literal.
    foreign_minor = "3.7" if sys.version_info[:2] != (3, 7) else "3.8"

    foreign = _fake_venv(tmp_path / "foreign", f"version_info = {foreign_minor}")
    monkeypatch.setenv("VIRTUAL_ENV", str(foreign))
    _ensure_windows_gateway_venv_imports()
    assert os.environ["VIRTUAL_ENV"] == str(foreign)
    assert os.environ.get("PYTHONPATH", "") == baseline_pythonpath
    assert str(foreign / "Lib" / "site-packages") not in sys.path

    matching = _fake_venv(tmp_path / "matching", f"version_info = {running}")
    monkeypatch.setenv("VIRTUAL_ENV", str(matching))
    _ensure_windows_gateway_venv_imports()
    adopted = str(matching.resolve())
    assert os.environ["VIRTUAL_ENV"] == adopted
    assert str(matching.resolve() / "Lib" / "site-packages") in os.environ["PYTHONPATH"]
