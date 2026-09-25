"""Windows detached-gateway venv injection must not adopt another interpreter's venv.

PM installs keep one dependency generation per install (a store interpreter + its own
environment) while the checkout may still hold the legacy pre-PM ``venv/``. Injecting that
leftover put a cp311 ``site-packages`` ahead of the real one on a cp314 interpreter, so
``pydantic`` resolved from the old tree and its compiled core refused to load:

    ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'

That killed the gateway's ``hosted_room_worker`` on every update that moved the store Python.
"""

import os
import sys
from pathlib import Path

import pytest

import gateway.run as gr

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows-only helper")


def _fake_venv(root: Path, version: str | None) -> Path:
    """A venv-shaped directory; ``version=None`` omits ``pyvenv.cfg`` (unknown interpreter)."""
    venv = root / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    if version is not None:
        (venv / "pyvenv.cfg").write_text(
            f"home = C:\\Python\nimplementation = CPython\nversion_info = {version}\n",
            encoding="utf-8",
        )
    return venv


def _running_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _other_version() -> str:
    """A Python version that is guaranteed not to be the running one."""
    return "3.12" if _running_version() != "3.12" else "3.11"


def _point_project_at(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fake a checkout root and a host with no PM generation committed."""
    monkeypatch.setattr(gr, "__file__", str(tmp_path / "gateway" / "run.py"))
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))

    import pm.environments as pm_env

    monkeypatch.setattr(pm_env, "committed_venv", lambda project_root=None: None)


def test_other_pythons_venv_is_not_injected(monkeypatch, tmp_path):
    """The pre-PM leftover is skipped instead of shadowing the interpreter's own packages."""
    venv = _fake_venv(tmp_path, _other_version())
    _point_project_at(monkeypatch, tmp_path)
    monkeypatch.setenv("PYTHONPATH", "sentinel")

    gr._ensure_windows_gateway_venv_imports()

    assert str(venv / "Lib" / "site-packages") not in sys.path
    assert os.environ["PYTHONPATH"] == "sentinel"


def test_matching_python_venv_is_still_injected(monkeypatch, tmp_path):
    """Legacy installs (venv built for the running interpreter) keep working unchanged."""
    venv = _fake_venv(tmp_path, _running_version())
    _point_project_at(monkeypatch, tmp_path)

    gr._ensure_windows_gateway_venv_imports()

    site_packages = str(venv / "Lib" / "site-packages")
    assert site_packages in sys.path
    assert os.environ["VIRTUAL_ENV"] == str(venv)


def test_unknown_python_venv_keeps_historical_behavior(monkeypatch, tmp_path):
    """No ``pyvenv.cfg`` → nothing to compare, so the candidate is used as before."""
    venv = _fake_venv(tmp_path, None)
    _point_project_at(monkeypatch, tmp_path)

    gr._ensure_windows_gateway_venv_imports()

    assert str(venv / "Lib" / "site-packages") in sys.path


def test_committed_generation_outranks_the_leftover(monkeypatch, tmp_path):
    """PM's committed environment wins, and the mismatched leftover stays off sys.path."""
    leftover = _fake_venv(tmp_path, _other_version())
    committed = _fake_venv(tmp_path / "installs" / "abc", _running_version())
    _point_project_at(monkeypatch, tmp_path)

    import pm.environments as pm_env

    monkeypatch.setattr(pm_env, "committed_venv", lambda project_root=None: committed)

    gr._ensure_windows_gateway_venv_imports()

    assert str(committed / "Lib" / "site-packages") in sys.path
    assert str(leftover / "Lib" / "site-packages") not in sys.path
