"""``gateway/run.py::_ensure_windows_gateway_venv_imports`` must not shadow the
dependency environment PM activated at boot with the legacy in-tree venv.

The shim exists for detached Windows launches that would otherwise see no
packages at all; when PM already put the selected generation's site-packages on
``sys.path``, injecting the legacy in-tree venv shadows it (wheels built for a
different interpreter) and breaks every hard-binary import — pydantic_core has
no pure-Python fallback, so the hosted-room worker crash-loops on it.
"""
import os
import sys
from pathlib import Path

import pytest


def _fake_project_root(monkeypatch, gateway_run, fake_root: Path) -> None:
    real_path = Path

    def _path(arg=None, *args, **kwargs):
        if isinstance(arg, str) and arg.replace("\\", "/").endswith("gateway/run.py"):
            # The call site chains .resolve().parent.parent onto this.
            return fake_root / "gateway" / "run.py"
        return real_path(arg, *args, **kwargs)

    monkeypatch.setattr(gateway_run, "Path", _path)


def _legacy_site_packages(fake_root: Path) -> str:
    (fake_root / "venv" / "Lib" / "site-packages").mkdir(parents=True)
    return str(fake_root / "venv" / "Lib" / "site-packages")


@pytest.mark.platforms("windows")
def test_legacy_venv_not_injected_when_pm_environment_active(monkeypatch, tmp_path):
    import pm.environments
    from gateway import run as gateway_run

    fake_root = tmp_path / "hermes-agent"
    stale = _legacy_site_packages(fake_root)
    _fake_project_root(monkeypatch, gateway_run, fake_root)
    monkeypatch.setattr(pm.environments, "running_from_selected_environment", lambda root: True)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    path_before = list(sys.path)
    pythonpath_before = os.environ.get("PYTHONPATH")

    try:
        gateway_run._ensure_windows_gateway_venv_imports()

        assert stale not in sys.path
        assert sys.path == path_before
        assert os.environ.get("PYTHONPATH") == pythonpath_before
    finally:
        sys.path[:] = path_before


@pytest.mark.platforms("windows")
def test_legacy_venv_still_injected_without_pm_environment(monkeypatch, tmp_path):
    import pm.environments
    from gateway import run as gateway_run

    fake_root = tmp_path / "hermes-agent"
    stale = _legacy_site_packages(fake_root)
    _fake_project_root(monkeypatch, gateway_run, fake_root)
    monkeypatch.setattr(pm.environments, "running_from_selected_environment", lambda root: False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("PYTHONPATH", "")
    path_before = list(sys.path)

    try:
        gateway_run._ensure_windows_gateway_venv_imports()

        assert stale in sys.path
        assert stale in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    finally:
        sys.path[:] = path_before
