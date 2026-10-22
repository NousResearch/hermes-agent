"""Shared fixtures for the PM suite."""
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_machine_home: keep Path.home() intact (a subprocess or stage_runtime needs the real one)")


@pytest.fixture(autouse=True)
def isolated_machine_home(request, tmp_path, monkeypatch):
    """Machine-scoped PM state (store, caches, profiles root) lands under tmp_path.

    The root conftest already sandboxes HERMES_HOME; PM additionally keys its machine cache
    off the home directory, so that moves too — in this process (Path.home) and in any child
    (HOME / USERPROFILE). Opt out with ``@pytest.mark.real_machine_home``.
    """
    if request.node.get_closest_marker("real_machine_home"):
        return
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
