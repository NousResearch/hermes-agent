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


@pytest.fixture(autouse=True)
def checkout_venvs_are_not_an_install(monkeypatch):
    """The checkout running the suite is never a test's pre-PM install.

    Many PM tests keep ``repo_root`` at the real checkout under an isolated HERMES_HOME (no
    recorded selection), where PM reads that checkout's ``venv/`` or ``.venv/`` as the old
    install's. Whatever a contributor keeps there must not change what those tests select; a
    test of the legacy carry builds its own tree under ``tmp_path``.
    """
    from pm import extras

    checkout = Path(__file__).resolve().parents[2]
    real = extras._pre_pm_site_packages
    monkeypatch.setattr(extras, "_pre_pm_site_packages",
                        lambda root: [] if Path(root).resolve() == checkout else real(root))
