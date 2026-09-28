"""A False availability verdict names its failing branch — the bare ``returned False`` was
undiagnosable from outside the process (#126634)."""

import logging
from pathlib import Path

import pytest

from tests.computer_use.driver_fixture import record_driver


@pytest.fixture
def cua_home(tmp_path, monkeypatch):
    from pm import paths

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    monkeypatch.delenv("HERMES_CUA_DRIVER_CMD", raising=False)
    monkeypatch.setenv("HERMES_DISABLE_LAZY_INSTALLS", "1")
    monkeypatch.setattr(paths, "lockfile_path", lambda: tmp_path / "lock.json")
    return tmp_path


@pytest.fixture
def gateway_placement(monkeypatch):
    """Pin placement to the gateway host so the availability verdict is the driver resolution's alone."""
    from tools.bot_desktop import placement

    monkeypatch.setattr(placement, "resolve", lambda: placement.Placement(placement.GATEWAY, "local"))


@pytest.fixture(autouse=True)
def reset_last_reason():
    import tools.computer_use.tool as cut

    cut._last_unavailable_reason = None
    yield
    cut._last_unavailable_reason = None


def _unavailable_lines(caplog):
    return [r.message for r in caplog.records if "computer_use unavailable" in r.message]


def test_missing_record_names_store_and_install_command(cua_home, gateway_placement, caplog):
    from tools.computer_use.tool import check_computer_use_requirements

    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is False
    (line,) = _unavailable_lines(caplog)
    assert "no install record" in line
    assert str(cua_home / "tools") in line
    assert "hermes computer-use install" in line


def test_pin_drift_names_recorded_and_pinned_versions(cua_home, gateway_placement, caplog):
    from pm import Lockfile, current_target, get_package, paths

    record_driver()
    package, target = get_package("cua-driver"), current_target()
    artifact = {"url": "https://example.invalid/cua-fixture", "sha256": "a" * 64}
    lock = Lockfile(paths.lockfile_path())
    lock.set_pin(package.name, "0.21.0", {target: artifact})
    lock.save()

    from tools.computer_use.tool import check_computer_use_requirements

    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is False
    (line,) = _unavailable_lines(caplog)
    assert "recorded v0.20.0" in line and "pin v0.21.0" in line


def test_deleted_binary_still_names_the_store(cua_home, gateway_placement, caplog):
    binary = record_driver()
    binary.unlink()

    from tools.computer_use.tool import check_computer_use_requirements

    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is False
    (line,) = _unavailable_lines(caplog)
    assert "PM does not select cua-driver" in line
    assert str(cua_home / "tools") in line


def test_unresolvable_override_names_the_env_var(cua_home, caplog):
    import os

    os.environ["HERMES_CUA_DRIVER_CMD"] = str(cua_home / "nowhere" / "cua-driver")
    try:
        from tools.computer_use.cua_backend_driver import cua_driver_binary_status

        available, reason = cua_driver_binary_status()
        assert available is False
        assert "HERMES_CUA_DRIVER_CMD" in reason and "nowhere" in reason
    finally:
        del os.environ["HERMES_CUA_DRIVER_CMD"]


def test_reason_logs_once_per_distinct_reason(cua_home, gateway_placement, caplog, monkeypatch):
    from tools.computer_use.tool import check_computer_use_requirements

    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is False
        assert check_computer_use_requirements() is False
    assert len(_unavailable_lines(caplog)) == 1

    caplog.clear()
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(cua_home / "elsewhere" / "cua-driver"))
    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is False
        assert check_computer_use_requirements() is False
    lines = _unavailable_lines(caplog)
    assert len(lines) == 1 and "HERMES_CUA_DRIVER_CMD" in lines[0]


def test_available_driver_stays_silent(cua_home, caplog):
    record_driver()

    from tools.computer_use.cua_backend_driver import cua_driver_binary_status
    from tools.computer_use.tool import check_computer_use_requirements

    assert cua_driver_binary_status() == (True, "")
    with caplog.at_level(logging.INFO, logger="tools.computer_use.tool"):
        assert check_computer_use_requirements() is True
    assert _unavailable_lines(caplog) == []
