"""Tests for gateway boot-line profile announcement (gateway/run.py)."""

import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.run import _log_boot_announcement


def _capture(caplog):
    """Return the announcement record message, or None if not emitted."""
    for rec in caplog.records:
        if rec.message.startswith("[hermes] starting"):
            return rec.message
    return None


def test_default_profile_announces_default(tmp_path, caplog, monkeypatch):
    home = tmp_path / "home"
    hermes_root = home / ".hermes"
    hermes_root.mkdir(parents=True)
    (hermes_root / "active_profile").write_text("ruta\n")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))

    caplog.set_level(logging.INFO, logger="gateway.run")
    log = logging.getLogger("gateway.run")
    _log_boot_announcement(log)

    msg = _capture(caplog)
    assert msg is not None
    assert "profile=default" in msg
    assert "active_profile_file=ruta" in msg
    assert f"pid={os.getpid()}" in msg


def test_profile_billprinter_announces_correctly(tmp_path, caplog, monkeypatch):
    home = tmp_path / "home"
    default_root = home / ".hermes"
    default_root.mkdir(parents=True)
    profile_home = default_root / "profiles" / "billprinter"
    profile_home.mkdir(parents=True)
    (default_root / "active_profile").write_text("default")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    caplog.set_level(logging.INFO, logger="gateway.run")
    log = logging.getLogger("gateway.run")
    _log_boot_announcement(log)

    msg = _capture(caplog)
    assert msg is not None
    assert "profile=billprinter" in msg
    assert "active_profile_file=default" in msg


def test_missing_active_profile_file(tmp_path, caplog, monkeypatch):
    home = tmp_path / "home"
    hermes_root = home / ".hermes"
    hermes_root.mkdir(parents=True)
    # No active_profile file written.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))

    caplog.set_level(logging.INFO, logger="gateway.run")
    log = logging.getLogger("gateway.run")
    _log_boot_announcement(log)

    msg = _capture(caplog)
    assert msg is not None
    assert "active_profile_file=(missing)" in msg


def test_unreadable_active_profile_file(tmp_path, caplog, monkeypatch):
    home = tmp_path / "home"
    hermes_root = home / ".hermes"
    hermes_root.mkdir(parents=True)
    active = hermes_root / "active_profile"
    active.write_text("ruta")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_root))

    original_read_text = Path.read_text

    def boom(self, *args, **kwargs):
        if self == active:
            raise OSError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", boom)

    caplog.set_level(logging.INFO, logger="gateway.run")
    log = logging.getLogger("gateway.run")
    _log_boot_announcement(log)

    msg = _capture(caplog)
    assert msg is not None
    assert "active_profile_file=(unreadable)" in msg
