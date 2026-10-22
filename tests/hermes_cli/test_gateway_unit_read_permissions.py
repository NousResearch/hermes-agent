"""Non-root reads of a root-only system unit must not raise.

A system-scope unit installed by root is commonly mode ``0640 root:root``. Every
unprivileged ``hermes gateway status`` / ``hermes gateway migrate`` read of that
file has to degrade to "unknown" instead of dying with ``PermissionError``.
"""

from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_cli

UNIT_TEXT = """[Unit]
Description=Hermes Agent Gateway

[Service]
User=ubuntu
Environment="HERMES_HOME=/home/ubuntu/.hermes"
ExecStart=/usr/bin/hermes gateway run
"""


@pytest.fixture
def unreadable_unit(tmp_path, monkeypatch):
    """A unit that exists but raises ``PermissionError`` on read, like a 0640 root unit."""
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(UNIT_TEXT, encoding="utf-8")
    real_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if self == unit:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    return unit


def test_read_systemd_user_returns_none_for_unreadable_unit(unreadable_unit):
    assert gateway_cli._read_systemd_user_from_unit(unreadable_unit) is None


def test_read_systemd_user_still_parses_readable_unit(tmp_path):
    unit = tmp_path / "hermes-gateway.service"
    unit.write_text(UNIT_TEXT, encoding="utf-8")

    assert gateway_cli._read_systemd_user_from_unit(unit) == "ubuntu"


def test_unit_is_current_does_not_raise_for_unreadable_unit(unreadable_unit, monkeypatch):
    # Isolate the read guard from the systemd/env lookups that surround it.
    monkeypatch.setattr(gateway_cli, "get_systemd_unit_path", lambda system=False: unreadable_unit)
    monkeypatch.setattr(gateway_cli, "_sync_hermes_home_from_systemd_unit", lambda system: None)

    # An unreadable unit cannot be proven stale, and rewriting it needs the same
    # privileges that would allow reading it -- so status must not nag about it.
    assert gateway_cli.systemd_unit_is_current(system=True) is True
