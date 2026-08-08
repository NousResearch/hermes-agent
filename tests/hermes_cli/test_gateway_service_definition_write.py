"""A service definition must never be left half-written.

Every caller hands the file straight to the service manager
(``systemctl daemon-reload`` / ``launchctl bootstrap``), so a truncated unit
or plist is parsed immediately — and the refresh paths rewrite a definition
that is already installed and running. The launchd refresh can even be
running inside the gateway's own process tree, which the bootout below it
tears down.
"""

from __future__ import annotations

import os

import pytest

import hermes_cli.gateway as gateway_cli

INSTALLED = "[Unit]\nDescription=Hermes Gateway (installed)\n"
REPLACEMENT = "[Unit]\nDescription=Hermes Gateway (updated)\n"


def test_existing_definition_survives_an_interrupted_write(tmp_path, monkeypatch):
    path = tmp_path / "hermes-gateway.service"
    path.write_text(INSTALLED, encoding="utf-8")
    original = path.read_bytes()

    def boom(fd):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(os, "fsync", boom)
    try:
        gateway_cli._write_service_definition(path, REPLACEMENT)
    except OSError:
        pass  # the durable path refuses to swap in a half-written file

    assert path.read_bytes() == original, (
        "the installed service definition was destroyed; the service manager "
        "would load an empty unit on the next reload"
    )


def test_normal_write_replaces_the_definition(tmp_path):
    path = tmp_path / "hermes-gateway.service"
    path.write_text(INSTALLED, encoding="utf-8")

    gateway_cli._write_service_definition(path, REPLACEMENT)

    assert path.read_text(encoding="utf-8") == REPLACEMENT


def test_write_creates_a_missing_definition(tmp_path):
    path = tmp_path / "nested" / "com.hermes.gateway.plist"

    gateway_cli._write_service_definition(path, REPLACEMENT)

    assert path.read_text(encoding="utf-8") == REPLACEMENT


def test_new_definition_is_manager_readable(tmp_path):
    """systemd and launchd both need to read the file; 0600 would hide it."""
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    import stat

    path = tmp_path / "hermes-gateway.service"
    gateway_cli._write_service_definition(path, REPLACEMENT)

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_existing_definition_keeps_its_mode(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX permission bits")
    import stat

    path = tmp_path / "hermes-gateway.service"
    path.write_text(INSTALLED, encoding="utf-8")
    os.chmod(path, 0o640)

    gateway_cli._write_service_definition(path, REPLACEMENT)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


class _Stop(Exception):
    """Halt the refresh right after the write, before any launchctl work."""


def test_launchd_refresh_writes_through_the_durable_path(tmp_path, monkeypatch):
    """Pins the wiring — a correct helper is worthless if a caller skips it."""
    plist = tmp_path / "com.hermes.gateway.plist"
    plist.write_text(INSTALLED, encoding="utf-8")

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist)
    monkeypatch.setattr(gateway_cli, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: REPLACEMENT)

    seen: dict = {}

    def spy(path, definition):
        seen["path"] = path
        seen["definition"] = definition
        raise _Stop

    monkeypatch.setattr(gateway_cli, "_write_service_definition", spy)

    with pytest.raises(_Stop):
        gateway_cli.refresh_launchd_plist_if_needed()

    assert seen["path"] == plist
    assert seen["definition"] == REPLACEMENT
