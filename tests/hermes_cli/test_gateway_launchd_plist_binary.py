"""Regression tests for binary-plist handling in ``launchd_plist_is_current``.

``plutil``, Xcode, and macOS tooling can rewrite an installed LaunchAgent plist
into Apple's binary format (``bplist00``). The staleness check read the file as
text, so a binary plist raised ``UnicodeDecodeError`` and took down
``hermes gateway status``, ``gateway install``, and the auto-refresh path --
the exact commands a user reaches for when something is already broken.

These tests pin three behaviors: both on-disk shapes parse, the comparison is
structural (so a plistlib round-trip does not read as stale), and malformed
input reports stale instead of raising.
"""
import plistlib

import pytest

from hermes_cli import gateway_launchd

# The read path keys off Path.exists(); no macOS host is required to exercise
# the parsing itself, so this file intentionally carries no platforms marker.
PLIST = {
    "Label": "ai.hermes.gateway",
    "ProgramArguments": ["/usr/local/bin/hermes", "gateway", "run"],
    "EnvironmentVariables": {"HERMES_HOME": "/Users/tester/.hermes", "PYTHONUNBUFFERED": "1"},
    "RunAtLoad": True,
}


class _FakeGateway:
    """Minimal stand-in for the gateway module the real shim resolves via ``_gw()``."""

    def __init__(self, plist_path, generated_xml: str):
        self._plist_path = plist_path
        self._generated_xml = generated_xml

    def get_launchd_plist_path(self):
        return self._plist_path

    def generate_launchd_plist(self) -> str:
        return self._generated_xml


def _generated_xml(plist: dict) -> str:
    return plistlib.dumps(plist, fmt=plistlib.FMT_XML).decode("utf-8")


def _write(path, data: dict, binary: bool):
    fmt = plistlib.FMT_BINARY if binary else plistlib.FMT_XML
    path.write_bytes(plistlib.dumps(data, fmt=fmt))
    return path


@pytest.fixture
def install(monkeypatch, tmp_path):
    """Write ``installed`` to disk, have ``generated`` be what Hermes would emit."""

    def _install(installed: dict, generated: dict, binary: bool):
        path = _write(tmp_path / "ai.hermes.gateway.plist", installed, binary)
        monkeypatch.setattr(
            gateway_launchd, "_gw", lambda: _FakeGateway(path, _generated_xml(generated))
        )
        return path

    return _install


def test_binary_plist_does_not_crash(install):
    """The regression: a bplist00 install used to raise UnicodeDecodeError."""
    install(PLIST, PLIST, binary=True)
    assert gateway_launchd.launchd_plist_is_current() is True


def test_xml_plist_is_still_current(install):
    install(PLIST, PLIST, binary=False)
    assert gateway_launchd.launchd_plist_is_current() is True


def test_reordered_and_reindented_plist_is_current(install):
    """Serialization differences must not read as stale.

    ``plistlib`` round-trips change indentation, key order, and the XML
    declaration, so a text comparison would flag every valid install and
    rewrite it on every check.
    """
    reordered = {
        "RunAtLoad": True,
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "HERMES_HOME": "/Users/tester/.hermes",
        },
        "ProgramArguments": ["/usr/local/bin/hermes", "gateway", "run"],
        "Label": "ai.hermes.gateway",
    }
    install(reordered, PLIST, binary=True)
    assert gateway_launchd.launchd_plist_is_current() is True


def test_path_env_var_is_exempt(install):
    """PATH is captured from the invoking shell and legitimately varies."""
    installed = {
        **PLIST,
        "EnvironmentVariables": {**PLIST["EnvironmentVariables"], "PATH": "/opt/homebrew/bin"},
    }
    install(installed, PLIST, binary=True)
    assert gateway_launchd.launchd_plist_is_current() is True


def test_installed_and_generated_are_distinct_objects(install):
    """Guards the fixture: a shared object would hide a comparison bug."""
    path = install(PLIST, PLIST, binary=True)
    generated = plistlib.loads(_generated_xml(PLIST).encode())
    installed = plistlib.loads(path.read_bytes())
    assert installed == generated
    assert installed is not generated


def test_changed_env_var_is_stale(install):
    installed = {
        **PLIST,
        "EnvironmentVariables": {**PLIST["EnvironmentVariables"], "HERMES_HOME": "/tmp/other"},
    }
    install(installed, PLIST, binary=True)
    assert gateway_launchd.launchd_plist_is_current() is False


def test_changed_program_arguments_is_stale(install):
    installed = {**PLIST, "ProgramArguments": ["/usr/local/bin/hermes", "gateway", "start"]}
    install(installed, PLIST, binary=False)
    assert gateway_launchd.launchd_plist_is_current() is False


def test_garbage_file_is_stale_not_a_crash(install):
    path = install(PLIST, PLIST, binary=True)
    path.write_bytes(b"not a plist at all")
    assert gateway_launchd.launchd_plist_is_current() is False


def test_malformed_xml_is_stale_not_a_crash(install):
    """ExpatError is not a ValueError subclass, so it needs its own handler."""
    path = install(PLIST, PLIST, binary=False)
    path.write_text(
        '<?xml version="1.0"?><plist version="1.0"><dict><key>Label</key></dict>',
        encoding="utf-8",
    )
    assert gateway_launchd.launchd_plist_is_current() is False


def test_empty_file_is_stale_not_a_crash(install):
    path = install(PLIST, PLIST, binary=True)
    path.write_bytes(b"")
    assert gateway_launchd.launchd_plist_is_current() is False


def test_missing_file_is_stale(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gateway_launchd, "_gw", lambda: _FakeGateway(tmp_path / "absent.plist", _generated_xml(PLIST))
    )
    assert gateway_launchd.launchd_plist_is_current() is False