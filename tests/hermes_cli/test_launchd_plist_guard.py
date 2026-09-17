"""Regression guard: one malformed LaunchAgent plist must not abort the stale-dashboard cleanup.

A hand-edited plist with a raw ``&`` (e.g. ``<string>&&</string>``) is not well-formed XML, so
``plistlib.load`` raises ``xml.parsers.expat.ExpatError`` — which is *not* a ``ValueError``. Before
the guard covered it, a single such plist escaped ``_loaded_launchd_backend_jobs`` and crashed the
whole post-pull update cleanup (``_finish_dashboard_update_cleanup`` ->
``_kill_stale_dashboard_processes`` -> ``_loaded_launchd_backend_jobs``) with a traceback, leaving
the fleet-restart marker set.
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from hermes_cli.main_dashboard import _loaded_launchd_backend_jobs

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="launchd plists are macOS-only")

# Verbatim shape of the plist that crashed a real `hermes update`: `&&` is a raw ampersand.
MALFORMED_PLIST = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
    b'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
    b'<plist version="1.0">\n<dict>\n'
    b'  <key>Label</key>\n  <string>com.example.bad</string>\n'
    b'  <key>ProgramArguments</key>\n  <array>\n'
    b'    <string>/usr/bin/python3</string>\n'
    b'    <string>&&</string>\n'
    b'  </array>\n</dict>\n</plist>\n'
)

# Same document with the ampersand escaped: well-formed, but no runnable hermes backend.
WELLFORMED_PLIST = MALFORMED_PLIST.replace(b"<string>&&</string>", b"<string>&amp;&amp;</string>")


def _write(dir_path: Path, name: str, payload: bytes) -> Path:
    path = dir_path / name
    path.write_bytes(payload)
    return path


def test_fixture_really_is_malformed(tmp_path: Path) -> None:
    """If the fixture parsed, the regression test below would prove nothing."""
    bad = _write(tmp_path, "com.example.bad.plist", MALFORMED_PLIST)
    with open(bad, "rb") as fh:
        with pytest.raises(Exception):
            plistlib.load(fh)


def test_malformed_plist_does_not_escape_the_scanner(tmp_path: Path) -> None:
    _write(tmp_path, "com.example.bad.plist", MALFORMED_PLIST)
    assert _loaded_launchd_backend_jobs([("agent", tmp_path)]) == []


def test_malformed_plist_does_not_hide_wellformed_siblings(tmp_path: Path) -> None:
    _write(tmp_path, "com.example.bad.plist", MALFORMED_PLIST)
    _write(tmp_path, "com.example.good.plist", WELLFORMED_PLIST)
    # Neither job is a loaded dashboard/serve backend, so the scan returns empty rather than raising.
    assert _loaded_launchd_backend_jobs([("agent", tmp_path)]) == []
