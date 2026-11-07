"""Tests for launchd plist generation when HERMES_HOME is on an external volume.

macOS launchd xpcproxy lacks Removable Volumes TCC entitlements, so StandardOutPath
and StandardErrorPath must route to ~/Library/Logs instead of /Volumes/...
"""

from __future__ import annotations

from pathlib import Path
import plistlib

import hermes_cli.gateway as gateway_cli


def test_launchd_plist_routes_stdio_to_library_logs_on_external_volume(monkeypatch):
    """When log_dir resolves to /Volumes/..., outer launchd stdio routes to ~/Library/Logs."""
    external_home = Path("/Volumes/ExternalMockVolume/.hermes")
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: external_home)
    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/usr/bin/python3")
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda *a, **k: Path("/usr/bin/python3"))
    label = gateway_cli.get_launchd_label()
    parsed = plistlib.loads(gateway_cli.generate_launchd_plist().encode("utf-8"))
    assert parsed["StandardOutPath"] == str(Path.home() / "Library" / "Logs" / f"{label}.log")
    assert parsed["StandardErrorPath"] == str(Path.home() / "Library" / "Logs" / f"{label}.error.log")
