"""Minimum permission-aware device observations for plugin runtimes."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any


def observe_location() -> dict[str, Any]:
    """Observe OS location without IP inference or credential exposure."""
    if platform.system() != "Windows":
        return _unavailable("unsupported_os")
    result = _run_windows_location()
    if result is None:
        return _unavailable("windows_location_api")
    if isinstance(result, str):
        if "denied" in result.lower() or "unauthorized" in result.lower():
            return {
                "status": "denied",
                "permission": "denied",
                "source": "windows_location_api",
            }
        return _unavailable("windows_location_api")
    try:
        return {
            "status": "granted",
            "permission": "os_managed",
            "source": "windows_location_api",
            "latitude": float(result["latitude"]),
            "longitude": float(result["longitude"]),
        }
    except (KeyError, TypeError, ValueError):
        return _unavailable("windows_location_api")


def _run_windows_location() -> dict[str, Any] | str | None:
    script = (
        "Add-Type -AssemblyName System.Device; "
        "$w=New-Object System.Device.Location.GeoCoordinateWatcher; "
        "$ok=$w.TryStart($false,[TimeSpan]::FromMilliseconds(1500)); "
        "if(-not $ok){Write-Error 'Location permission denied'; exit 1}; "
        "$c=$w.Position.Location; "
        "if($c.IsUnknown){exit 2}; "
        "@{latitude=$c.Latitude;longitude=$c.Longitude}|ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=os.environ.copy(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return str(result.stderr or "")
    try:
        return json.loads(result.stdout) if result.stdout.strip() else None
    except json.JSONDecodeError:
        return None


def _unavailable(source: str) -> dict[str, str]:
    return {
        "status": "unavailable",
        "permission": "unavailable",
        "source": source,
    }


__all__ = ["observe_location"]
