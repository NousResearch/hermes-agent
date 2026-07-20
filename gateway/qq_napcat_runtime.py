"""Helpers for diagnosing local QQ NapCat runtime availability."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

_LOCAL_QQ_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _display_path(path: Path) -> str:
    try:
        return str(path).replace(str(Path.home()), "~", 1)
    except Exception:
        return str(path)


def _expected_napcat_paths() -> list[Path]:
    home = Path.home()
    return [
        home / ".config/QQ/NapCat",
        home / ".local/share/LiteLoaderQQNT/plugins/NapCat",
        Path("/opt/QQ/resources/app/app_launcher/napcat"),
        Path("/opt/QQ/resources/app/LiteLoaderQQNT/plugins/NapCat"),
    ]


def _existing_napcat_paths() -> list[Path]:
    return [path for path in _expected_napcat_paths() if path.exists()]


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def diagnose_local_qq_napcat_endpoint(ws_url: str) -> Optional[dict[str, str]]:
    """Return a human-readable local-runtime diagnosis when a local WS endpoint is down."""
    parsed = urlsplit(str(ws_url or "").strip())
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port
    if parsed.scheme not in {"ws", "wss"} or host not in _LOCAL_QQ_HOSTS or port is None:
        return None
    if _port_is_open(host, port):
        return None

    found_paths = _existing_napcat_paths()
    if found_paths:
        details = ", ".join(_display_path(path) for path in found_paths)
        return {
            "code": "qq_napcat_local_service_offline",
            "message": (
                f"QQ NapCat local service is offline: {ws_url} is not listening, "
                f"but NapCat files still exist at {details}. Start or restart NapCat, "
                "then restart Hermes gateway."
            ),
        }

    expected = ", ".join(_display_path(path) for path in _expected_napcat_paths())
    return {
        "code": "qq_napcat_runtime_missing",
        "message": (
            f"QQ NapCat local runtime is missing: {ws_url} is not listening, and no NapCat "
            f"install/config was found at {expected}. Reinstall or restore NapCat, then "
            "restart Hermes gateway."
        ),
    }
