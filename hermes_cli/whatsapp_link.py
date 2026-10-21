"""Decide whether a WhatsApp session is actually linked.

``creds.json`` on disk is not a live link. WhatsApp can log the device out
and leave that file behind; the dashboard must not report Linked from the
file alone.
"""

from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

_LIVE_BRIDGE_STATUSES = {"needs_qr", "disconnected", "connecting"}


def fetch_whatsapp_bridge_health(port: int = 3000, timeout: float = 2) -> Optional[dict[str, Any]]:
    """GET the bridge ``/health`` payload, or None when the bridge is not answering."""
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def whatsapp_onboarding_link_decision(creds_exist: bool, health: Optional[dict[str, Any]]) -> str:
    """What starting a QR link should do.

    ``linked`` — the bridge reports a live connection. Keep the session.
    ``watch`` — a bridge is up but not linked (including ``needs_qr``). Show
    its QR; do not open a second Baileys socket on the same session.
    ``clear_and_pair`` — the bridge says the device is logged out and is not
    serving a QR. Delete the session, then pair.
    ``pair`` — nothing is answering. Start the pairing bridge, which deletes
    the session itself if that attempt is logged out.
    """
    status = ""
    if isinstance(health, dict):
        status = str(health.get("status") or "").strip().lower()
    if status == "connected" and creds_exist:
        return "linked"
    if status == "connected":
        return "watch"
    if status == "logged_out":
        return "clear_and_pair"
    if status in _LIVE_BRIDGE_STATUSES:
        return "watch"
    return "pair"


def discard_whatsapp_session(session_path: Path) -> None:
    """Delete a logged-out session directory. Do not rename or copy it."""
    if session_path.exists():
        shutil.rmtree(session_path)
