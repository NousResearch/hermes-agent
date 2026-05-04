from __future__ import annotations

import json
from typing import Any

from hermes_android.python_path import prefer_hermes_package_root

prefer_hermes_package_root()

from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL, get_nous_auth_status


def read_nous_portal_state() -> dict[str, Any]:
    status = get_nous_auth_status()
    portal_url = str(status.get("portal_base_url") or DEFAULT_NOUS_PORTAL_URL).strip() or DEFAULT_NOUS_PORTAL_URL
    inference_url = str(status.get("inference_base_url") or "").strip()
    return {
        "portal_url": portal_url,
        "logged_in": bool(status.get("logged_in")),
        "inference_url": inference_url,
    }


def read_nous_portal_state_json() -> str:
    return json.dumps(read_nous_portal_state())
