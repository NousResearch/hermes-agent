"""Interactive GitHub Copilot device-code login."""

from __future__ import annotations

import time
from typing import Optional

from auth.providers.copilot import (
    COPILOT_OAUTH_CLIENT_ID,
    _DEVICE_CODE_POLL_INTERVAL,
    _DEVICE_CODE_POLL_SAFETY_MARGIN,
    _DEVICE_CODE_TERMINAL_ERRORS,
    _post_form,
    logger,
)


def copilot_device_code_login(
    *, host: str = "github.com", timeout_seconds: float = 300) -> Optional[str]:
    """Run the GitHub OAuth device code flow for Copilot."""
    domain = host.rstrip("/")
    try:
        device_data = _post_form(f"https://{domain}/login/device/code",
                                 {"client_id": COPILOT_OAUTH_CLIENT_ID, "scope": "read:user"}, 15)
    except Exception as exc:
        logger.error("Failed to initiate device authorization: %s", exc)
        print(f"  ✗ Failed to start device authorization: {exc}")
        return None

    verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
    user_code = device_data.get("user_code", "")
    device_code = device_data.get("device_code", "")
    interval = max(device_data.get("interval", _DEVICE_CODE_POLL_INTERVAL), 1)
    if not device_code or not user_code:
        print("  ✗ GitHub did not return a device code.")
        return None
    print(f"\n  Open this URL in your browser: {verification_uri}\n"
          f"  Enter this code: {user_code}\n")
    print("  Waiting for authorization...", end="", flush=True)
    poll_fields = {"client_id": COPILOT_OAUTH_CLIENT_ID, "device_code": device_code,
                   "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        time.sleep(interval + _DEVICE_CODE_POLL_SAFETY_MARGIN)
        try:
            result = _post_form(f"https://{domain}/login/oauth/access_token", poll_fields, 10)
        except Exception:
            print(".", end="", flush=True)
            continue
        if result.get("access_token"):
            print(" ✓")
            return result["access_token"]
        error = result.get("error", "")
        if error == "slow_down":
            # RFC 8628: add 5 seconds to polling interval (or honor a server-supplied one)
            server_interval = result.get("interval")
            is_num = isinstance(server_interval, (int, float)) and server_interval > 0
            interval = int(server_interval) if is_num else interval + 5
        if error in ("authorization_pending", "slow_down"):
            print(".", end="", flush=True)
            continue
        if error:
            print("\n" + _DEVICE_CODE_TERMINAL_ERRORS.get(error,
                                                       f"  ✗ Authorization failed: {error}"))
            return None
    print("\n  ✗ Timed out waiting for authorization.")
    return None
