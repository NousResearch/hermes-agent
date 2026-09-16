"""DingTalk Device Flow authorization."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Optional, Tuple

import requests

REGISTRATION_BASE_URL = os.environ.get("DINGTALK_REGISTRATION_BASE_URL", "https://oapi.dingtalk.com").rstrip("/")
REGISTRATION_SOURCE = os.environ.get("DINGTALK_REGISTRATION_SOURCE", "openClaw")
_POLL_STATUSES = {"WAITING", "SUCCESS", "FAIL", "EXPIRED"}
_RETRY_WINDOW = 120  # seconds of transient errors / non-success statuses tolerated before giving up


class RegistrationError(Exception):
    """Raised when a DingTalk registration API call fails."""


def _api_post(path: str, payload: dict) -> dict:
    """POST to the registration API and return the parsed JSON body."""
    url = f"{REGISTRATION_BASE_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RegistrationError(f"Network error calling {url}: {exc}") from exc
    errcode = data.get("errcode", -1)
    if errcode != 0:
        raise RegistrationError(f"API error [{path}]: {data.get('errmsg', 'unknown error')} (errcode={errcode})")
    return data


def begin_registration() -> dict:
    """Start a device-flow registration: init → nonce, begin → device_code + verification URL."""
    nonce = str(_api_post("/app/registration/init", {"source": REGISTRATION_SOURCE}).get("nonce", "")).strip()
    if not nonce:
        raise RegistrationError("init response missing nonce")
    begin_data = _api_post("/app/registration/begin", {"nonce": nonce})
    reg = {key: str(begin_data.get(key, "")).strip() for key in ("device_code", "verification_uri_complete")}
    for key, value in reg.items():
        if not value:
            raise RegistrationError(f"begin response missing {key}")
    reg["expires_in"] = int(begin_data.get("expires_in", 7200))
    reg["interval"] = max(int(begin_data.get("interval", 3)), 2)
    return reg


def poll_registration(device_code: str) -> dict:
    """Poll the registration status once."""
    data = _api_post("/app/registration/poll", {"device_code": device_code})
    status_raw = str(data.get("status", "")).strip().upper()
    result = {"status": status_raw if status_raw in _POLL_STATUSES else "UNKNOWN"}
    result.update({key: str(data.get(key, "")).strip() or None for key in ("client_id", "client_secret", "fail_reason")})
    return result


def wait_for_registration_success(
    device_code: str, interval: int = 3, expires_in: int = 7200, on_waiting: Optional[callable] = None,
) -> Tuple[str, str]:
    """Block until the registration succeeds or times out.

    Transient errors and FAIL/EXPIRED/UNKNOWN statuses are retried for ``_RETRY_WINDOW`` seconds
    before being raised; a WAITING status resets that window.
    """
    deadline = time.monotonic() + expires_in
    retry_start = 0.0

    def _within_retry_window() -> bool:
        nonlocal retry_start
        if retry_start == 0:
            retry_start = time.monotonic()
        return time.monotonic() - retry_start < _RETRY_WINDOW

    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            result = poll_registration(device_code)
        except RegistrationError:
            if _within_retry_window():
                continue
            raise
        status = result["status"]
        if status == "WAITING":
            retry_start = 0
            if on_waiting:
                on_waiting()
        elif status == "SUCCESS":
            cid, csecret = result["client_id"], result["client_secret"]
            if not cid or not csecret:
                raise RegistrationError("authorization succeeded but credentials are missing")
            return cid, csecret
        elif not _within_retry_window():
            raise RegistrationError(f"authorization failed: {result.get('fail_reason') or status}")
    raise RegistrationError("authorization timed out, please retry")


def _ensure_qrcode_installed() -> bool:
    """Try to import qrcode; if missing, auto-install it via pip/uv."""
    with contextlib.suppress(ImportError):
        import qrcode  # noqa: F401
        return True
    import subprocess
    from hermes_cli.tools_config import _pip_install
    with contextlib.suppress(subprocess.SubprocessError, ImportError, OSError):
        if _pip_install(["-q", "qrcode"], timeout=120).returncode == 0:
            import qrcode  # noqa: F401,F811
            return True
    return False


def render_qr_to_terminal(url: str) -> bool:
    """Render *url* as a compact QR code (half-block glyphs, 2 rows per character) in the terminal."""
    try:
        import qrcode
    except ImportError:
        return False
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=1, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    matrix = qr.get_matrix()
    glyph = {(True, True): "\u2588", (True, False): "\u2580", (False, True): "\u2584", (False, False): " "}  # █ ▀ ▄ space
    lines = []
    for r in range(0, len(matrix), 2):
        bottom_row = matrix[r + 1] if r + 1 < len(matrix) else [False] * len(matrix[r])
        lines.append("    " + "".join(glyph[(bool(top), bool(bottom))] for top, bottom in zip(matrix[r], bottom_row)))
    print("\n".join(lines))
    return True


def dingtalk_qr_auth() -> Optional[Tuple[str, str]]:
    """Run the interactive QR-code device-flow authorization (setup wizard entry point)."""
    from hermes_cli.setup import print_info, print_success, print_warning, print_error
    print()
    print_info("  Initializing DingTalk device authorization...")
    print_info("  Note: the scan page is branded 'OpenClaw' — DingTalk's")
    print_info("        ecosystem onboarding bridge. Safe to use.")
    try:
        reg = begin_registration()
    except RegistrationError as exc:
        print_error(f"  Authorization init failed: {exc}")
        return None
    url = reg["verification_uri_complete"]
    if not _ensure_qrcode_installed():
        print_warning("  qrcode library install failed, will show link only.")
    print()
    print_info("  Please scan the QR code below with DingTalk to authorize:")
    print()
    if not render_qr_to_terminal(url):
        print_warning("  QR code render failed, please open the link below to authorize:")
    print()
    print_info(f"  Or open this link manually: {url}")
    print()
    print_info("  Waiting for QR scan authorization... (timeout: 2 hours)")
    dot_count = 0

    def _on_waiting():
        nonlocal dot_count
        dot_count += 1
        if dot_count % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    try:
        client_id, client_secret = wait_for_registration_success(
            device_code=reg["device_code"], interval=reg["interval"], expires_in=reg["expires_in"],
            on_waiting=_on_waiting,
        )
    except RegistrationError as exc:
        print()
        print_error(f"  Authorization failed: {exc}")
        return None
    print()
    print_success("  QR scan authorization successful!")
    print_success(f"  Client ID:     {client_id}")
    print_success(f"  Client Secret: {client_secret[:8]}{'*' * (len(client_secret) - 8)}")
    return client_id, client_secret


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'logger': ('hermes_cli.auth', 'logger'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
