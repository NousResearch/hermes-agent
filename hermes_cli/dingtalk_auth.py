"""
DingTalk Device Flow authorization.

Implements the same 3-step registration flow as dingtalk-openclaw-connector:
  1. POST /app/registration/init   → get nonce
  2. POST /app/registration/begin  → get device_code + verification_uri_complete
  3. POST /app/registration/poll   → poll until SUCCESS → get client_id + client_secret

The verification_uri_complete is rendered as a QR code in the terminal so the
user can scan it with DingTalk to authorize, yielding AppKey + AppSecret
automatically.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from typing import Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

REGISTRATION_BASE_URL = os.environ.get(
    "DINGTALK_REGISTRATION_BASE_URL", "https://oapi.dingtalk.com"
).rstrip("/")

REGISTRATION_SOURCE = os.environ.get("DINGTALK_REGISTRATION_SOURCE", "openClaw")


# ── API helpers ────────────────────────────────────────────────────────────

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
        errmsg = data.get("errmsg", "unknown error")
        raise RegistrationError(f"API error [{path}]: {errmsg} (errcode={errcode})")
    return data


# ── Core flow ──────────────────────────────────────────────────────────────

def begin_registration() -> dict:
    """Start a device-flow registration.

    Returns a dict with keys:
        device_code, verification_uri_complete, expires_in, interval
    """
    # Step 1: init → nonce
    init_data = _api_post("/app/registration/init", {"source": REGISTRATION_SOURCE})
    nonce = str(init_data.get("nonce", "")).strip()
    if not nonce:
        raise RegistrationError("init response missing nonce")

    # Step 2: begin → device_code, verification_uri_complete
    begin_data = _api_post("/app/registration/begin", {"nonce": nonce})
    device_code = str(begin_data.get("device_code", "")).strip()
    verification_uri_complete = str(begin_data.get("verification_uri_complete", "")).strip()
    if not device_code:
        raise RegistrationError("begin response missing device_code")
    if not verification_uri_complete:
        raise RegistrationError("begin response missing verification_uri_complete")

    return {
        "device_code": device_code,
        "verification_uri_complete": verification_uri_complete,
        "expires_in": int(begin_data.get("expires_in", 7200)),
        "interval": max(int(begin_data.get("interval", 3)), 2),
    }


def poll_registration(device_code: str) -> dict:
    """Poll the registration status once.

    Returns a dict with keys:  status, client_id?, client_secret?, fail_reason?
    """
    data = _api_post("/app/registration/poll", {"device_code": device_code})
    status_raw = str(data.get("status", "")).strip().upper()
    if status_raw not in ("WAITING", "SUCCESS", "FAIL", "EXPIRED"):
        status_raw = "UNKNOWN"
    return {
        "status": status_raw,
        "client_id": str(data.get("client_id", "")).strip() or None,
        "client_secret": str(data.get("client_secret", "")).strip() or None,
        "fail_reason": str(data.get("fail_reason", "")).strip() or None,
    }


def wait_for_registration_success(
    device_code: str,
    interval: int = 3,
    expires_in: int = 7200,
    on_waiting: Optional[callable] = None,
) -> Tuple[str, str]:
    """Block until the registration succeeds or times out.

    Returns (client_id, client_secret).
    """
    deadline = time.monotonic() + expires_in
    retry_window = 120  # 2 minutes for transient errors
    retry_start = 0.0

    while time.monotonic() < deadline:
        time.sleep(interval)
        try:
            result = poll_registration(device_code)
        except RegistrationError:
            if retry_start == 0:
                retry_start = time.monotonic()
            if time.monotonic() - retry_start < retry_window:
                continue
            raise

        status = result["status"]
        if status == "WAITING":
            retry_start = 0
            if on_waiting:
                on_waiting()
            continue
        if status == "SUCCESS":
            cid = result["client_id"]
            csecret = result["client_secret"]
            if not cid or not csecret:
                raise RegistrationError("authorization succeeded but credentials are missing")
            return cid, csecret
        # FAIL / EXPIRED / UNKNOWN
        if retry_start == 0:
            retry_start = time.monotonic()
        if time.monotonic() - retry_start < retry_window:
            continue
        reason = result.get("fail_reason") or status
        raise RegistrationError(f"authorization failed: {reason}")

    raise RegistrationError("authorization timed out, please retry")


# ── QR code rendering ─────────────────────────────────────────────────────

def _ensure_qrcode_installed() -> bool:
    """Try to import qrcode; if missing, auto-install it via pip/uv."""
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        pass

    import subprocess

    # Try uv first (Hermes convention), then pip
    for cmd in (
        [sys.executable, "-m", "uv", "pip", "install", "qrcode"],
        [sys.executable, "-m", "pip", "install", "-q", "qrcode"],
    ):
        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import qrcode  # noqa: F401,F811
            return True
        except (subprocess.CalledProcessError, ImportError, FileNotFoundError):
            continue
    return False


def render_qr_to_terminal(url: str) -> bool:
    """Render *url* as a compact QR code in the terminal.

    Returns True if the QR code was printed, False if the library is missing.
    """
    try:
        import qrcode
    except ImportError:
        return False

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)

    # Use half-block characters for compact rendering (2 rows per character)
    matrix = qr.get_matrix()
    rows = len(matrix)
    lines: list[str] = []

    TOP_HALF = "\u2580"      # ▀
    BOTTOM_HALF = "\u2584"   # ▄
    FULL_BLOCK = "\u2588"    # █
    EMPTY = " "

    for r in range(0, rows, 2):
        line_chars: list[str] = []
        for c in range(len(matrix[r])):
            top = matrix[r][c]
            bottom = matrix[r + 1][c] if r + 1 < rows else False
            if top and bottom:
                line_chars.append(FULL_BLOCK)
            elif top:
                line_chars.append(TOP_HALF)
            elif bottom:
                line_chars.append(BOTTOM_HALF)
            else:
                line_chars.append(EMPTY)
        lines.append("    " + "".join(line_chars))

    print("\n".join(lines))
    return True


# ── High-level entry point for the setup wizard ───────────────────────────

def dingtalk_qr_auth() -> Optional[Tuple[str, str]]:
    """Run the interactive QR-code device-flow authorization.

    Returns (client_id, client_secret) on success, or None if the user
    cancelled or the flow failed.
    """
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

    # Ensure qrcode library is available (auto-install if missing)
    if not _ensure_qrcode_installed():
        print_warning("  qrcode library install failed, will show link only.")

    print()
    print_info("  Please scan the QR code below with DingTalk to authorize:")
    print()

    if not render_qr_to_terminal(url):
        print_warning(f"  QR code render failed, please open the link below to authorize:")

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
            device_code=reg["device_code"],
            interval=reg["interval"],
            expires_in=reg["expires_in"],
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
