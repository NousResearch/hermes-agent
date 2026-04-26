"""Feishu OAuth 2.0 Device Flow authorization.

Implements RFC 8628 (Device Authorization Grant) for Feishu/Lark user identity:
  1. POST https://accounts.feishu.cn/oauth/v1/device_authorization
     → get device_code + verification_uri_complete + expires_in + interval
  2. Render verification_uri_complete as ASCII QR code in terminal
  3. POST https://open.feishu.cn/open-apis/authen/v2/oauth/token (poll)
     → get access_token + refresh_token on user scan-and-approve

Tokens are persisted to ~/.hermes/feishu_uat.json (mode 0600).

Entry point for ``hermes setup feishu-uat``:  feishu_qr_auth(client_id)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

from hermes_constants import display_hermes_home, get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEISHU_ACCOUNTS_BASE_URL = os.environ.get(
    "FEISHU_ACCOUNTS_BASE_URL", "https://accounts.feishu.cn"
).rstrip("/")

FEISHU_OPEN_BASE_URL = os.environ.get(
    "FEISHU_OPEN_BASE_URL", "https://open.feishu.cn"
).rstrip("/")

# Default scope for UAT — covers core OAPI tool families.
# WARNING: im:message:send_as_user is intentionally excluded from defaults —
# it is a privileged scope. Pass it explicitly via --scope if needed.
# TODO(worker-3): update feishu-uat-tools.md setup command to reflect this change.
FEISHU_DEFAULT_SCOPE = (
    "calendar:calendar "
    "drive:drive "
    "docs:document:readonly "
    "docs:document "
    "bitable:app "
    "wiki:wiki:readonly "
    "sheets:spreadsheet "
    "task:task:write "
    "task:task:read "
    "contact:user.base:readonly"
)

# Path to persisted UAT token file
FEISHU_UAT_PATH = get_hermes_home() / "feishu_uat.json"

# Polling backoff cap in seconds (RFC 8628 §3.5)
_POLL_INTERVAL_CAP = 30


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class FeishuAuthError(Exception):
    """Raised when a Feishu OAuth API call fails or the flow cannot complete."""


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    (re.compile(r"Bearer\s+\S+"), "[REDACTED]"),
    (re.compile(r"access_token=\S+"), "access_token=[REDACTED]"),
    (re.compile(r"device_code=\S+"), "device_code=[REDACTED]"),
    (re.compile(r"user_code=\S+"), "user_code=[REDACTED]"),
    (re.compile(r"refresh_token=\S+"), "refresh_token=[REDACTED]"),
]


def _safe_error_text(exc: BaseException) -> str:
    """Return str(exc) with sensitive token/code values redacted."""
    text = str(exc)
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _api_post(path: str, base_url: str, payload: dict) -> dict:
    """POST to a Feishu endpoint and return the parsed JSON body.

    Args:
        path: URL path (e.g. '/oauth/v1/device_authorization').
        base_url: Base URL of the Feishu endpoint host.
        payload: JSON body dict to send.

    Returns:
        Parsed response JSON as a dict.

    Raises:
        FeishuAuthError: On network errors or non-retryable API errors.
    """
    url = f"{base_url}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise FeishuAuthError(f"Network error calling {url}: {exc}") from exc

    # RFC 6749 / Feishu error model: error field present means failure
    # "authorization_pending" and "slow_down" are handled by the caller
    error_code = data.get("error") or data.get("error_code")
    if error_code and error_code not in ("authorization_pending", "slow_down"):
        description = data.get("error_description", "unknown error")
        raise FeishuAuthError(
            f"API error [{path}]: {description} (error={error_code})"
        )
    return data


# ---------------------------------------------------------------------------
# Step 1: request device code
# ---------------------------------------------------------------------------

def begin_device_authorization(
    client_id: str,
    scope: Optional[str] = None,
) -> dict:
    """Start a Feishu device-flow authorization.

    Args:
        client_id: Feishu app ID (FEISHU_APP_ID).
        scope: Space-separated OAuth scopes. Defaults to FEISHU_DEFAULT_SCOPE.

    Returns:
        Dict with keys: device_code, user_code, verification_uri,
        verification_uri_complete, expires_in, interval.

    Raises:
        FeishuAuthError: If the API call fails or required fields are missing.
    """
    payload: dict = {"client_id": client_id}
    if scope:
        payload["scope"] = scope
    else:
        payload["scope"] = FEISHU_DEFAULT_SCOPE

    data = _api_post(
        "/oauth/v1/device_authorization",
        FEISHU_ACCOUNTS_BASE_URL,
        payload,
    )

    required = [
        "device_code",
        "user_code",
        "verification_uri",
        "verification_uri_complete",
        "expires_in",
        "interval",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        raise FeishuAuthError(
            f"device_authorization response missing fields: {', '.join(missing)}"
        )

    return {
        "device_code": str(data["device_code"]).strip(),
        "user_code": str(data["user_code"]).strip(),
        "verification_uri": str(data["verification_uri"]).strip(),
        "verification_uri_complete": str(data["verification_uri_complete"]).strip(),
        "expires_in": int(data.get("expires_in", 1800)),
        "interval": max(int(data.get("interval", 3)), 2),
    }


# ---------------------------------------------------------------------------
# Step 3: poll for token
# ---------------------------------------------------------------------------

def poll_device_token(device_code: str, client_id: str) -> dict:
    """Poll the Feishu token endpoint once for a device code grant.

    Args:
        device_code: The device_code from begin_device_authorization().
        client_id: Feishu app ID.

    Returns:
        Dict with keys: access_token?, refresh_token?, open_id?,
        error?, error_description?

    Raises:
        FeishuAuthError: On non-retryable API errors.
    """
    payload = {
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "client_id": client_id,
        "device_code": device_code,
    }

    # _api_post raises on hard errors; authorization_pending / slow_down pass through
    try:
        data = _api_post(
            "/open-apis/authen/v2/oauth/token",
            FEISHU_OPEN_BASE_URL,
            payload,
        )
    except FeishuAuthError:
        raise

    return {
        "access_token": str(data.get("access_token", "")).strip() or None,
        "refresh_token": str(data.get("refresh_token", "")).strip() or None,
        "open_id": str(data.get("open_id", "")).strip() or None,
        "expires_in": int(data.get("expires_in", 7200)),
        "refresh_expires_in": int(data.get("refresh_expires_in", 2592000)),
        "token_type": str(data.get("token_type", "Bearer")).strip(),
        "scope": str(data.get("scope", "")).strip(),
        "error": str(data.get("error") or data.get("error_code", "")).strip() or None,
        "error_description": str(data.get("error_description", "")).strip() or None,
    }


# ---------------------------------------------------------------------------
# Polling loop
# ---------------------------------------------------------------------------

def wait_for_authorization_success(
    device_code: str,
    client_id: str,
    interval: int = 3,
    expires_in: int = 1800,
    on_waiting: Optional[callable] = None,
) -> Tuple[str, str, str, int, int]:
    """Block until Feishu device authorization succeeds or times out.

    Args:
        device_code: Device code from begin_device_authorization().
        client_id: Feishu app ID.
        interval: Initial poll interval in seconds.
        expires_in: Total timeout in seconds.
        on_waiting: Optional callback invoked on each pending poll iteration.

    Returns:
        Tuple of (access_token, refresh_token, open_id, expires_in, refresh_expires_in).

    Raises:
        FeishuAuthError: On authorization failure, denial, or timeout.
    """
    deadline = time.monotonic() + expires_in
    # 2-minute transient-error tolerance window
    retry_window = 120.0
    retry_start = 0.0
    current_interval = interval

    while time.monotonic() < deadline:
        time.sleep(current_interval)

        try:
            result = poll_device_token(device_code, client_id)
        except FeishuAuthError:
            if retry_start == 0.0:
                retry_start = time.monotonic()
            if time.monotonic() - retry_start < retry_window:
                continue
            raise

        error = result.get("error")

        # Still waiting — user hasn't scanned yet
        if not error or error == "authorization_pending":
            retry_start = 0.0
            if on_waiting:
                on_waiting()
            continue

        # Server requests slower polling
        if error == "slow_down":
            current_interval = min(current_interval + 5, _POLL_INTERVAL_CAP)
            retry_start = 0.0
            if on_waiting:
                on_waiting()
            continue

        # Success — tokens present
        if result.get("access_token"):
            token = result["access_token"]
            refresh = result.get("refresh_token") or ""
            open_id = result.get("open_id") or ""
            tok_expires_in = int(result.get("expires_in") or 7200)
            tok_refresh_expires_in = int(result.get("refresh_expires_in") or 2592000)
            return token, refresh, open_id, tok_expires_in, tok_refresh_expires_in

        # Authorization explicitly denied or expired
        if retry_start == 0.0:
            retry_start = time.monotonic()
        if time.monotonic() - retry_start < retry_window:
            continue
        description = result.get("error_description") or error
        raise FeishuAuthError(f"authorization failed: {description}")

    raise FeishuAuthError("authorization timed out — please retry 'hermes setup feishu-auth'")


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------

def save_uat(
    access_token: str,
    refresh_token: str,
    open_id: str,
    expires_in: int,
    refresh_expires_in: int,
    scope: str,
    app_id: str,
) -> None:
    """Persist UAT tokens to ~/.hermes/feishu_uat.json (mode 0600).

    Args:
        access_token: Feishu user access token.
        refresh_token: Feishu refresh token.
        open_id: User open_id.
        expires_in: access_token TTL in seconds.
        refresh_expires_in: refresh_token TTL in seconds.
        scope: Granted OAuth scopes string.
        app_id: Feishu app ID that obtained these tokens.
    """
    now_ms = int(time.time() * 1000)
    token_data = {
        "app_id": app_id,
        "user_open_id": open_id,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": now_ms + expires_in * 1000,
        "refresh_expires_at": now_ms + refresh_expires_in * 1000,
        "scope": scope,
        "granted_at": now_ms,
    }
    parent = FEISHU_UAT_PATH.parent
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(parent, 0o700)
    # Atomic write: write to a temp file then os.replace to avoid partial reads
    fd, tmp_path = tempfile.mkstemp(dir=parent)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(token_data, fh, indent=2)
        os.replace(tmp_path, FEISHU_UAT_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    logger.info("Feishu UAT saved to %s/%s", display_hermes_home(), "feishu_uat.json")


def load_uat() -> Optional[dict]:
    """Load stored UAT from ~/.hermes/feishu_uat.json.

    Returns:
        Token dict or None if the file is missing or unreadable.
    """
    if not FEISHU_UAT_PATH.exists():
        return None
    try:
        with open(FEISHU_UAT_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load feishu UAT from %s/%s: %s", display_hermes_home(), "feishu_uat.json", exc)
        return None


def refresh_uat(client_id: str, client_secret: str) -> None:
    """Attempt to refresh the stored UAT using its refresh_token.

    On success, persists new tokens via save_uat() (atomic write).
    On 4xx or expired refresh_token, removes the stale token file and raises
    NeedAuthorizationError so the caller knows re-authorization is required.

    Args:
        client_id: Feishu app ID (FEISHU_APP_ID).
        client_secret: Feishu app secret (FEISHU_APP_SECRET).

    Raises:
        NeedAuthorizationError: If refresh fails or token file is missing.
        FeishuAuthError: On non-auth network/API errors.
    """
    from tools.feishu_oapi_client import NeedAuthorizationError

    data = load_uat()
    if not data:
        raise NeedAuthorizationError(reason="no token file; run 'hermes feishu-auth' first")

    refresh_token = data.get("refresh_token", "")
    if not refresh_token:
        raise NeedAuthorizationError(reason="no refresh_token in stored UAT")

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    try:
        resp_data = _api_post(
            "/open-apis/authen/v2/oauth/token",
            FEISHU_OPEN_BASE_URL,
            payload,
        )
    except FeishuAuthError as exc:
        # Treat refresh failure as expired — clean up and require re-auth
        logger.warning("refresh_uat failed: %s", _safe_error_text(exc))
        try:
            os.remove(FEISHU_UAT_PATH)
        except OSError:
            pass
        raise NeedAuthorizationError(
            user_open_id=data.get("user_open_id", "unknown"),
            reason="refresh_token expired or invalid; re-run 'hermes feishu-auth'",
        ) from exc

    new_access_token = str(resp_data.get("access_token", "")).strip()
    new_refresh_token = str(resp_data.get("refresh_token", "")).strip()
    if not new_access_token:
        try:
            os.remove(FEISHU_UAT_PATH)
        except OSError:
            pass
        raise NeedAuthorizationError(
            user_open_id=data.get("user_open_id", "unknown"),
            reason="refresh response missing access_token; re-run 'hermes feishu-auth'",
        )

    save_uat(
        access_token=new_access_token,
        refresh_token=new_refresh_token or refresh_token,
        open_id=data.get("user_open_id", ""),
        expires_in=int(resp_data.get("expires_in") or 7200),
        refresh_expires_in=int(resp_data.get("refresh_expires_in") or 2592000),
        scope=str(resp_data.get("scope", "")).strip() or data.get("scope", ""),
        app_id=data.get("app_id", client_id),
    )
    logger.info("Feishu UAT refreshed for user %s", data.get("user_open_id", "unknown"))


# ---------------------------------------------------------------------------
# QR code rendering (copied pattern from dingtalk_auth.py)
# ---------------------------------------------------------------------------

def _ensure_qrcode_installed() -> bool:
    """Try to import qrcode; auto-install via uv/pip if missing.

    Returns:
        True if qrcode is available after the call.
    """
    try:
        import qrcode  # noqa: F401
        return True
    except ImportError:
        pass

    import subprocess

    for cmd in (
        [sys.executable, "-m", "uv", "pip", "install", "qrcode"],
        [sys.executable, "-m", "pip", "install", "-q", "qrcode"],
    ):
        try:
            subprocess.check_call(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            import qrcode  # noqa: F401,F811
            return True
        except (subprocess.CalledProcessError, ImportError, FileNotFoundError):
            continue
    return False


def render_qr_to_terminal(url: str) -> bool:
    """Render *url* as a compact half-block QR code in the terminal.

    Args:
        url: The URL to encode as a QR code.

    Returns:
        True if the QR code was printed, False if qrcode is unavailable.
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

    matrix = qr.get_matrix()
    rows = len(matrix)
    lines: list[str] = []

    TOP_HALF = "▀"    # ▀
    BOTTOM_HALF = "▄" # ▄
    FULL_BLOCK = "█"  # █
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


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def feishu_qr_auth(
    client_id: str,
    scope: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Run the interactive QR-code Feishu device-flow authorization.

    Args:
        client_id: Feishu app ID (FEISHU_APP_ID env var value).
        scope: Override OAuth scopes. Defaults to FEISHU_DEFAULT_SCOPE.

    Returns:
        (access_token, refresh_token) on success, or None on failure/cancel.
    """
    from hermes_cli.setup import print_info, print_success, print_warning, print_error

    print()
    print_info("  Initializing Feishu user authorization (OAuth device flow)...")

    try:
        auth_data = begin_device_authorization(client_id, scope)
    except FeishuAuthError as exc:
        print_error(f"  Authorization init failed: {exc}")
        return None

    url = auth_data["verification_uri_complete"]
    user_code = auth_data["user_code"]

    if not _ensure_qrcode_installed():
        print_warning("  qrcode library install failed, will show link only.")

    print()
    print_info("  Please scan the QR code below with Feishu to authorize:")
    if user_code:
        print_info(f"  User Code: {user_code}")
    print()

    if not render_qr_to_terminal(url):
        print_warning("  QR code render failed, please open the link below:")

    print()
    print_info(f"  Or open this link manually: {url}")
    print()
    print_info("  Waiting for authorization... (timeout: 30 minutes)")

    dot_count = 0

    def _on_waiting() -> None:
        nonlocal dot_count
        dot_count += 1
        if dot_count % 10 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    try:
        access_token, refresh_token, open_id, tok_expires_in, tok_refresh_expires_in = (
            wait_for_authorization_success(
                device_code=auth_data["device_code"],
                client_id=client_id,
                interval=auth_data["interval"],
                expires_in=auth_data["expires_in"],
                on_waiting=_on_waiting,
            )
        )
    except FeishuAuthError as exc:
        print()
        print_error(f"  Authorization failed: {exc}")
        return None

    # Persist tokens using real expires_in values from the token response
    try:
        save_uat(
            access_token=access_token,
            refresh_token=refresh_token,
            open_id=open_id,
            expires_in=tok_expires_in,
            refresh_expires_in=tok_refresh_expires_in,
            scope=scope or FEISHU_DEFAULT_SCOPE,
            app_id=client_id,
        )
    except OSError as exc:
        print_warning(f"  Token save failed: {exc}")

    print()
    print_success("  Feishu user authorization successful!")
    print_success(f"  Open ID:      {open_id or '(not returned)'}")
    print_success(
        f"  Access Token: {access_token[:8]}{'*' * max(0, len(access_token) - 8)}"
    )
    print_success(f"  Tokens saved to: {display_hermes_home()}/feishu_uat.json")

    return access_token, refresh_token


# ---------------------------------------------------------------------------
# CLI entry point (called by hermes_cli/main.py cmd_feishu_auth_setup)
# ---------------------------------------------------------------------------

def cmd_feishu_auth_setup(args) -> None:
    """Handle ``hermes feishu-auth`` CLI command.

    Previously named cmd_feishu_uat_setup. The subcommand was renamed from
    feishu-uat to feishu-auth for consistency with dingtalk-auth naming.

    Args:
        args: Parsed argparse namespace.
    """
    from hermes_cli.config import get_env_value
    from hermes_cli.setup import print_info, print_success, print_warning, print_error

    client_id = get_env_value("FEISHU_APP_ID") or ""
    if not client_id:
        print_error(
            "  FEISHU_APP_ID is not set. Run 'hermes setup' first to configure"
            " the Feishu bot credentials before authorizing user identity."
        )
        return

    # Check for existing token
    existing = load_uat()
    if existing:
        now_ms = int(time.time() * 1000)
        expires_at = existing.get("expires_at", 0)
        refresh_expires_at = existing.get("refresh_expires_at", 0)
        open_id = existing.get("user_open_id", "(unknown)")

        if now_ms < expires_at:
            remaining_min = (expires_at - now_ms) // 60000
            print_success(
                f"  Feishu UAT already valid for user {open_id}"
                f" (~{remaining_min} min remaining)."
            )
        elif now_ms < refresh_expires_at:
            print_warning(
                f"  Access token expired for user {open_id},"
                " but refresh token is still valid."
            )
        else:
            print_warning(f"  All tokens expired for user {open_id}.")

        scope = getattr(args, "scope", None) or ""
        force = getattr(args, "force", False)
        if not force:
            try:
                from hermes_cli.setup import prompt_yes_no
                if not prompt_yes_no("  Re-authorize Feishu user identity?", False):
                    return
            except Exception:
                print_info("  Run with --force to re-authorize.")
                return
    else:
        scope = getattr(args, "scope", None) or ""

    scope = scope.strip() or None
    result = feishu_qr_auth(client_id=client_id, scope=scope)
    if result is None:
        import sys
        sys.exit(1)
