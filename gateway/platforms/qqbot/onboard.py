"""QQBot scan-to-configure (QR code onboard). Mirrors the Feishu pattern: synchronous HTTP + one entry-point
``qr_register()`` (create task → display QR → poll → decrypt) against ``q.qq.com`` bind-task endpoints."""

from __future__ import annotations

import logging
import time
from enum import IntEnum
from typing import Optional, Tuple
from urllib.parse import quote

from .constants import (
    ONBOARD_API_TIMEOUT, ONBOARD_CREATE_PATH, ONBOARD_POLL_INTERVAL, ONBOARD_POLL_PATH, PORTAL_HOST, QR_URL_TEMPLATE)
from .crypto import decrypt_secret, generate_bind_key
from .utils import get_api_headers

logger = logging.getLogger(__name__)


class BindStatus(IntEnum):
    """Status codes returned by ``_poll_bind_result``."""
    NONE, PENDING, COMPLETED, EXPIRED = 0, 1, 2, 3


try:
    import qrcode as _qrcode_mod
except (ImportError, TypeError):
    _qrcode_mod = None  # type: ignore[assignment]


def _render_qr(url: str) -> bool:
    """Try to render a QR code in the terminal. Returns True if successful."""
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode(error_correction=_qrcode_mod.constants.ERROR_CORRECT_M, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


def _portal_post(path: str, payload: dict, timeout: float, fail_msg: str) -> dict:
    """Synchronous POST to the portal host; raises RuntimeError on non-zero ``retcode``."""
    import httpx

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.post(f"https://{PORTAL_HOST}{path}", json=payload, headers=get_api_headers())
        resp.raise_for_status()
        data = resp.json()
    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", fail_msg))
    return data


def _create_bind_task(timeout: float = ONBOARD_API_TIMEOUT) -> Tuple[str, str]:
    """Create a bind task and return *(task_id, aes_key_base64)*."""
    key = generate_bind_key()
    data = _portal_post(ONBOARD_CREATE_PATH, {"key": key}, timeout, "create_bind_task failed")
    task_id = (data.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError("create_bind_task: missing task_id in response")
    logger.debug("create_bind_task ok: task_id=%s", task_id)
    return task_id, key


def _poll_bind_result(task_id: str, timeout: float = ONBOARD_API_TIMEOUT) -> Tuple[BindStatus, str, str, str]:
    """Poll *task_id*; returns ``(status, bot_appid, bot_encrypt_secret, user_openid)``."""
    d = _portal_post(ONBOARD_POLL_PATH, {"task_id": task_id}, timeout, "poll_bind_result failed").get("data", {})
    return (BindStatus(d.get("status", 0)), str(d.get("bot_appid", "")), d.get("bot_encrypt_secret", ""),
            d.get("user_openid", ""))


def build_connect_url(task_id: str) -> str:
    """Build the QR-code target URL for a given *task_id*."""
    return QR_URL_TEMPLATE.format(task_id=quote(task_id))


_MAX_REFRESHES = 3


def qr_register(timeout_seconds: int = 600) -> Optional[dict]:
    """Run the QR registration flow; returns ``{"app_id", "client_secret", "user_openid"}``
    or None on failure / expiry / cancellation. Unexpected errors propagate."""
    deadline = time.monotonic() + timeout_seconds
    for refresh_count in range(_MAX_REFRESHES + 1):
        try:
            task_id, aes_key = _create_bind_task()
        except Exception as exc:
            logger.warning("[QQBot onboard] Failed to create bind task: %s", exc)
            return None
        url = build_connect_url(task_id)
        print()
        if _render_qr(url):
            print(f"  Scan the QR code above, or open this URL directly:\n  {url}")
        else:
            print(f"  Open this URL in QQ on your phone:\n  {url}")
            print("  Tip: pip install qrcode  to display a scannable QR code here")
        print()
        while time.monotonic() < deadline:
            try:
                status, app_id, encrypted_secret, user_openid = _poll_bind_result(task_id)
            except Exception:
                time.sleep(ONBOARD_POLL_INTERVAL)
                continue
            if status == BindStatus.COMPLETED:
                client_secret = decrypt_secret(encrypted_secret, aes_key)
                print()
                print(f"  QR scan complete! (App ID: {app_id})")
                if user_openid:
                    print(f"  Scanner's OpenID: {user_openid}")
                return {"app_id": app_id, "client_secret": client_secret, "user_openid": user_openid}
            if status == BindStatus.EXPIRED:
                if refresh_count >= _MAX_REFRESHES:
                    logger.warning("[QQBot onboard] QR code expired %d times — giving up", _MAX_REFRESHES)
                    return None
                print(f"\n  QR code expired, refreshing... ({refresh_count + 1}/{_MAX_REFRESHES})")
                break  # next for-loop iteration creates a new task
            time.sleep(ONBOARD_POLL_INTERVAL)
        else:
            logger.warning("[QQBot onboard] Poll timed out after %ds", timeout_seconds)
            return None
    return None
