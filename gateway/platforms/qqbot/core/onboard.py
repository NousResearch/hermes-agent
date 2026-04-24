# -*- coding: utf-8 -*-
"""QQBot scan-to-configure (QR code onboard) module.

Calls the ``q.qq.com`` ``create_bind_task`` / ``poll_bind_result`` APIs to
generate a QR-code URL and poll for scan completion.  On success the caller
receives the bot's *app_id*, *client_secret* (decrypted locally), and the
scanner's *user_openid*.

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import IntEnum
from typing import Optional, Tuple
from urllib.parse import quote

from .constants import (
    ONBOARD_API_TIMEOUT,
    ONBOARD_CREATE_PATH,
    ONBOARD_POLL_INTERVAL,
    ONBOARD_POLL_PATH,
    PORTAL_HOST,
    QR_URL_TEMPLATE,
)
from .crypto import decrypt_secret, generate_bind_key
from .utils import get_api_headers

logger = logging.getLogger(__name__)

_MAX_REFRESHES = 3

try:
    import qrcode as _qrcode_mod
except ImportError:
    _qrcode_mod = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Bind status
# ---------------------------------------------------------------------------

class BindStatus(IntEnum):
    """Status codes returned by ``poll_bind_result``."""

    NONE = 0
    PENDING = 1
    COMPLETED = 2
    EXPIRED = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_bind_task(
    timeout: float = ONBOARD_API_TIMEOUT,
) -> Tuple[str, str]:
    """Create a bind task and return *(task_id, aes_key_base64)*.

    The AES key is generated locally and sent to the server so it can
    encrypt the bot credentials before returning them.

    :param timeout: HTTP request timeout in seconds.
    :returns: ``(task_id, aes_key_base64)`` tuple.
    :raises RuntimeError: If the API returns a non-zero ``retcode``.
    """
    import httpx

    url = f"https://{PORTAL_HOST}{ONBOARD_CREATE_PATH}"
    key = generate_bind_key()

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.post(url, json={"key": key}, headers=get_api_headers())
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "create_bind_task failed"))

    task_id = data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError("create_bind_task: missing task_id in response")

    logger.debug("create_bind_task ok: task_id=%s", task_id)
    return task_id, key


async def poll_bind_result(
    task_id: str,
    timeout: float = ONBOARD_API_TIMEOUT,
) -> Tuple[BindStatus, str, str, str]:
    """Poll the bind result for *task_id*.

    :param task_id: Task ID from :func:`create_bind_task`.
    :param timeout: HTTP request timeout in seconds.
    :returns: ``(status, bot_appid, bot_encrypt_secret, user_openid)``.
        ``bot_encrypt_secret`` is AES-256-GCM encrypted — decrypt it with
        :func:`~gateway.platforms.qqbot.core.crypto.decrypt_secret`.
    :raises RuntimeError: If the API returns a non-zero ``retcode``.
    """
    import httpx

    url = f"https://{PORTAL_HOST}{ONBOARD_POLL_PATH}"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.post(
            url,
            json={"task_id": task_id},
            headers=get_api_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

    if data.get("retcode") != 0:
        raise RuntimeError(data.get("msg", "poll_bind_result failed"))

    d = data.get("data", {})
    return (
        BindStatus(d.get("status", 0)),
        str(d.get("bot_appid", "")),
        d.get("bot_encrypt_secret", ""),
        d.get("user_openid", ""),
    )


def build_connect_url(task_id: str) -> str:
    """Build the QR-code target URL for *task_id*.

    :param task_id: Task ID from :func:`create_bind_task`.
    :returns: Full HTTPS URL to embed in a QR code.
    """
    return QR_URL_TEMPLATE.format(task_id=quote(task_id))


# ---------------------------------------------------------------------------
# Interactive QR registration (used by hermes_cli/gateway.py)
# ---------------------------------------------------------------------------

def _render_qr(url: str) -> bool:
    """Try to render a QR code in the terminal.

    :returns: ``True`` if the QR code was rendered successfully.
    """
    if _qrcode_mod is None:
        return False
    try:
        qr = _qrcode_mod.QRCode(
            error_correction=_qrcode_mod.constants.ERROR_CORRECT_M,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        return True
    except Exception:
        return False


def qr_register(timeout_seconds: int = 600) -> Optional[dict]:
    """Run the QQBot scan-to-configure QR registration flow.

    Mirrors ``feishu.qr_register()``: handles create → display → poll →
    decrypt in one synchronous call.  Unexpected errors propagate to the
    caller.

    :param timeout_seconds: Total seconds before giving up.
    :returns: ``{"app_id": ..., "client_secret": ..., "user_openid": ...}``
        on success, or ``None`` on failure / expiry / cancellation.
    """
    deadline = time.monotonic() + timeout_seconds

    for refresh_count in range(_MAX_REFRESHES + 1):
        # ── Create bind task ──
        try:
            task_id, aes_key = asyncio.run(create_bind_task())
        except Exception as exc:
            logger.warning("[QQBot onboard] Failed to create bind task: %s", exc)
            return None

        url = build_connect_url(task_id)

        # ── Display QR code + URL ──
        print()
        if _render_qr(url):
            print(f"  Scan the QR code above, or open this URL directly:\n  {url}")
        else:
            print(f"  Open this URL in QQ on your phone:\n  {url}")
            print("  Tip: pip install qrcode  to display a scannable QR code here")
        print()

        # ── Poll loop ──
        while time.monotonic() < deadline:
            try:
                status, app_id, encrypted_secret, user_openid = asyncio.run(
                    poll_bind_result(task_id)
                )
            except Exception:
                time.sleep(ONBOARD_POLL_INTERVAL)
                continue

            if status == BindStatus.COMPLETED:
                client_secret = decrypt_secret(encrypted_secret, aes_key)
                print()
                print(f"  QR scan complete! (App ID: {app_id})")
                if user_openid:
                    print(f"  Scanner's OpenID: {user_openid}")
                return {
                    "app_id": app_id,
                    "client_secret": client_secret,
                    "user_openid": user_openid,
                }

            if status == BindStatus.EXPIRED:
                if refresh_count >= _MAX_REFRESHES:
                    logger.warning(
                        "[QQBot onboard] QR code expired %d times — giving up",
                        _MAX_REFRESHES,
                    )
                    return None
                print(f"\n  QR code expired, refreshing... ({refresh_count + 1}/{_MAX_REFRESHES})")
                break  # next for-loop iteration creates a new task

            time.sleep(ONBOARD_POLL_INTERVAL)
        else:
            # deadline reached without completing
            logger.warning("[QQBot onboard] Poll timed out after %ds", timeout_seconds)
            return None

    return None
