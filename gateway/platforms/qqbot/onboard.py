"""
QQBot scan-to-configure (QR code onboard) module.

Calls the ``q.qq.com`` ``create_bind_task`` / ``poll_bind_result`` APIs to
generate a QR-code URL and poll for scan completion.  On success the caller
receives the bot's *app_id*, *client_secret* (decrypted locally), and the
scanner's *user_openid* — enough to fully configure the QQBot gateway.

Reference: https://bot.q.qq.com/wiki/develop/api-v2/
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import Tuple
from urllib.parse import quote

from .constants import (
    ONBOARD_API_TIMEOUT,
    ONBOARD_CREATE_PATH,
    ONBOARD_POLL_PATH,
    PORTAL_HOST,
    QR_URL_TEMPLATE,
)
from .crypto import generate_bind_key
from .utils import get_api_headers

logger = logging.getLogger(__name__)


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

    Raises:
        RuntimeError: If the API returns a non-zero ``retcode``.
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

    Returns:
        A 4-tuple of ``(status, bot_appid, bot_encrypt_secret, user_openid)``.

        * ``bot_encrypt_secret`` is AES-256-GCM encrypted — decrypt it with
          :func:`~gateway.platforms.qqbot.crypto.decrypt_secret` using the
          key from :func:`create_bind_task`.
        * ``user_openid`` is the OpenID of the person who scanned the code
          (available when ``status == COMPLETED``).

    Raises:
        RuntimeError: If the API returns a non-zero ``retcode``.
    """
    import httpx

    url = f"https://{PORTAL_HOST}{ONBOARD_POLL_PATH}"

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        resp = await client.post(url, json={"task_id": task_id}, headers=get_api_headers())
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
    """Build the QR-code target URL for a given *task_id*."""
    return QR_URL_TEMPLATE.format(task_id=quote(task_id))
