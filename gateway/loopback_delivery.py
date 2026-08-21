"""Loopback delivery to the live gateway's api_server.

Out-of-process cron (``hermes cron run``, a separate Chronos worker, etc.)
has no live platform adapters. Relay-fronted logical platforms (Discord /
Slack / … whose credential lives in the Team Gateway connector) have no
native standalone sender either — the only working transport is the live
gateway process's relay websocket.

This module POSTs to the local api_server's ``/api/delivery/send`` route so
delivery reuses that socket instead of opening a second connector handshake
(see issue #86249, fix option 2).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_API_SERVER_PORT = 8642


def resolve_api_server_loopback_base(home: Optional[Path] = None) -> str:
    """Return ``http://127.0.0.1:<port>`` for the local gateway api_server.

    Port resolution mirrors the Chronos fire-forward helper
    (``hermes_cli.web_server._gateway_fire_endpoint``): config.yaml
    ``platforms.api_server.extra.port`` → ``API_SERVER_PORT`` → 8642.
    """
    port = 0
    try:
        from hermes_cli.config import cfg_get, load_config
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        token = None
        if home is not None:
            token = set_hermes_home_override(str(home))
        try:
            cfg = load_config()
        finally:
            if token is not None:
                reset_hermes_home_override(token)
        raw = cfg_get(cfg, "platforms", "api_server", "extra", "port", default=None)
        if raw:
            port = int(raw)
    except Exception:
        port = 0

    if not port:
        raw = os.getenv("API_SERVER_PORT", "").strip()
        try:
            port = int(raw) if raw else 0
        except ValueError:
            port = 0
    if not port:
        port = _DEFAULT_API_SERVER_PORT
    return f"http://127.0.0.1:{port}"


def _api_server_key() -> str:
    """Bearer key for loopback calls (``API_SERVER_KEY`` / config mirror)."""
    key = (os.getenv("API_SERVER_KEY") or "").strip()
    if key:
        return key
    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        raw = cfg_get(cfg, "platforms", "api_server", "extra", "key", default="") or ""
        return str(raw).strip()
    except Exception:
        return ""


def deliver_via_gateway_loopback(
    platform: str,
    chat_id: str,
    content: str,
    *,
    thread_id: Optional[str] = None,
    timeout: float = 60.0,
) -> Optional[str]:
    """Deliver ``content`` through the live gateway on loopback.

    Returns ``None`` on success, or an error string on failure (unreachable
    gateway, auth failure, or upstream send error). Never raises.
    """
    platform_name = (platform or "").strip().lower()
    chat = str(chat_id or "").strip()
    if not platform_name or not chat:
        return "gateway loopback delivery requires platform and chat_id"

    url = f"{resolve_api_server_loopback_base()}/api/delivery/send"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = _api_server_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    payload: dict[str, Any] = {
        "platform": platform_name,
        "chat_id": chat,
        "content": content if isinstance(content, str) else str(content or ""),
    }
    if thread_id is not None and str(thread_id).strip():
        payload["thread_id"] = str(thread_id).strip()

    try:
        import httpx

        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as exc:
        msg = (
            f"gateway loopback unreachable at {url} ({type(exc).__name__}: {exc}). "
            "Is the gateway running with api_server enabled? Relay-fronted "
            "platforms can only deliver from the live gateway process."
        )
        logger.warning(msg)
        return msg

    try:
        body = resp.json()
    except Exception:
        body = {"raw": (resp.text or "")[:500]}
    if not isinstance(body, dict):
        body = {"raw": body}

    if resp.status_code == 200 and body.get("success"):
        return None

    err = body.get("error") or body.get("raw") or f"HTTP {resp.status_code}"
    msg = f"gateway loopback delivery failed: {err}"
    logger.warning(msg)
    return msg
