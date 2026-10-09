"""Hermes seam over qqbot-agent-sdk.

The SDK owns the QQ socket, REST client, media upload, and QR onboard.
This module only binds the calling profile onto the SDK's WebSocket thread:
``run_coroutine_threadsafe`` copies that thread's context onto the gateway
loop, and a fresh thread does not inherit contextvars unless we copy them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def scoped_websocket_class():
    """``QQWebSocket`` subclass whose thread runs in the context captured at ``start``."""
    from qqbot_agent_sdk import QQWebSocket

    class ScopedQQWebSocket(QQWebSocket):
        def start(self, gateway_url: str, main_loop: asyncio.AbstractEventLoop) -> None:
            from agent.memory_provider import spawn_context_thread

            self._main_loop = main_loop
            self._ws_thread = spawn_context_thread(
                self._run_ws_thread,
                name=f"qqbot-ws-{self._log_tag}",
                args=(gateway_url,),
            )
            self._ws_thread.start()

    return ScopedQQWebSocket


def qr_register(timeout_seconds: int = 300) -> Optional[dict]:
    """Scan-to-configure via the SDK. Returns credential dict, or None on failure."""
    try:
        from qqbot_agent_sdk import OnboardError, configure, start_onboard
    except ImportError:
        logger.warning("[QQBot onboard] qqbot-agent-sdk is not installed")
        return None

    configure(source="hermes")

    def _show_qr(url: str) -> None:
        print()
        rendered = False
        try:
            import qrcode

            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
            rendered = True
        except Exception:
            logger.debug("[QQBot onboard] QR render failed; printing the URL", exc_info=True)
            rendered = False
        if rendered:
            print(f"  Scan the QR code above, or open this URL directly:\n  {url}")
        else:
            print(f"  Open this URL in QQ on your phone:\n  {url}")
        print()

    async def _run():
        return await start_onboard(on_qr_ready=_show_qr, poll_timeout=float(timeout_seconds))

    try:
        result = asyncio.run(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        raise
    except OnboardError as exc:
        logger.warning("[QQBot onboard] %s", exc)
        return None
    except Exception:
        logger.warning("[QQBot onboard] QR setup failed", exc_info=True)
        return None

    print()
    print(f"  QR scan complete! (App ID: {result.app_id})")
    if result.user_openid:
        print(f"  Scanner's OpenID: {result.user_openid}")
    return {
        "app_id": result.app_id,
        "client_secret": result.client_secret,
        "user_openid": result.user_openid or "",
    }
