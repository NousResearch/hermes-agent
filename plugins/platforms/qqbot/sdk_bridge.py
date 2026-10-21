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


def _hermes_version() -> str:
    """Running Hermes version. Package metadata is the ``0.0.0`` placeholder."""
    from hermes_cli.version_info import get_version_info

    return get_version_info().base_version


def apply_sdk_identity() -> None:
    """Stamp ``Hermes/<version>`` on the SDK user agent and ``source=hermes`` on QR URLs.

    The SDK builds ``QQBotAdapter/<sdk> (Python/…; <os>; <extra>)`` at request time.
    REST, the gateway socket, and CDN downloads all read that string.
    """
    from qqbot_agent_sdk import configure

    configure(source="hermes", extra_ua_items=[f"Hermes/{_hermes_version()}"])


def scoped_websocket_class():
    """``QQWebSocket`` subclass whose thread runs in the context captured at ``start``."""
    from qqbot_agent_sdk import QQWebSocket

    class ScopedQQWebSocket(QQWebSocket):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # First open() runs before start_listeners(). A failure there kills the
            # thread with no reconnect and no SDK callback.
            self._listeners_started = False

        def start(self, gateway_url: str, main_loop: asyncio.AbstractEventLoop) -> None:
            from agent.memory_provider import spawn_context_thread

            self._main_loop = main_loop
            self._ws_thread = spawn_context_thread(
                self._run_ws_thread,
                name=f"qqbot-ws-{self._log_tag}",
                args=(gateway_url,),
            )
            self._ws_thread.start()

        def start_listeners(self) -> None:
            self._listeners_started = True
            super().start_listeners()

        def _run_ws_thread(self, gateway_url: str) -> None:
            try:
                super()._run_ws_thread(gateway_url)
            finally:
                self._report_startup_failure()

        def _report_startup_failure(self) -> None:
            """Tell the adapter when the thread died before the listen loop existed."""
            if self._listeners_started or self._stop_requested:
                return
            cb = self._cb
            loop = self._main_loop
            if cb is None or loop is None or loop.is_closed():
                return

            def _notify() -> None:
                try:
                    cb.on_disconnected()
                    cb.on_fatal_error(
                        "qq_gateway_open_failed",
                        "QQ gateway socket exited before the listen loop started",
                    )
                except Exception:
                    logger.warning(
                        "[%s] Startup failure report failed", self._log_tag, exc_info=True)

            try:
                loop.call_soon_threadsafe(_notify)
            except RuntimeError:
                logger.warning(
                    "[%s] Main loop closed before startup failure could be reported",
                    self._log_tag,
                )

    return ScopedQQWebSocket


def qr_register(timeout_seconds: int = 300) -> Optional[dict]:
    """Scan-to-configure via the SDK. Returns credential dict, or None on failure."""
    try:
        from qqbot_agent_sdk import OnboardError, start_onboard
    except ImportError:
        logger.warning("[QQBot onboard] qqbot-agent-sdk is not installed")
        return None

    apply_sdk_identity()

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
