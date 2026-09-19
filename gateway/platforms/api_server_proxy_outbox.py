"""Thin HTTP and outbound-delivery glue for the proxy outbox."""

import asyncio
import logging
import re
from typing import Any, Optional

from gateway.platforms.base import SendResult


logger = logging.getLogger("gateway.platforms.api_server")


async def send_for_platform(
    self,
    platform: Any,
    chat_id: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
    *,
    _api_server: Any,
) -> Any:
    from gateway.proxy_outbox import enqueue

    try:
        delivery_id = await asyncio.to_thread(
            enqueue,
            platform=platform,
            chat_id=chat_id,
            content=content,
            metadata=metadata,
        )
        delivered, error = await self._wait_for_proxy_delivery(delivery_id)
        return _api_server.SendResult(
            success=delivered,
            message_id=delivery_id,
            error=None if delivered else error,
        )
    except Exception as exc:
        logger.warning("Could not queue proxy delivery: %s", type(exc).__name__)
        return _api_server.SendResult(
            success=False,
            error=_api_server._redact_api_error_text(exc),
        )


async def wait_for_delivery(
    delivery_id: str,
    *,
    timeout: float = 620.0,
    poll_interval: float = 0.1,
) -> tuple[bool, Optional[str]]:
    """Poll ACK state without monopolising the shared blocking-I/O pool."""
    from gateway.proxy_outbox import delivery_result, fail_pending

    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, float(timeout))
    poll_interval = max(0.01, min(float(poll_interval), 1.0))
    while True:
        result = await asyncio.to_thread(delivery_result, delivery_id)
        if result is not None:
            return result
        remaining = deadline - loop.time()
        if remaining <= 0:
            error = "delivery confirmation timed out"
            await asyncio.to_thread(fail_pending, delivery_id, error)
            return False, error
        await asyncio.sleep(min(poll_interval, remaining))


async def handle_proxy_outbox(self, request: Any, *, _api_server: Any) -> Any:
    web = _api_server.web
    assert web is not None
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    try:
        limit = int(request.query.get("limit", "4"))
        requested = {
            value.strip().lower()
            for value in request.query.get("platforms", "").split(",")
            if value.strip()
        }
    except ValueError:
        return web.json_response(_api_server._openai_error("Invalid limit"), status=400)
    if not requested:
        return web.json_response(
            _api_server._openai_error("platforms is required"), status=400
        )
    try:
        from gateway.proxy_outbox import lease

        items = await asyncio.to_thread(lease, platforms=requested, limit=limit)
        return web.json_response(
            {"object": "hermes.proxy.outbox", "data": items},
            headers={"Cache-Control": "no-store"},
        )
    except Exception as exc:
        logger.exception("Proxy outbox lease failed")
        return web.json_response(
            _api_server._openai_error(_api_server._redact_api_error_text(exc)),
            status=500,
        )


async def handle_proxy_outbox_ack(self, request: Any, *, _api_server: Any) -> Any:
    web = _api_server.web
    assert web is not None
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    delivery_id = request.match_info.get("delivery_id", "")
    if not re.fullmatch(r"[0-9a-f]{32}", delivery_id):
        return web.json_response(
            _api_server._openai_error("Invalid delivery id"), status=400
        )
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            _api_server._openai_error("Invalid JSON body"), status=400
        )
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("success"), bool)
        or type(body.get("attempt")) is not int
        or body["attempt"] < 1
    ):
        return web.json_response(
            _api_server._openai_error(
                "success must be a boolean and attempt a positive integer"
            ),
            status=400,
        )
    from gateway.proxy_outbox import acknowledge

    accepted = await asyncio.to_thread(
        acknowledge,
        delivery_id,
        attempt=body["attempt"],
        success=body["success"],
        error=_api_server._redact_api_error_text(body.get("error", ""), limit=500),
    )
    if not accepted:
        return web.json_response(
            _api_server._openai_error("Delivery is not actively leased"), status=409
        )
    return web.json_response({"delivery_id": delivery_id, "acknowledged": True})


class ProxyOutboxAPIMixin:
    def fronts_platform(self, platform: Any) -> bool:
        """Return whether this API listener queues output for a thin gateway."""
        from gateway.proxy_outbox import fronts_platform

        return fronts_platform(platform)

    async def send_for_platform(
        self,
        platform: Any,
        chat_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        from gateway.platforms import api_server as _api_server

        return await send_for_platform(
            self,
            platform,
            chat_id,
            content,
            metadata,
            _api_server=_api_server,
        )

    @staticmethod
    async def _wait_for_proxy_delivery(
        delivery_id: str,
        *,
        timeout: float = 620.0,
        poll_interval: float = 0.1,
    ) -> tuple[bool, Optional[str]]:
        return await wait_for_delivery(
            delivery_id,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    async def _handle_proxy_outbox(self, request: Any) -> Any:
        from gateway.platforms import api_server as _api_server

        return await handle_proxy_outbox(
            self,
            request,
            _api_server=_api_server,
        )

    async def _handle_proxy_outbox_ack(self, request: Any) -> Any:
        from gateway.platforms import api_server as _api_server

        return await handle_proxy_outbox_ack(
            self,
            request,
            _api_server=_api_server,
        )
