"""ActivePieces platform adapter: AP automation triggers wake the agent.

ActivePieces (self-hosted AP, e.g. the meeting pod's in-pod instance) runs
flows whose triggers — Gmail new email, generic webhooks, Twilio SMS — must
wake the agent exactly like a Telegram message. Rather than duplicating the
webhook receiver's pipeline, this adapter REUSES it: subclass of
:class:`WebhookAdapter` (auth → rate-limit → parse → event/payload filters →
route script → prompt render → skills attach → agent dispatch → delivery,
``gateway/platforms/webhook.py::_handle_webhook``), registered as its own
first-class ``Platform.ACTIVEPIECES`` the same way
:class:`~gateway.platforms.msgraph_webhook.MSGraphWebhookAdapter` is.

Only the ActivePieces-specific edges live here:

- **Bearer auth.** An AP flow's HTTP-request piece sets STATIC headers; it
  cannot compute an HMAC. ``Authorization: Bearer <secret>`` is compared
  timing-safely against the route (or global) secret; the inherited HMAC
  schemes still work for flows that use a code piece. No secret → the route
  never starts (inherited fail-closed validation).
- **Reply delivery — ``deliver: "activepieces"``.** The agent's final
  response is POSTed as ``{delivery_id, flow, reply}`` to the route's
  ``deliver_extra.url`` (trusted route config, never payload-derived) with
  optional static ``deliver_extra.headers``. That URL is the AP-native
  reply channel: a Webhook-Response/HTTP step in the flow's reply leg.
- **Loopback default.** AP events arrive over the pod's control-doc bridge
  (session-api control event → app task_loop → loopback POST), so the
  listener binds 127.0.0.1 unless ``extra.host`` says otherwise.

Envelope contract (shared with the cold-start/Twilio consumers): the POST
body is ``{source: "activepieces", piece, trigger, project_id, event: <raw
trigger payload>}``; per-flow skills attach via the route's ``skills`` key
and pieces are distinguished with payload filters (``source``/``piece``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from aiohttp import ClientSession, ClientTimeout

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - aiohttp is a hard requirement, same as the base
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _hmac_str_equal
from gateway.response_filters import is_autonomous_silence_response

logger = logging.getLogger(__name__)

#: Pod-loopback by default: AP reaches this adapter through the app's
#: control-doc bridge, never from a public network.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8647
_DELIVER_TYPE = "activepieces"
_REPLY_TIMEOUT_SECONDS = 15


def check_activepieces_requirements() -> bool:
    """Check if ActivePieces adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class ActivePiecesAdapter(WebhookAdapter):
    """Receive ActivePieces flow events and surface them to the agent."""

    _session_id_prefix = "activepieces"
    _source_label = "activepieces"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.ACTIVEPIECES)
        extra = config.extra
        # Flows use the same route schema as webhook routes (secret, events,
        # filters, script, prompt, skills, toolsets, deliver, deliver_extra)
        # keyed by the AP flow name; static-only — AP flows are provisioned
        # alongside the config, not agent-created subscriptions.
        flows = extra.get("flows", {})
        self._static_routes: Dict[str, dict] = {str(k): v for k, v in flows.items()} if isinstance(flows, dict) else {}
        self._routes: Dict[str, dict] = dict(self._static_routes)
        self._host: Optional[str] = str(extra.get("host") or DEFAULT_HOST)
        self._port: int = int(extra.get("port", DEFAULT_PORT))

    def _validate_signature(self, request, body: bytes, secret: str) -> bool:
        """Bearer first (what AP flows can actually send), then the inherited HMAC schemes."""
        auth = request.headers.get("Authorization", "") or ""
        if auth.startswith("Bearer "):
            return _hmac_str_equal(auth[len("Bearer "):].strip(), secret)
        return super()._validate_signature(request, body, secret)

    async def send(self, chat_id: str, content: str, reply_to: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Agent-mode delivery: ``deliver: activepieces`` POSTs the reply back into AP."""
        if is_autonomous_silence_response(content):
            logger.info("[activepieces] Response for %s is a silence marker — not delivering", chat_id)
            return SendResult(success=True)
        delivery = self._delivery_info.get(chat_id, {})
        if delivery.get("deliver") == _DELIVER_TYPE:
            return await self._deliver_activepieces(content, delivery, chat_id)
        return await super().send(chat_id, content, reply_to, metadata)

    def _validate_route(self, name: str, route: dict) -> None:
        """Inherited secret checks, plus: ``deliver: activepieces`` needs an http(s) URL."""
        super()._validate_route(name, route)
        if route.get("deliver") == _DELIVER_TYPE:
            url = str((route.get("deliver_extra") or {}).get("url", "")).strip()
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"[activepieces] Flow '{name}' sets deliver='{_DELIVER_TYPE}' but deliver_extra.url "
                    f"is not an http(s) URL: {url!r}")

    async def _direct_deliver(self, content: str, delivery: dict) -> SendResult:
        """deliver_only parity for the activepieces target."""
        if delivery.get("deliver") == _DELIVER_TYPE:
            return await self._deliver_activepieces(
                content, delivery, f"{self._session_id_prefix}:{delivery.get('route', '')}:"
                                   f"{delivery.get('delivery_id', '')}")
        return await super()._direct_deliver(content, delivery)

    async def _deliver_activepieces(self, content: str, delivery: dict, chat_id: str) -> SendResult:
        """POST the agent reply to the flow's reply URL ({delivery_id, flow, reply})."""
        extra = delivery.get("deliver_extra", {})
        url = str(extra.get("url", "")).strip()
        if not url.startswith(("http://", "https://")):
            logger.error("[activepieces] Reply URL missing or not http(s) for %s", chat_id)
            return SendResult(success=False, error="Reply URL not configured")
        headers = {str(k): str(v) for k, v in (extra.get("headers") or {}).items()}
        body = {"delivery_id": chat_id.rsplit(":", 1)[-1], "flow": delivery.get("route", ""), "reply": content}
        try:
            async with ClientSession(timeout=ClientTimeout(total=_REPLY_TIMEOUT_SECONDS)) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if 200 <= resp.status < 300:
                        logger.info("[activepieces] Reply delivered for %s -> %s (HTTP %d)",
                                    chat_id, url, resp.status)
                        return SendResult(success=True)
                    detail = (await resp.text())[:200]
                    logger.warning("[activepieces] Reply target rejected %s -> %s (HTTP %d): %s",
                                   chat_id, url, resp.status, detail)
                    return SendResult(success=False, error=f"HTTP {resp.status}")
        except Exception as e:  # a dead reply target must never fail the completed run
            logger.warning("[activepieces] Reply delivery failed for %s -> %s: %s", chat_id, url, e)
            return SendResult(success=False, error=str(e))
