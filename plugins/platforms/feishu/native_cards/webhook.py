"""保留原生卡片回调响应；认证、限流、请求体约束与上游一致。"""

from __future__ import annotations
import asyncio
import hmac
import json
import logging
from typing import Any
import lark_oapi as lark
from aiohttp import web
from plugins.platforms.feishu.adapter import (
    _read_limited_feishu_webhook_body,
    _FEISHU_WEBHOOK_MAX_BODY_BYTES,
    _FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class WebhookMixin:
    async def _handle_webhook_request(self, request: Any) -> Any:
        remote_ip = getattr(request, "remote", None) or "unknown"

        # Rate-limit key is app_id:path:remote_ip (matches openclaw key structure).
        if not self._check_webhook_rate_limit(f"{self._app_id}:{self._webhook_path}:{remote_ip}"):
            logger.warning("[Feishu] Webhook rate limit exceeded for %s", remote_ip)
            return self._webhook_reject(remote_ip, "429", 429, "Too Many Requests")

        headers = getattr(request, "headers", {}) or {}
        content_type = str(headers.get("Content-Type", "") or "").split(";")[0].strip().lower()
        if content_type and content_type != "application/json":  # Feishu always sends JSON
            logger.warning(
                "[Feishu] Webhook rejected: unexpected Content-Type %r from %s",
                content_type,
                remote_ip,
            )
            return self._webhook_reject(remote_ip, "415", 415, "Unsupported Media Type")

        content_length = getattr(request, "content_length", None)
        if content_length is not None and content_length > _FEISHU_WEBHOOK_MAX_BODY_BYTES:
            logger.warning(
                "[Feishu] Webhook body too large (%d bytes) from %s", content_length, remote_ip
            )
            return self._webhook_reject(remote_ip, "413", 413, "Request body too large")

        try:
            body_bytes: bytes = await asyncio.wait_for(
                _read_limited_feishu_webhook_body(request, _FEISHU_WEBHOOK_MAX_BODY_BYTES),
                timeout=_FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS,
            )
        except ValueError:
            logger.warning("[Feishu] Webhook body exceeds limit from %s", remote_ip)
            return self._webhook_reject(remote_ip, "413", 413, "Request body too large")
        except TimeoutError:
            logger.warning(
                "[Feishu] Webhook body read timed out after %ds from %s",
                _FEISHU_WEBHOOK_BODY_TIMEOUT_SECONDS,
                remote_ip,
            )
            return self._webhook_reject(remote_ip, "408", 408, "Request Timeout")
        except Exception:
            logger.warning("Failed to read Feishu webhook body", exc_info=True)
            return self._webhook_reject(remote_ip, "400", 400, json_msg="failed to read body")

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._webhook_reject(remote_ip, "400", 400, json_msg="invalid json")

        # Verification token: second defence layer beyond the signature (matches openclaw).
        if self._verification_token:
            header = payload.get("header") or {}
            incoming_token = str(header.get("token") or payload.get("token") or "")
            # compare_digest as bytes — it raises TypeError on non-ASCII str, and the token is remote input.
            if not incoming_token or not hmac.compare_digest(
                incoming_token.encode(), self._verification_token.encode()
            ):
                logger.warning(
                    "[Feishu] Webhook rejected: invalid verification token from %s", remote_ip
                )
                return self._webhook_reject(
                    remote_ip, "401-token", 401, "Invalid verification token"
                )

        # Token is validated above BEFORE reflecting the challenge, so an unauthenticated
        # remote can't prove endpoint control by getting its own challenge echoed back.
        if payload.get("type") == "url_verification":
            return web.json_response({"challenge": payload.get("challenge", "")})

        if self._encrypt_key and not self._is_webhook_signature_valid(request.headers, body_bytes):
            logger.warning("[Feishu] Webhook rejected: invalid signature from %s", remote_ip)
            return self._webhook_reject(remote_ip, "401-sig", 401, "Invalid signature")

        if payload.get("encrypt"):
            logger.error(
                "[Feishu] Encrypted webhook payloads are not supported by Hermes webhook mode"
            )
            return self._webhook_reject(
                remote_ip,
                "400-encrypted",
                400,
                json_msg="encrypted webhook payloads are not supported",
            )

        self._clear_webhook_anomaly(remote_ip)

        event_type = str((payload.get("header") or {}).get("event_type") or "")
        if event_type == "card.action.trigger":
            # Authentication is already checked above. SDK decoding preserves
            # action.value/form_value as mappings (namespace conversion does not).
            response = self._build_event_handler()._do_without_validation(body_bytes)
            return web.json_response(
                json.loads(lark.JSON.marshal(response)) if response is not None else {}
            )
        data = self._namespace_from_mapping(payload)
        if event_type in {"im.message.reaction.created_v1", "im.message.reaction.deleted_v1"}:
            self._on_reaction_event(event_type, data)
        else:
            handler = self._WEBHOOK_EVENT_HANDLERS.get(event_type)
            if handler is None:
                logger.debug("[Feishu] Ignoring webhook event type: %s", event_type or "unknown")
            else:
                getattr(self, handler)(data)
        return web.json_response({"code": 0, "msg": "ok"})
