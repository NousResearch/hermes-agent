"""Authenticated native gateway ingress and durable chat identity."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
import json
import logging
import re
import time
from typing import Mapping, Optional
from aiohttp import web

logger = logging.getLogger("gateway.platforms.api_server")


@dataclass
class ChatRequestContext:
    body: dict[str, Any]
    stream: bool
    system_prompt: Any
    user_message: Any
    history: list
    session_id: str
    provided_session_id: str
    gateway_session_key: str | None
    gateway_proxy_origin: dict[str, str] | None
    relay_metadata: dict


class GatewayContextMixin:
    def _parse_session_key_header(
        self, request: web.Request) -> tuple[str | None, web.Response | None]:
        """Validate ``X-Hermes-Session-Key`` (per-channel memory scope) -> ``(key_or_None, None)``
        or ``(None, error)``. Requires API-key auth so a client can't guess another scope."""
        from gateway.platforms.api_server import _api_request_profile, _error_response, _invalid_request
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None
        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping.")
            return None, _error_response(
                "X-Hermes-Session-Key requires API key authentication. "
                "Configure API_SERVER_KEY to enable this feature.", 403)
        # Control characters could enable header injection on the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, _invalid_request("Invalid session key")
        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, _invalid_request("Session key too long")
        return raw, None


    async def _verify_gateway_proxy_session_context(
        self, *, session_id: str, origin: Mapping[str, str], session_key: str,
    ) -> web.Response | None:
        """Create or verify durable origin + exact session key atomically."""
        from gateway.platforms.api_server import _api_request_profile, _error_response, _invalid_request
        from gateway.recall_scope import canonical_recall_identity

        db = await self._ensure_session_db_async()
        if db is None:
            return _error_response(
                "Gateway proxy scope could not open state.db", 503,
                err_type="server_error", code="gateway_proxy_state_unavailable")
        try:
            existing = await asyncio.to_thread(db.get_session, session_id)
            if not existing:
                encoded = json.dumps(
                    dict(origin), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

                def _insert_if_absent(conn):
                    return conn.execute(
                        """INSERT INTO sessions (
                               id, source, user_id, session_key, chat_id,
                               chat_type, thread_id, profile_name, origin_json,
                               started_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO NOTHING""",
                        (
                            session_id, str(origin.get("platform") or ""),
                            origin.get("user_id") or None, session_key,
                            origin.get("chat_id") or None, origin.get("chat_type") or None,
                            origin.get("thread_id") or None, origin.get("profile") or None,
                            encoded, time.time(),
                        ),
                    ).rowcount == 1

                if not hasattr(db, "_execute_write"):
                    raise RuntimeError("SessionDB lacks atomic insert support")
                await asyncio.to_thread(db._execute_write, _insert_if_absent)
                existing = await asyncio.to_thread(db.get_session, session_id)
        except Exception:
            logger.exception("Gateway proxy could not establish durable origin for %s", session_id)
            return _error_response(
                "Gateway proxy scope could not be persisted", 503,
                err_type="server_error", code="gateway_proxy_scope_persist_failed")

        raw_durable = (existing or {}).get("origin_json")
        try:
            durable_origin = json.loads(raw_durable) if raw_durable else None
        except (TypeError, ValueError):
            durable_origin = None
        live_identity = canonical_recall_identity(origin)
        durable_identity = canonical_recall_identity(durable_origin)
        if live_identity is None or durable_identity is None:
            return _error_response(
                "Gateway proxy session has no unambiguous durable origin", 409,
                code="gateway_proxy_scope_unavailable")
        durable_session_key = (existing or {}).get("session_key")
        if not isinstance(durable_session_key, str) or not durable_session_key:
            return _error_response(
                "Gateway proxy session has no durable session key", 409,
                code="gateway_proxy_session_key_unavailable")
        if durable_session_key != session_key:
            return _error_response(
                "Gateway proxy session key does not match the durable session", 409,
                code="gateway_proxy_session_key_mismatch")
        if live_identity != durable_identity:
            return _error_response(
                "Gateway proxy context does not match the durable session origin", 409,
                code="gateway_proxy_scope_mismatch")
        return None


    def _parse_gateway_proxy_context(
        self, request: web.Request,
    ) -> tuple[dict[str, str] | None, str | None, web.Response | None]:
        """Mandatory, non-downgradeable identity for the dedicated proxy route."""
        from gateway.platforms.api_server import _api_request_profile, _error_response, _invalid_request
        from gateway.recall_scope import (
            GATEWAY_PROXY_MARKER_HEADER,
            GATEWAY_PROXY_ORIGIN_HEADER,
            GATEWAY_PROXY_SESSION_KEY_HEADER,
            canonical_recall_identity,
            decode_gateway_proxy_origin,
            decode_gateway_proxy_session_key,
        )

        marker = self._single_gateway_proxy_header(request, GATEWAY_PROXY_MARKER_HEADER)
        encoded_origin = self._single_gateway_proxy_header(request, GATEWAY_PROXY_ORIGIN_HEADER)
        encoded_session_key = self._single_gateway_proxy_header(
            request, GATEWAY_PROXY_SESSION_KEY_HEADER)
        if marker != "1" or not encoded_origin or not encoded_session_key:
            return None, None, _error_response(
                "Gateway proxy context is missing or malformed", 400,
                code="gateway_proxy_context_invalid")
        # Normal route admission authenticated this request. Keep direct/manual
        # handler wiring from treating the historical no-key test mode as trusted.
        if not self._expected_api_key():
            return None, None, _error_response(
                "Gateway proxy context requires API key authentication", 403,
                err_type="gateway_auth_error", code="gateway_proxy_auth_required")
        try:
            origin = decode_gateway_proxy_origin(encoded_origin)
            session_key = decode_gateway_proxy_session_key(encoded_session_key)
        except ValueError as exc:
            return None, None, _error_response(
                str(exc), 400, code="gateway_proxy_context_invalid")
        identity = canonical_recall_identity(origin)
        request_profile = (_api_request_profile.get() or "default").strip().lower()
        if identity is None or identity.profile != request_profile:
            return None, None, _error_response(
                "Gateway proxy context does not match the requested profile", 409,
                code="gateway_proxy_profile_mismatch")
        return origin, session_key, None


    @staticmethod
    def _single_gateway_proxy_header(request: web.Request, name: str) -> str | None:
        """One exact header value; missing and duplicate fields are invalid."""
        headers = request.headers
        getall = getattr(headers, "getall", None)
        if callable(getall):
            values = list(getall(name, []))
        elif name in headers:
            values = [headers.get(name)]
        else:
            values = []
        return values[0] if len(values) == 1 and isinstance(values[0], str) else None



    def _cron_origin_from_request(self, request: web.Request) -> dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {"platform": "api_server", "chat_id": "api"}
        for ctx_key, origin_key in (("remote", "source_ip"), ("peer_ip", "peer_ip"),
                                    ("forwarded_for", "forwarded_for"), ("real_ip", "real_ip"),
                                    ("user_agent", "user_agent")):
            if ctx.get(ctx_key):
                origin[origin_key] = ctx[ctx_key]
        return origin



    async def _resolve_gateway_chat_request(self, request, *, gateway_proxy_route):
        """Resolve payload and durable session identity before model execution."""
        from gateway.platforms.api_server import (
            _api_request_profile, _chat_usage_payload, _coerce_request_bool,
            _content_has_visible_payload, _derive_chat_session_id, _error_response, _invalid_request,
        )
        from gateway.platforms.api_server_openai_routes import _parse_chat_messages
        gateway_proxy_origin: dict[str, str] | None = None
        gateway_proxy_session_key: str | None = None
        if gateway_proxy_route:
            (gateway_proxy_origin, gateway_proxy_session_key,
             proxy_context_err) = self._parse_gateway_proxy_context(request)
            if proxy_context_err is not None:
                return proxy_context_err
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            return _error_response("Invalid JSON in request body", 400)
        from gateway.platforms.api_server import _request_relay_metadata
        relay_metadata = _request_relay_metadata(body)
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return _invalid_request("Missing or invalid 'messages' field")
        stream = _coerce_request_bool(body.get("stream"), default=False)

        system_prompt, conversation_messages, messages_err = _parse_chat_messages(messages)
        if messages_err is not None:
            return messages_err
        user_message: Any = (conversation_messages[-1].get("content", "") if conversation_messages else "")
        history = conversation_messages[:-1]
        if not _content_has_visible_payload(user_message):
            return _invalid_request("No user message found in messages")

        # X-Hermes-Session-Key scopes long-term memory per channel; independent of
        # X-Hermes-Session-Id (the key persists across transcripts, the id rotates on /new).
        if gateway_proxy_route:
            gateway_session_key = gateway_proxy_session_key
        else:
            gateway_session_key, key_err = self._parse_session_key_header(request)
            if key_err is not None:
                return key_err
        # X-Hermes-Session-Id continues an existing session (history from state.db, not the body);
        # requires a configured API key or any client could read history by guessing ids.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if gateway_proxy_route and not provided_session_id:
            return _error_response(
                "Gateway proxy context requires X-Hermes-Session-Id", 400,
                code="gateway_proxy_session_required")
        if provided_session_id:
            if not self._expected_api_key():
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity.")
                return _error_response("Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature.", 403)
            # Same guard as the native gateway: ids are interpolated into on-disk filenames.
            from gateway.session import _is_path_unsafe
            if re.search(r'[\r\n\x00]', provided_session_id) or _is_path_unsafe(provided_session_id):
                return _invalid_request("Invalid session ID")
            if len(provided_session_id) > self._MAX_SESSION_HEADER_LEN:
                return _invalid_request("Session ID too long")
            session_id = provided_session_id
            if gateway_proxy_route:
                assert gateway_proxy_origin is not None
                assert gateway_session_key is not None
                proxy_scope_err = await self._verify_gateway_proxy_session_context(
                    session_id=session_id, origin=gateway_proxy_origin,
                    session_key=gateway_session_key)
                if proxy_scope_err is not None:
                    return proxy_scope_err
            db = await self._ensure_session_db_async()
            if db is None:
                return _error_response("Session continuation state is unavailable", 503)
            from gateway.platforms.api_server_runs import _resolve_live_session_id
            session_id = await _resolve_live_session_id(self, provided_session_id)
            history = await asyncio.to_thread(db.get_messages_as_conversation, session_id)
        else:
            # Stable id from the conversation fingerprint so Open WebUI-style clients map onto
            # one Hermes session; namespaced by the routed profile (#123989).
            first_user = next(
                (cm.get("content", "") for cm in conversation_messages if cm.get("role") == "user"), "")
            session_id = _derive_chat_session_id(system_prompt, first_user, _api_request_profile.get())
            session_id, lookup_error = await self._implicit_chat_session(request, session_id)
            if lookup_error is not None:
                return lookup_error
        return ChatRequestContext(
            body, stream, system_prompt, user_message, history, session_id, provided_session_id,
            gateway_session_key, gateway_proxy_origin, relay_metadata,
        )
