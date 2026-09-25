"""Authenticated API-server routes for Mattermost session bindings."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web

from .session_bindings import (
    BindingValidationError,
    MattermostSessionBindingStore,
    SessionBinding,
)

API_PREFIX = "/api/plugins/mattermost/v1"

TargetNormalizer = Callable[[str, str], Awaitable[tuple[str, str]]]
ConnectedProbe = Callable[[], bool]


def _error(message: str, status: int, code: str) -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": "mattermost_binding_error", "code": code}},
        status=status,
    )


def _binding_response(binding: SessionBinding) -> dict[str, Any]:
    return {"object": "mattermost.session_binding", "binding": binding.as_dict()}


class MattermostSessionBindingAPI:
    """Versioned binding resource mounted into the existing Hermes API server."""

    def __init__(
        self,
        api_adapter: Any,
        *,
        target_normalizer: TargetNormalizer | None = None,
        connected_probe: ConnectedProbe | None = None,
        store_factory: Callable[[], MattermostSessionBindingStore] = MattermostSessionBindingStore,
    ) -> None:
        self._api_adapter = api_adapter
        self._target_normalizer = target_normalizer
        self._connected_probe = connected_probe or (lambda: target_normalizer is not None)
        self._store_factory = store_factory

    def register_routes(self, app: web.Application) -> None:
        routes = (
            ("GET", f"{API_PREFIX}/capabilities", self.capabilities),
            ("GET", f"{API_PREFIX}/session-bindings", self.list_bindings),
            ("GET", f"{API_PREFIX}/session-bindings/resolve", self.resolve_binding),
            ("GET", f"{API_PREFIX}/session-bindings/{{session_id}}", self.get_binding),
            ("PUT", f"{API_PREFIX}/session-bindings/{{session_id}}", self.put_binding),
            ("DELETE", f"{API_PREFIX}/session-bindings/{{session_id}}", self.delete_binding),
        )
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)

    def _auth_error(self, request: web.Request) -> web.Response | None:
        return self._api_adapter._check_auth(request)

    @staticmethod
    def _pagination(request: web.Request) -> tuple[int, int] | web.Response:
        try:
            limit = int(request.query.get("limit", "100"))
            offset = int(request.query.get("offset", "0"))
        except ValueError:
            return _error("limit and offset must be integers", 400, "invalid_pagination")
        if limit < 0 or offset < 0:
            return _error("limit and offset must be non-negative", 400, "invalid_pagination")
        return min(limit, 200), offset

    async def capabilities(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        return web.json_response({
            "object": "mattermost.capabilities",
            "version": 1,
            "features": ["session_bindings", "bidirectional_sync"],
            "mattermost_connected": bool(self._connected_probe()),
        })

    async def list_bindings(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        pagination = self._pagination(request)
        if isinstance(pagination, web.Response):
            return pagination
        limit, offset = pagination
        bindings = await asyncio.to_thread(
            self._store_factory().list_bindings, limit=limit, offset=offset
        )
        return web.json_response({
            "object": "list",
            "data": [binding.as_dict() for binding in bindings],
            "limit": limit,
            "offset": offset,
            "has_more": len(bindings) >= limit if limit else False,
        })

    async def resolve_binding(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        try:
            binding = await asyncio.to_thread(
                self._store_factory().resolve,
                request.query.get("channel_id"),
                request.query.get("root_post_id"),
            )
        except BindingValidationError as exc:
            return _error(str(exc), 400, "invalid_binding")
        if binding is None:
            return _error("Mattermost thread is not bound", 404, "binding_not_found")
        return web.json_response(_binding_response(binding))

    async def get_binding(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        try:
            binding = await asyncio.to_thread(
                self._store_factory().get_by_session, request.match_info["session_id"]
            )
        except BindingValidationError as exc:
            return _error(str(exc), 400, "invalid_binding")
        if binding is None:
            return _error("Hermes session is not bound", 404, "binding_not_found")
        return web.json_response(_binding_response(binding))

    async def put_binding(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        session_id = request.match_info["session_id"]
        _, session_error = await self._api_adapter._get_existing_session_or_404(session_id)
        if session_error:
            return session_error
        body, body_error = await self._api_adapter._read_json_body(request)
        if body_error:
            return body_error
        unknown = sorted(set(body) - {"channel_id", "root_post_id"})
        if unknown:
            return _error(
                f"Unsupported binding fields: {', '.join(unknown)}",
                400,
                "unsupported_binding_field",
            )
        channel_id = body.get("channel_id")
        root_post_id = body.get("root_post_id")
        if self._target_normalizer is not None:
            try:
                channel_id, root_post_id = await self._target_normalizer(channel_id, root_post_id)
            except BindingValidationError as exc:
                return _error(str(exc), 400, "invalid_binding")
            except LookupError as exc:
                return _error(str(exc), 404, "mattermost_thread_not_found")
            except RuntimeError as exc:
                return _error(str(exc), 503, "mattermost_unavailable")
        try:
            binding = await asyncio.to_thread(
                self._store_factory().replace, session_id, channel_id, root_post_id
            )
        except BindingValidationError as exc:
            return _error(str(exc), 400, "invalid_binding")
        return web.json_response(_binding_response(binding), status=200)

    async def delete_binding(self, request: web.Request) -> web.Response:
        if auth_error := self._auth_error(request):
            return auth_error
        session_id = request.match_info["session_id"]
        try:
            deleted = await asyncio.to_thread(self._store_factory().delete, session_id)
        except BindingValidationError as exc:
            return _error(str(exc), 400, "invalid_binding")
        return web.json_response({
            "object": "mattermost.session_binding.deleted",
            "id": session_id,
            "deleted": deleted,
        })
