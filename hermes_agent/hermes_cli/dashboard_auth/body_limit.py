"""Bound public auth bodies before FastAPI assembles JSON for validation."""
from __future__ import annotations

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hermes_cli.dashboard_auth.middleware import _path_is_public

# Auth exchanges contain credentials and tokens, not files. Leave ample room for
# those small JSON payloads without allowing anonymous, unbounded buffering.
AUTH_BODY_LIMIT: int = 64 * 1024


class _AuthBodyTooLarge(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="Request body too large")


class AuthBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] != "http" or scope["method"] != "POST"
                or not scope["path"].startswith("/auth/")
                or not _path_is_public(scope["path"])):
            await self.app(scope, receive, send)
            return

        response = JSONResponse({"detail": "Request body too large"}, status_code=413)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    length = int(value)
                except ValueError:
                    continue
                if length > AUTH_BODY_LIMIT:
                    await response(scope, receive, send)
                    return

        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > AUTH_BODY_LIMIT:
                    # HTTPException survives FastAPI's JSON parsing error handler.
                    raise _AuthBodyTooLarge()
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _AuthBodyTooLarge:
            if started:
                raise
            await response(scope, receive, send)
