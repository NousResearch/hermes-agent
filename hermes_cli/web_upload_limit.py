"""Bound Webapp attachment ingestion before FastAPI parses/spools multipart."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES

# The file cap remains exact in the handler; allow ordinary multipart headers
# and boundaries without letting arbitrary extra parts consume unbounded disk.
CHAT_FILE_BODY_MAX_BYTES = WEBAPP_ATTACHMENT_MAX_BYTES + 64 * 1024


class ChatFileUploadLimitMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        if scope["type"] != "http" or scope["method"] != "POST" or path != "/api/chat/file-upload":
            await self.app(scope, receive, send)
            return

        rejection = JSONResponse(status_code=413, content={"detail": "Upload request is too large"})
        length = Headers(scope=scope).get("content-length", "").lstrip("0") or "0"
        if (
            length.isascii()
            and length.isdecimal()
            and (len(length) > len(str(CHAT_FILE_BODY_MAX_BYTES)) or int(length) > CHAT_FILE_BODY_MAX_BYTES)
        ):
            await rejection(scope, receive, send)
            return

        received = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > CHAT_FILE_BODY_MAX_BYTES:
                    exceeded = True
                    # This exception is deliberate: Starlette's multipart parser
                    # closes *all* open spools on MultiPartException, unlike an
                    # HTTPException or cancellation. The rejected chunk never
                    # reaches the parser. Request._get_form converts it to 400.
                    raise MultiPartException("Upload request is too large")
            return message

        async def limited_send(message: Message) -> None:
            if not exceeded:
                await send(message)
            elif message["type"] == "http.response.start":
                # Replace the parser's 400 (including its Content-Length) with
                # 413, after its cleanup, and suppress the original body.
                await rejection(scope, receive, send)

        await self.app(scope, limited_receive, limited_send)
