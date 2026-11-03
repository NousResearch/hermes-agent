"""Bound chat upload request bodies before FastAPI parses, spools or buffers them."""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hermes_cli.web_server_surface import policy
from hermes_constants import WEBAPP_ATTACHMENT_MAX_BYTES

# The file cap remains exact in the handler; allow ordinary multipart headers
# and boundaries without letting arbitrary extra parts consume unbounded disk.
CHAT_FILE_BODY_MAX_BYTES = WEBAPP_ATTACHMENT_MAX_BYTES + 64 * 1024

# Largest decoded image /api/chat/image-upload stores. The image arrives base64-encoded in a JSON
# body, which FastAPI reads whole before the handler checks the image: bound the body to that
# image's base64 length plus room for the JSON around it, so an oversized paste is never buffered.
CHAT_IMAGE_MAX_BYTES = 25 * 1024 * 1024
CHAT_IMAGE_BODY_MAX_BYTES = 4 * -(-CHAT_IMAGE_MAX_BYTES // 3) + 64 * 1024

_BODY_MAX_BYTES = {
    "/api/chat/file-upload": CHAT_FILE_BODY_MAX_BYTES,
    "/api/chat/image-upload": CHAT_IMAGE_BODY_MAX_BYTES,
}


class ChatFileUploadLimitMiddleware:
    """Refuse a chat file or image upload with 413 once its body passes the route's cap: at once
    when Content-Length declares more, else at the first chunk past it. Only on the surfaces whose
    policy serves the browser's file routes (the Webapp); elsewhere uploads reach their handlers
    as on main."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        root = scope.get("root_path", "")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
        body_max = _BODY_MAX_BYTES.get(path)
        if (scope["type"] != "http" or scope["method"] != "POST" or body_max is None
                or not policy(getattr(scope.get("app"), "state", None)).browser_files):
            await self.app(scope, receive, send)
            return

        rejection = JSONResponse(status_code=413, content={"detail": "Upload request is too large"})
        length = Headers(scope=scope).get("content-length", "").lstrip("0") or "0"
        if (
            length.isascii()
            and length.isdecimal()
            and (len(length) > len(str(body_max)) or int(length) > body_max)
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
                if received > body_max:
                    exceeded = True
                    # This exception is deliberate: Starlette's multipart parser
                    # closes *all* open spools on MultiPartException, unlike an
                    # HTTPException or cancellation. The rejected chunk never
                    # reaches the parser. Request._get_form converts it to 400,
                    # as FastAPI does for a JSON body whose read fails.
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
