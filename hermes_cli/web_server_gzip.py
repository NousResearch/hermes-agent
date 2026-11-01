"""Compress finite dashboard reads without changing streams or downloads."""

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import Message, Receive, Scope, Send


class _DashboardGZipResponder(GZipResponder):
    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] == "http.response.start":
            headers = Headers(raw=message["headers"])
            media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            # BaseHTTPMiddleware re-chunks even finite responses. Content-Length
            # preserves their size; real streams omit it. FileResponse advertises
            # ranges, and downloads may also declare Content-Disposition.
            self.content_type_is_excluded = (
                self.content_type_is_excluded
                or not (media_type == "application/json" or media_type.startswith("text/"))
                or int(headers.get("content-length", "0")) < self.minimum_size
                or message["status"] == 206
                or any(name in headers for name in (
                    "content-range", "content-disposition", "accept-ranges",
                ))
            )


class DashboardGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or "gzip" not in Headers(scope=scope).get("accept-encoding", "")
            or "range" in Headers(scope=scope)
        ):
            await self.app(scope, receive, send)
            return
        responder = _DashboardGZipResponder(self.app, self.minimum_size, self.compresslevel)
        await responder(scope, receive, send)
