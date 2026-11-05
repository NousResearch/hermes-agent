"""Compress finite dashboard reads without changing streams or downloads."""

from starlette.datastructures import Headers, MutableHeaders
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import Message, Receive, Scope, Send


def _compressible(message: Message, minimum_size: int) -> bool:
    headers = Headers(raw=message["headers"])
    media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
    # BaseHTTPMiddleware re-chunks even finite responses. Content-Length
    # preserves their size; real streams omit it. FileResponse advertises
    # ranges, and downloads may also declare Content-Disposition.
    return (
        (media_type == "application/json" or media_type.startswith("text/"))
        and int(headers.get("content-length", "0")) >= minimum_size
        and message["status"] != 206
        and not any(name in headers for name in (
            "content-range", "content-disposition", "accept-ranges", "content-encoding",
        ))
    )


class _DashboardGZipResponder(GZipResponder):
    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] == "http.response.start":
            self.content_type_is_excluded = (
                self.content_type_is_excluded or not _compressible(message, self.minimum_size)
            )


def _accepts_gzip(accept_encoding: str) -> bool:
    """True when the request lists gzip with a non-zero q-value (RFC 9110 12.5.3)."""
    for item in accept_encoding.lower().split(","):
        coding, *params = (part.strip() for part in item.split(";"))
        if coding != "gzip":
            continue
        for param in params:
            name, _, value = param.partition("=")
            if name.strip() == "q":
                try:
                    return float(value) > 0
                except ValueError:
                    return False
        return True
    return False


class DashboardGZipMiddleware(GZipMiddleware):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "range" in (headers := Headers(scope=scope)):
            await self.app(scope, receive, send)
            return
        if _accepts_gzip(headers.get("accept-encoding", "")):
            responder = _DashboardGZipResponder(self.app, self.minimum_size, self.compresslevel)
            await responder(scope, receive, send)
            return

        # An identity response that could have been compressed still says it varies, so a
        # cache never hands it to a client that asks for gzip. Decided from the headers alone,
        # so nothing is buffered and streams keep their timing.
        async def send_identity(message: Message) -> None:
            if message["type"] == "http.response.start" and _compressible(message, self.minimum_size):
                MutableHeaders(scope=message).add_vary_header("Accept-Encoding")
            await send(message)

        await self.app(scope, receive, send_identity)
