"""Range responses over a descriptor opened under the file ticket's profile lease."""
from __future__ import annotations

import os
from secrets import token_hex

import anyio
from anyio.to_thread import run_sync
from starlette.datastructures import MutableHeaders
from starlette.responses import FileResponse


class OpenedFileResponse(FileResponse):
    """Keep Starlette's Range parsing/headers, but never reopen a reusable path.

    The opener runs immediately before response dispatch and returns a checked
    file object. Its lifecycle lease ends before any network await. Every exit,
    including HEAD, invalid Range and disconnect, closes the pinned descriptor.
    """

    def __init__(self, *args, opener, **kwargs):
        super().__init__(*args, **kwargs)
        self._opener = opener

    async def __call__(self, scope, receive, send):
        handle = await run_sync(self._opener)
        self._file = anyio.wrap_file(handle)
        try:
            self.stat_result = os.fstat(handle.fileno())
            self._file_size = self.stat_result.st_size
            self.set_stat_headers(self.stat_result)
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._file.aclose()

    async def _send_range(self, send, start, end):
        await self._file.seek(start)
        while start < end:
            chunk = await self._file.read(min(self.chunk_size, end - start))
            if not chunk:
                break
            start += len(chunk)
            await send({"type": "http.response.body", "body": chunk, "more_body": True})

    async def _handle_simple(self, send, send_header_only, send_pathsend):
        # ASGI pathsend would reopen the pathname and lose the pinned identity.
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        if not send_header_only:
            await self._send_range(send, 0, self._file_size)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_single_range(self, send, start, end, file_size, send_header_only):
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if not send_header_only:
            await self._send_range(send, start, end)
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _handle_multiple_ranges(self, send, ranges, file_size, send_header_only):
        boundary = token_hex(13)
        length, header = self.generate_multipart(ranges, boundary, file_size, self.headers["content-type"])
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(length)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if not send_header_only:
            for start, end in ranges:
                await send({"type": "http.response.body", "body": header(start, end), "more_body": True})
                await self._send_range(send, start, end)
                await send({"type": "http.response.body", "body": b"\r\n", "more_body": True})
            await send({"type": "http.response.body", "body": f"--{boundary}--".encode("ascii"), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})
