"""Inline audio/video from a profile's SSH workspace for ``/api/files/stream``.

A host file streams through Starlette's ``FileResponse``, which answers ``Range`` itself. A file on
the SSH machine has no local handle, so this answers the one byte range a media element asks for
by reading exactly that slice through the workspace adapter (``SshWorkspaceFs.stream_file``).
The caller has already chosen the backend and checked the file (sensitive path, media type, size).
"""

from __future__ import annotations

import re
import urllib.parse
from pathlib import PurePosixPath
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse

_SINGLE_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.IGNORECASE)
# More digits than any file this route serves can need. A longer number is read as one value past
# every file instead of being converted: int() refuses a few thousand digits (a 500).
_MAX_POSITION_DIGITS = 19
_PAST_ANY_FILE = 10**_MAX_POSITION_DIGITS


def _position(digits: str) -> int:
    """A ``Range`` byte position, or ``_PAST_ANY_FILE`` for one too long to be in any file."""
    digits = digits.lstrip("0") or "0"
    return int(digits) if len(digits) <= _MAX_POSITION_DIGITS else _PAST_ANY_FILE


def _unsatisfiable(size: int) -> HTTPException:
    return HTTPException(
        status_code=416, detail="Requested range not satisfiable", headers={"Content-Range": f"bytes */{size}"},
    )


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The inclusive ``(first, last)`` byte span one ``bytes=`` range asks of a *size*-byte file,
    or None to serve the whole file. A header this does not read (several ranges, another unit, a
    last byte before the first) is ignored, as RFC 9110 allows; a range the file cannot satisfy
    raises 416. A last byte past the end is the end, and a suffix longer than the file is all of it,
    however many digits either has."""
    match = _SINGLE_RANGE.match(header or "")
    if match is None or match.groups() == ("", ""):
        return None
    first_digits, last_digits = match.groups()
    if not first_digits:  # a suffix: the last N bytes
        suffix = _position(last_digits)
        if suffix == 0 or size == 0:
            raise _unsatisfiable(size)
        return max(size - suffix, 0), size - 1
    first = _position(first_digits)
    last = _position(last_digits) if last_digits else None
    if last is not None and last < first:
        return None
    if first >= size:
        raise _unsatisfiable(size)
    return first, min(last, size - 1) if last is not None else size - 1


def remote_media_response(request: Request, workspace: Any, target: str, size: int, media_type: str) -> Response:
    """*target* on the SSH *workspace* as inline media: 200 for the whole file, 206 for the range
    the request asks, 416 for one it cannot have. A HEAD reads none of it."""
    span = byte_range(request.headers.get("range"), size)
    first, last = span if span is not None else (0, size - 1)
    length = max(last - first + 1, 0)
    name = urllib.parse.quote(PurePosixPath(target).name)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"inline; filename*=UTF-8''{name}",
        "Content-Length": str(length),
        "X-Content-Type-Options": "nosniff",
    }
    status = 200
    if span is not None:
        status = 206
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    if request.method == "HEAD" or length == 0:
        return Response(status_code=status, media_type=media_type, headers=headers)
    body = workspace.stream_file(target, offset=first, length=length) if span else workspace.stream_file(target)
    return StreamingResponse(body, status_code=status, media_type=media_type, headers=headers)
