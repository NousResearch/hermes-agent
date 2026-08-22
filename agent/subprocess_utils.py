"""asyncio subprocess helper with a larger default StreamReader limit."""

from __future__ import annotations

import asyncio
from typing import Any


# asyncio's default is 64 KiB, tuned for network protocols.  Local subprocess
# output (LSP diagnostics, base64 blobs, quoted source files) can easily exceed
# that on a single line, causing readline() to raise LimitOverrunError and
# deadlocking the pipe.  16 MiB covers realistic agent workloads while bounding
# unbounded growth.
SUBPROCESS_STREAM_LIMIT = 16 * 1024 * 1024  # 16 MiB


async def create_subprocess(
    *args: str,
    stream_limit: int = SUBPROCESS_STREAM_LIMIT,
    **kwargs: Any,
) -> asyncio.subprocess.Process:
    """asyncio.create_subprocess_exec with a larger default StreamReader limit."""
    return await asyncio.create_subprocess_exec(*args, limit=stream_limit, **kwargs)
