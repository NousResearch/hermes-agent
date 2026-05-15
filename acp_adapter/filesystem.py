"""ACP editor filesystem bridge.

This module lets synchronous Hermes file tools call ACP client filesystem
requests while an ACP session is running.  The ACP SDK methods are async, while
Hermes tools run synchronously inside the ACP executor thread, so the active
client/loop/session are bound via a ContextVar and awaited with
``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from tools.file_operations import ReadResult, WriteResult, normalize_read_pagination


@dataclass(frozen=True)
class ACPFilesystemContext:
    """Active ACP filesystem context for synchronous file tools."""

    client: Any
    session_id: str
    loop: asyncio.AbstractEventLoop
    cwd: str | None = None
    can_read: bool = False
    can_write: bool = False
    timeout: float = 30.0


_context: contextvars.ContextVar[ACPFilesystemContext | None] = contextvars.ContextVar(
    "acp_filesystem_context",
    default=None,
)


def _cap_bool(capabilities: Any, name: str) -> bool:
    fs = getattr(capabilities, "fs", None)
    return bool(getattr(fs, name, False))


def supports_read(capabilities: Any) -> bool:
    """Return whether ACP client capabilities include fs/read_text_file."""

    return _cap_bool(capabilities, "read_text_file")


def supports_write(capabilities: Any) -> bool:
    """Return whether ACP client capabilities include fs/write_text_file."""

    return _cap_bool(capabilities, "write_text_file")


@contextlib.contextmanager
def use_acp_filesystem(
    *,
    client: Any,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    cwd: str | None,
    capabilities: Any,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Bind ACP editor filesystem access for file tools in this context."""

    ctx = ACPFilesystemContext(
        client=client,
        session_id=session_id,
        loop=loop,
        cwd=cwd,
        can_read=supports_read(capabilities),
        can_write=supports_write(capabilities),
        timeout=timeout,
    )
    token = _context.set(ctx)
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> ACPFilesystemContext | None:
    """Return the currently bound ACP filesystem context, if any."""

    return _context.get()


def _absolute_path(path: str, cwd: str | None) -> str:
    raw = os.path.expanduser(str(path or ""))
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(cwd or os.getcwd()) / candidate
    return str(candidate.resolve(strict=False))


def _run_client_coro(ctx: ACPFilesystemContext, coro: Any) -> Any:
    """Run an ACP client coroutine from the synchronous tool thread."""

    future = asyncio.run_coroutine_threadsafe(coro, ctx.loop)
    return future.result(timeout=ctx.timeout)


def _add_line_numbers(content: str, offset: int) -> str:
    lines = content.splitlines()
    return "\n".join(f"{offset + idx:6d}|{line}" for idx, line in enumerate(lines))


def read_text_file(path: str, offset: int = 1, limit: int = 500) -> ReadResult | None:
    """Read through ACP fs/read_text_file if active and supported.

    Returns ``None`` when there is no active ACP filesystem or the client did
    not advertise read support, allowing callers to fall back unchanged.
    """

    ctx = current_context()
    if ctx is None or not ctx.can_read:
        return None
    offset, limit = normalize_read_pagination(offset, limit)
    abs_path = _absolute_path(path, ctx.cwd)
    try:
        response = _run_client_coro(
            ctx,
            ctx.client.read_text_file(
                path=abs_path,
                session_id=ctx.session_id,
                limit=limit,
                line=offset,
            ),
        )
        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content or "")
        line_count = len(content.splitlines())
        return ReadResult(
            content=_add_line_numbers(content, offset),
            total_lines=(offset + line_count - 1) if line_count else 0,
            file_size=len(content.encode("utf-8")),
            truncated=False,
        )
    except Exception as exc:
        return ReadResult(error=f"ACP editor filesystem read failed for '{abs_path}': {exc}")


def write_text_file(path: str, content: str) -> WriteResult | None:
    """Write through ACP fs/write_text_file if active and supported.

    Returns ``None`` when there is no active ACP filesystem or the client did
    not advertise write support, allowing callers to fall back unchanged.
    """

    ctx = current_context()
    if ctx is None or not ctx.can_write:
        return None
    abs_path = _absolute_path(path, ctx.cwd)
    try:
        _run_client_coro(
            ctx,
            ctx.client.write_text_file(
                content=content,
                path=abs_path,
                session_id=ctx.session_id,
            ),
        )
        return WriteResult(
            bytes_written=len(content.encode("utf-8")),
            dirs_created=False,
            warning="Wrote via ACP editor filesystem; local disk fallback was not used.",
        )
    except Exception as exc:
        return WriteResult(error=f"ACP editor filesystem write failed for '{abs_path}': {exc}")
