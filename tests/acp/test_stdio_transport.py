"""Tests for robust ACP stdio transport selection."""

from __future__ import annotations

import asyncio
import io
import sys
from unittest.mock import AsyncMock

import pytest

from acp_adapter import stdio_transport as st


@pytest.mark.asyncio
async def test_posix_uses_pipe_transport_when_available(monkeypatch):
    """Happy path: connect_*_pipe succeeds → no fallback."""
    calls = {"posix": 0, "fallback": 0}

    async def fake_posix(loop, limit=None):
        calls["posix"] += 1
        reader = asyncio.StreamReader()
        # minimal fake writer via fallback machinery without real fds
        return reader, object()

    async def fake_fallback(loop, limit=None):
        calls["fallback"] += 1
        raise AssertionError("fallback must not run when posix works")

    monkeypatch.setattr(st, "_posix_pipe_stdio_streams", fake_posix)
    monkeypatch.setattr(st, "_fallback_stdio_streams", fake_fallback)
    monkeypatch.setattr(st, "_stdio_fds_support_pipe_transport", lambda: (True, "ok"))
    monkeypatch.setattr(st.platform, "system", lambda: "Darwin")

    reader, writer = await st.open_acp_stdio_streams(limit=1024)
    assert calls == {"posix": 1, "fallback": 0}
    assert reader is not None
    assert writer is not None


@pytest.mark.asyncio
async def test_posix_falls_back_on_pipe_transport_error(monkeypatch):
    """When asyncio rejects the fd, use thread/buffer transport."""
    calls = {"posix": 0, "fallback": 0}

    async def fake_posix(loop, limit=None):
        calls["posix"] += 1
        raise ValueError(
            "Pipe transport is only for pipes, sockets and character devices"
        )

    async def fake_fallback(loop, limit=None):
        calls["fallback"] += 1
        reader = asyncio.StreamReader()
        return reader, object()

    monkeypatch.setattr(st, "_posix_pipe_stdio_streams", fake_posix)
    monkeypatch.setattr(st, "_fallback_stdio_streams", fake_fallback)
    monkeypatch.setattr(st, "_stdio_fds_support_pipe_transport", lambda: (True, "ok"))
    monkeypatch.setattr(st.platform, "system", lambda: "Darwin")

    reader, writer = await st.open_acp_stdio_streams()
    assert calls == {"posix": 1, "fallback": 1}
    assert reader is not None
    assert writer is not None


@pytest.mark.asyncio
async def test_posix_reraises_unrelated_valueerror(monkeypatch):
    async def fake_posix(loop, limit=None):
        raise ValueError("something else entirely")

    monkeypatch.setattr(st, "_posix_pipe_stdio_streams", fake_posix)
    monkeypatch.setattr(st, "_stdio_fds_support_pipe_transport", lambda: (True, "ok"))
    monkeypatch.setattr(st.platform, "system", lambda: "Linux")

    with pytest.raises(ValueError, match="something else"):
        await st.open_acp_stdio_streams()


@pytest.mark.asyncio
async def test_windows_always_uses_fallback(monkeypatch):
    calls = {"posix": 0, "fallback": 0}

    async def fake_posix(loop, limit=None):
        calls["posix"] += 1
        raise AssertionError("posix path must not run on Windows")

    async def fake_fallback(loop, limit=None):
        calls["fallback"] += 1
        return asyncio.StreamReader(), object()

    monkeypatch.setattr(st, "_posix_pipe_stdio_streams", fake_posix)
    monkeypatch.setattr(st, "_fallback_stdio_streams", fake_fallback)
    monkeypatch.setattr(st.platform, "system", lambda: "Windows")

    await st.open_acp_stdio_streams()
    assert calls == {"posix": 0, "fallback": 1}


def test_is_pipe_transport_error_matches_cpython_messages():
    assert st._is_pipe_transport_error(
        ValueError("Pipe transport is only for pipes, sockets and character devices")
    )
    assert st._is_pipe_transport_error(
        ValueError("Pipe transport is for pipes/sockets only.")
    )
    assert not st._is_pipe_transport_error(ValueError("nope"))
    assert not st._is_pipe_transport_error(OSError("Pipe transport is only for pipes"))


def test_preflight_skips_posix_when_fds_incompatible(monkeypatch):
    calls = {"posix": 0, "fallback": 0}

    async def fake_posix(loop, limit=None):
        calls["posix"] += 1
        raise AssertionError("posix must not run when preflight fails")

    async def fake_fallback(loop, limit=None):
        calls["fallback"] += 1
        return asyncio.StreamReader(), object()

    monkeypatch.setattr(st, "_posix_pipe_stdio_streams", fake_posix)
    monkeypatch.setattr(st, "_fallback_stdio_streams", fake_fallback)
    monkeypatch.setattr(
        st, "_stdio_fds_support_pipe_transport", lambda: (False, "stdin=reg stdout=reg")
    )
    monkeypatch.setattr(st.platform, "system", lambda: "Darwin")

    import asyncio as aio

    async def run():
        await st.open_acp_stdio_streams()

    aio.run(run())
    assert calls == {"posix": 0, "fallback": 1}


@pytest.mark.asyncio
async def test_fallback_transport_writes_jsonrpc_bytes(monkeypatch):
    """End-ish unit: buffer transport can emit a JSON-RPC line to stdout.buffer."""
    buf = io.BytesIO()

    class _Stdout:
        buffer = buf

        def flush(self):
            pass

    monkeypatch.setattr(st.sys, "stdout", _Stdout())

    # Don't start a real stdin feeder that blocks forever on this process's stdin.
    monkeypatch.setattr(st, "_start_stdin_feeder", lambda loop, reader: None)

    loop = asyncio.get_running_loop()
    reader, writer = await st._fallback_stdio_streams(loop, limit=64 * 1024)
    assert isinstance(reader, asyncio.StreamReader)

    frame = b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
    writer.write(frame)
    await writer.drain()
    assert buf.getvalue() == frame


@pytest.mark.asyncio
async def test_fallback_works_when_stdout_is_regular_file(tmp_path, monkeypatch):
    """Sandbox: simulate the failing host shape (stdout = regular file).

    Native connect_write_pipe rejects REG files; fallback must still write.
    """
    out_file = tmp_path / "stdout.bin"
    # Open binary-capable text wrapper like a redirected process stdout.
    raw = open(out_file, "wb", buffering=0)

    class _TextOut:
        def __init__(self, buffer):
            self.buffer = buffer

        def flush(self):
            self.buffer.flush()

        def fileno(self):
            return raw.fileno()

    monkeypatch.setattr(sys, "stdout", _TextOut(raw))
    monkeypatch.setattr(st.sys, "stdout", sys.stdout)
    # Avoid blocking on real stdin feeder in unit test.
    monkeypatch.setattr(st, "_start_stdin_feeder", lambda loop, reader: None)
    # Preflight says "not ok" so we never enter posix (avoids partial connect).
    monkeypatch.setattr(
        st,
        "_stdio_fds_support_pipe_transport",
        lambda: (False, "stdin=fifo stdout=reg"),
    )
    monkeypatch.setattr(st.platform, "system", lambda: "Darwin")

    reader, writer = await st.open_acp_stdio_streams()
    frame = b'{"jsonrpc":"2.0","id":7,"result":{}}\n'
    writer.write(frame)
    await writer.drain()
    raw.flush()
    raw.close()
    assert out_file.read_bytes() == frame
