"""W54-F037 (B15): the control-socket pointer-file reader must never observe a torn write.

Writer/reader race on ``gateway.sock.path``: production data is written with
``pointer_file.write_text(...)`` (truncate-then-write). A reader that lands inside the
truncate/write window can observe EMPTY or PARTIAL content; ``resolve_client_socket_path``
then either degrades to ``None`` or — worse — returns a WRONG path whose prefix happens to
exist (e.g. the socket's parent directory).

This test drives the real writer (``GatewayControlServer.start`` → ``_start_posix``) and the
real reader (``resolve_client_socket_path``) concurrently. The write seam below stretches the
``Path.write_text`` window deterministically (event-gated, zero sleeps) because APFS happens to
render tiny writes atomic in practice — behaviour the contract fix must NOT rely on: the
atomic-rename helper avoids the window entirely, so post-fix the seam simply never fires and
the pointer is always observed complete (or absent).
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest

from gateway.control_socket import (
    GatewayControlServer,
    resolve_client_socket_path,
    resolve_server_socket_path,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-socket transport; the named-pipe half is covered on the wine2e lane",
)

_WINDOW_TIMEOUT = 10.0
_real_write_text = Path.write_text


def test_reader_never_observes_torn_pointer_file(tmp_path: Path, monkeypatch):
    # A long HERMES_HOME defeats sun_path so the server uses the temp-dir socket + pointer file.
    home = tmp_path / ("h" * 120)
    home.mkdir()
    bind_path, pointer_file = resolve_server_socket_path(home)
    assert pointer_file is not None, "test requires the temp-dir pointer fallback"

    target = Path(pointer_file)
    # Chunk 1 is the socket's parent directory plus its trailing slash: mid-window content that
    # resolves to an EXISTING path — the wrong-socket failure mode of a torn read.
    split_at = len(str(Path(bind_path).parent)) + 1
    started = threading.Event()
    release = threading.Event()

    def seamed_write_text(path_self, data, encoding=None, errors=None, newline=None):
        if Path(path_self) == target:
            text = str(data)
            with open(path_self, "w", encoding=encoding or "utf-8") as handle:
                handle.write(text[:split_at])
                handle.flush()
                started.set()
                assert release.wait(_WINDOW_TIMEOUT), "reader never entered the write window"
                handle.write(text[split_at:])
            return len(text)
        return _real_write_text(path_self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", seamed_write_text)
    server = GatewayControlServer(home)

    observations: list[tuple[str, Path | None, str | None]] = []
    errors: list[str] = []

    def writer() -> None:
        if not asyncio.run(server.start()):
            errors.append("server failed to start")

    def reader() -> None:
        if started.wait(_WINDOW_TIMEOUT):
            try:
                raw = pointer_file.read_text(encoding="utf-8") if pointer_file.is_file() else None
                observations.append(("window", resolve_client_socket_path(home), raw))
            finally:
                release.set()
        else:
            # Atomic write path: no torn window can exist; verify the published content.
            raw = pointer_file.read_text(encoding="utf-8") if pointer_file.is_file() else None
            observations.append(("after", resolve_client_socket_path(home), raw))

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    reader_thread.join(_WINDOW_TIMEOUT + 5)
    writer_thread.join(_WINDOW_TIMEOUT + 5)
    assert not writer_thread.is_alive() and not reader_thread.is_alive(), "race threads hung"
    assert not errors, errors
    assert observations, "reader produced no observation"

    for label, resolved, raw in observations:
        if label == "window":
            assert resolved == bind_path, (
                f"torn window read: resolver returned {resolved!r}, expected {str(bind_path)!r} "
                f"(raw pointer content: {raw!r})"
            )
            assert raw == str(bind_path), f"torn pointer content in write window: {raw!r}"
        else:
            assert resolved == bind_path, (
                f"post-write read returned {resolved!r}, expected {str(bind_path)!r}"
            )
            assert raw == str(bind_path), f"incomplete final pointer content: {raw!r}"

    # Cleanup: remove socket + pointer files.
    asyncio.run(server.stop())