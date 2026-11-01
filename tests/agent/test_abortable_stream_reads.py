"""A watchdog's socket abort must reach a reader parked in a socket read on every platform (#143704).

``_shutdown_socket`` wakes a blocked ``recv()`` on Linux and macOS but not on Windows. There
``build_keepalive_http_client`` installs a backend whose streams read in short slices and stop once
the socket is marked aborted, so only the owner thread ever touches its own descriptor.
"""
import socket
import threading
import time

import httpcore
import pytest

from agent import process_bootstrap
from agent.agent_runtime_helpers import _shutdown_socket
from agent.process_bootstrap import socket_aborted


@pytest.fixture
def stalled_peer():
    """A connected socket pair whose peer sends one chunk, then goes silent."""
    listener = socket.create_server(("127.0.0.1", 0))
    client = socket.create_connection(listener.getsockname())
    peer, _ = listener.accept()
    listener.close()
    peer.sendall(b"first chunk\n")
    yield client, peer
    for sock in (client, peer):
        sock.close()


def _abortable(sock):
    return process_bootstrap._interruptible_classes()[0](sock)


def test_shutdown_unblocks_a_parked_read(stalled_peer):
    client, _peer = stalled_peer
    stream = _abortable(client)
    assert stream.read(4096, timeout=20) == b"first chunk\n"
    outcome = {}

    def reader():
        started = time.monotonic()
        try:
            outcome["data"] = stream.read(4096, timeout=20)
        except httpcore.ReadError:
            outcome["data"] = "aborted"
        outcome["elapsed"] = time.monotonic() - started

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    time.sleep(0.3)  # health: allow HX013 -- the read must already be parked when the abort arrives
    _shutdown_socket(client)  # what the stale-stream / no-progress watchdog thread does
    thread.join(5)
    # EOF where the OS wakes the read itself (Linux, macOS); ReadError where only the mark does (Windows).
    assert outcome.get("data") in (b"", "aborted"), outcome
    assert outcome["elapsed"] < 3.0
    assert socket_aborted(client)


def test_slow_live_stream_is_not_cut_between_slices(stalled_peer):
    client, peer = stalled_peer
    stream = _abortable(client)
    assert stream.read(4096, timeout=5) == b"first chunk\n"

    def trickle():
        for i in range(3):
            # health: allow HX013 -- gaps longer than one poll slice are the behaviour under test
            time.sleep(process_bootstrap._ABORT_POLL_SECONDS * 2)
            peer.sendall(f"chunk-{i}\n".encode())

    threading.Thread(target=trickle, daemon=True).start()
    received = b""
    while received.count(b"\n") < 3:
        received += stream.read(4096, timeout=5)
    assert received == b"chunk-0\nchunk-1\nchunk-2\n"


def test_read_timeout_still_fires(stalled_peer):
    client, _peer = stalled_peer
    stream = _abortable(client)
    stream.read(4096, timeout=5)
    started = time.monotonic()
    with pytest.raises(httpcore.ReadTimeout):
        stream.read(4096, timeout=0.6)
    assert 0.5 <= time.monotonic() - started < 2.0


def test_keepalive_client_reads_through_the_abortable_backend(monkeypatch):
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(process_bootstrap, "_INTERRUPTIBLE_READS", True)
    monkeypatch.setattr(process_bootstrap, "_SHARED_TRANSPORTS", {})
    backend_cls = process_bootstrap._interruptible_classes()[1]
    client = process_bootstrap.build_keepalive_http_client("https://api.example.com/v1")
    try:
        backends = [transport._pool._network_backend
                    for transport in (client._transport, *client._mounts.values()) if hasattr(transport, "_pool")]
        assert backends and all(isinstance(backend, backend_cls) for backend in backends)
    finally:
        client.close()
