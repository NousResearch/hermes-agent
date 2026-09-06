"""Invariant tests for the dev-sandbox MITM proxy's upstream relay.

The proxy forwards real-Internet HTTPS (PyPI, npm, the uv/Node CDNs) for the
install/update E2E legs. In CI those peers frequently abort the TLS session
without a close_notify, which Python surfaces as ``ssl.SSLEOFError``
(``UNEXPECTED_EOF_WHILE_READING``) on the next ``recv`` -- NOT a clean empty
read. ``relay`` must end the stream cleanly on that signal, because the
recipient usually already holds a complete response; bubbling the error
truncates the payload and the install dies (every install.sh download failed
at once on the first Install & Update E2E run). We assert ``relay`` survives
the abrupt EOF and a raw reset/abort the same way.
"""

from __future__ import annotations

import importlib.util
import ssl
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROXY = REPO_ROOT / "scripts" / "sandbox" / "proxy.py"


def _load_proxy():
    spec = importlib.util.spec_from_file_location("sandbox_proxy_test", PROXY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # proxy.py reads sys.argv[1:4] at import time; satisfy it harmlessly.
    import sys

    sys.argv = ["proxy.py", "/tmp/root", "/tmp/certs", "/tmp/ca.pem"]
    spec.loader.exec_module(module)
    return module


def _fake_socket(payload: bytes, *, raise_on_recv=None):
    """A stand-in for the wrapped upstream socket.

    ``raise_on_recv`` (an exception class) makes ``recv`` raise it on the
    FIRST call after returning the payload, modelling a peer that sent a full
    response and then dropped the TLS session without a close_notify.
    """

    class _Sock:
        _buf = memoryview(payload)
        _pos = 0
        _raised = False

        def recv(self, n):
            if raise_on_recv is not None and not self._raised and self._pos >= len(self._buf):
                self._raised = True
                raise raise_on_recv("peer aborted")
            if self._pos >= len(self._buf):
                return b""
            out = bytes(self._buf[self._pos : self._pos + n])
            self._pos += len(out)
            return out

        def sendall(self, data):  # pragma: no cover - not exercised here
            pass

    return _Sock()


@pytest.mark.parametrize(
    "error_cls",
    [
        ssl.SSLEOFError,
        ConnectionResetError,
        BrokenPipeError,
    ],
)
def test_relay_survives_abrupt_upstream_eof(error_cls, monkeypatch):
    proxy = _load_proxy()
    payload = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello"
    sent = []

    class _Dest:
        def sendall(self, data):
            sent.append(bytes(data))

    up = _fake_socket(payload, raise_on_recv=error_cls)
    proxy.relay(up, _Dest())

    # The full response reached the recipient before the error fired.
    assert b"".join(sent) == payload
    # No exception escaped relay().
    assert sent


def test_relay_clean_eof_still_terminates(monkeypatch):
    proxy = _load_proxy()
    payload = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
    sent = []

    class _Dest:
        def sendall(self, data):
            sent.append(bytes(data))

    up = _fake_socket(payload)  # plain clean end-of-stream
    proxy.relay(up, _Dest())
    assert b"".join(sent) == payload
