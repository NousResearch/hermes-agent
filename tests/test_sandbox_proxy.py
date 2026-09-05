from __future__ import annotations

import importlib.util
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest


PROXY_PATH = Path(__file__).parents[1] / "scripts" / "sandbox" / "proxy.py"


def load_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fixture_root = tmp_path / "http"
    certs = tmp_path / "certs"
    fixture_root.mkdir()
    certs.mkdir()
    real_ca = certs / "real-ca.pem"
    real_ca.write_text("test-only", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(PROXY_PATH), str(fixture_root), str(certs), str(real_ca)],
    )
    spec = importlib.util.spec_from_file_location("sandbox_proxy_under_test", PROXY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_client_ca_bundle_trusts_sandbox_and_public_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = load_proxy(tmp_path, monkeypatch)
    sandbox_ca = (
        b"-----BEGIN CERTIFICATE-----\nsandbox\n-----END CERTIFICATE-----\n"
    )
    stale_public_ca = (
        b"-----BEGIN CERTIFICATE-----\nstale\n-----END CERTIFICATE-----\n"
    )
    public_cas = b"-----BEGIN CERTIFICATE-----\npublic\n-----END CERTIFICATE-----\n"
    client_bundle = proxy.CERTS / "ca.pem"
    client_bundle.write_bytes(sandbox_ca + stale_public_ca)
    proxy.REAL_CA.write_bytes(public_cas)

    proxy.prepare_client_ca_bundle()
    first = client_bundle.read_bytes()
    proxy.prepare_client_ca_bundle()

    assert first == sandbox_ca + public_cas
    assert client_bundle.read_bytes() == first


def test_connect_without_fixture_uses_raw_tunnel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = load_proxy(tmp_path, monkeypatch)
    tunneled: list[tuple[object, str, int]] = []

    class Connection:
        def sendall(self, _data: bytes) -> None:
            pass

    connection = Connection()

    monkeypatch.setattr(
        proxy,
        "tunnel_connect",
        lambda conn, host, port: tunneled.append((conn, host, port)),
        raising=False,
    )
    monkeypatch.setattr(
        proxy,
        "cert_for",
        lambda _host: pytest.fail("non-fixture HTTPS must not be intercepted"),
    )

    proxy.handle_connect(connection, "registry.npmjs.org:443")

    assert tunneled == [(connection, "registry.npmjs.org", 443)]


def test_connect_with_fixture_keeps_mitm_interception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = load_proxy(tmp_path, monkeypatch)
    host = "hermes-agent.nousresearch.com"
    (proxy.ROOT / host).mkdir()
    sent: list[bytes] = []
    loaded: list[tuple[Path, Path]] = []

    class Connection:
        def sendall(self, data: bytes) -> None:
            sent.append(data)

    class EmptyTls:
        def __enter__(self) -> EmptyTls:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def recv(self, _size: int) -> bytes:
            return b""

    class ServerContext:
        def load_cert_chain(self, cert: Path, key: Path) -> None:
            loaded.append((cert, key))

        def wrap_socket(self, _conn: object, *, server_side: bool) -> EmptyTls:
            assert server_side is True
            return EmptyTls()

    cert = tmp_path / "fixture.pem"
    key = tmp_path / "fixture.key"
    monkeypatch.setattr(
        proxy,
        "tunnel_connect",
        lambda *_args: pytest.fail("fixture HTTPS must remain intercepted"),
    )
    monkeypatch.setattr(proxy, "cert_for", lambda _host: (cert, key))
    monkeypatch.setattr(proxy.ssl, "SSLContext", lambda _protocol: ServerContext())

    proxy.handle_connect(Connection(), f"{host}:443")

    assert sent == [b"HTTP/1.1 200 Connection Established\r\n\r\n"]
    assert loaded == [(cert, key)]


def test_raw_tunnel_relays_both_directions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = load_proxy(tmp_path, monkeypatch)
    listener = socket.create_server(("127.0.0.1", 0))
    target_port = listener.getsockname()[1]
    client, proxy_side = socket.socketpair()
    server_errors: list[BaseException] = []

    def echo_once() -> None:
        try:
            upstream, _ = listener.accept()
            with upstream:
                request = upstream.recv(64)
                upstream.sendall(b"echo:" + request)
        except BaseException as error:  # pragma: no cover - surfaced below
            server_errors.append(error)

    echo_thread = threading.Thread(target=echo_once)
    proxy_thread = threading.Thread(
        target=proxy.tunnel_connect,
        args=(proxy_side, "127.0.0.1", target_port),
    )
    echo_thread.start()
    proxy_thread.start()
    try:
        response = b""
        while b"\r\n\r\n" not in response:
            response += client.recv(64)
        assert response == b"HTTP/1.1 200 Connection Established\r\n\r\n"

        client.sendall(b"ping")
        assert client.recv(64) == b"echo:ping"
        client.shutdown(socket.SHUT_WR)
    finally:
        client.close()
        proxy_side.close()
        listener.close()
        echo_thread.join(timeout=2)
        proxy_thread.join(timeout=2)

    assert not server_errors
    assert not echo_thread.is_alive()
    assert not proxy_thread.is_alive()
