from __future__ import annotations

import socket
import struct
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.discord_edge_client import (
    DiscordEdgeClient,
    DiscordEdgeClientError,
    DiscordEdgeServerPeer,
)
from gateway.discord_edge_protocol import (
    RECONCILIATION_NOT_AVAILABLE_ERROR,
    RECONCILIATION_RESPONSE_VERSION,
    DiscordEdgeAuthorityKind,
    DiscordEdgeIntent,
    DiscordEdgeOperation,
    DiscordEdgeReceiptOutcome,
    DiscordEdgeReconciliationQuery,
    DiscordPublicTarget,
    DiscordPublicTargetType,
    canonical_json_bytes,
    make_request,
    parse_request,
    parse_request_for_reconciliation,
    parse_reconciliation_query,
    sign_capability,
    sign_receipt,
    verify_request_capability,
)

_HEADER = struct.Struct("!I")


@pytest.fixture
def short_socket_path():
    directory = Path(tempfile.mkdtemp(prefix="dec-", dir="/tmp"))
    try:
        yield directory / "edge.sock"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class _Authorizer:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def authorize(self, peer: DiscordEdgeServerPeer) -> bool:
        return self.allowed and peer == DiscordEdgeServerPeer(222, 333, 444)


def _peer(_sock: socket.socket) -> DiscordEdgeServerPeer:
    return DiscordEdgeServerPeer(222, 333, 444)


def _signed_request():
    writer_key = Ed25519PrivateKey.generate()
    edge_key = Ed25519PrivateKey.generate()
    intent = DiscordEdgeIntent(
        operation=DiscordEdgeOperation.PUBLIC_MESSAGE_SEND,
        target=DiscordPublicTarget(
            DiscordPublicTargetType.PUBLIC_GUILD_CHANNEL,
            "123456789012345678",
            "223456789012345678",
        ),
        payload={"content": "exact route-back"},
        idempotency_key="case-1:routeback:1",
    )
    now = int(time.time() * 1000)
    capability_envelope = sign_capability(
        writer_key,
        intent,
        authority_kind=DiscordEdgeAuthorityKind.CANONICAL_ROUTEBACK,
        authority_ref="routeauth:exact",
        issued_at_unix_ms=now,
        expires_at_unix_ms=now + 60_000,
    )
    request = make_request(
        intent,
        capability_envelope,
        now_unix_ms=now,
        timeout_seconds=20,
    )
    capability = verify_request_capability(
        request,
        writer_key.public_key(),
        now_unix_ms=now,
    )
    return request, capability, edge_key


def _reconciliation_query(request):
    return DiscordEdgeReconciliationQuery(
        idempotency_key=request.intent.idempotency_key,
        operation=request.intent.operation,
        target=request.intent.target,
        request_sha256=request.intent.request_sha256,
        content_sha256=request.intent.content_sha256,
    )


def _response(request, capability, edge_key, *, channel_id=None) -> bytes:
    receipt = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id="323456789012345678",
        bot_user_id="423456789012345678",
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        occurred_at_unix_ms=int(time.time() * 1000),
    ).to_message()
    if channel_id is not None:
        receipt["payload"]["target"]["channel_id"] = channel_id
    body = canonical_json_bytes(
        {
            "state": "verified",
            "blocker": None,
            "replayed": False,
            "receipt": receipt,
        }
    )
    return _HEADER.pack(len(body)) + body


def _server(
    path: Path,
    response_builder,
    *,
    close_without_response: bool = False,
) -> threading.Thread:
    ready = threading.Event()

    def run() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        conn, _ = listener.accept()
        try:
            header = conn.recv(_HEADER.size)
            if close_without_response or not header:
                return
            (size,) = _HEADER.unpack(header)
            body = b""
            while len(body) < size:
                body += conn.recv(size - len(body))
            request = parse_request(__import__("json").loads(body))
            conn.sendall(response_builder(request))
        finally:
            conn.close()
            listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread


def _reconciliation_server(path: Path, response_builder) -> threading.Thread:
    ready = threading.Event()

    def run() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        conn, _ = listener.accept()
        try:
            header = conn.recv(_HEADER.size)
            if not header:
                return
            (size,) = _HEADER.unpack(header)
            body = b""
            while len(body) < size:
                body += conn.recv(size - len(body))
            query = parse_reconciliation_query(__import__("json").loads(body))
            response = canonical_json_bytes(response_builder(query))
            conn.sendall(_HEADER.pack(len(response)) + response)
        finally:
            conn.close()
            listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2)
    return thread


def _reconciliation_response(request, capability, edge_key):
    receipt = sign_receipt(
        edge_key,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id="323456789012345678",
        bot_user_id="423456789012345678",
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        occurred_at_unix_ms=int(time.time() * 1000),
    )
    return {
        "protocol": RECONCILIATION_RESPONSE_VERSION,
        "request": request.to_message(),
        "state": "verified",
        "blocker": None,
        "replayed": True,
        "receipt": receipt.to_message(),
    }


def test_preconnected_client_returns_exact_signed_receipt(short_socket_path):
    request, capability, edge_key = _signed_request()
    path = short_socket_path
    thread = _server(
        path,
        lambda observed: _response(observed, capability, edge_key),
    )
    client = DiscordEdgeClient(
        path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )
    client.connect()
    result = client.execute(request.to_message())
    client.close()
    thread.join(2)
    assert result.state == "verified"
    assert result.blocker is None
    assert result.replayed is False
    assert result.receipt.payload["edge_request_id"] == request.request_id


def test_preconnect_replaces_peer_closed_cached_socket_before_claim(
    short_socket_path,
):
    request, capability, edge_key = _signed_request()
    ready = threading.Event()
    first_closed = threading.Event()

    def run() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(short_socket_path))
        listener.listen(2)
        ready.set()
        first, _ = listener.accept()
        first.close()
        first_closed.set()
        second, _ = listener.accept()
        try:
            header = second.recv(_HEADER.size)
            (size,) = _HEADER.unpack(header)
            body = b""
            while len(body) < size:
                body += second.recv(size - len(body))
            observed = parse_request(__import__("json").loads(body))
            second.sendall(_response(observed, capability, edge_key))
        finally:
            second.close()
            listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(2)
    client = DiscordEdgeClient(
        short_socket_path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )

    client.connect()
    assert first_closed.wait(2)
    client.connect()
    result = client.execute(request.to_message())
    client.close()
    thread.join(2)

    assert result.state == "verified"
    assert result.receipt.payload["edge_request_id"] == request.request_id


def test_response_binding_mismatch_is_uncertain_and_never_retried(short_socket_path):
    request, capability, edge_key = _signed_request()
    path = short_socket_path
    thread = _server(
        path,
        lambda observed: _response(
            observed,
            capability,
            edge_key,
            channel_id="999999999999999999",
        ),
    )
    client = DiscordEdgeClient(
        path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )
    client.connect()
    with pytest.raises(DiscordEdgeClientError) as caught:
        client.execute(request.to_message())
    thread.join(2)
    assert caught.value.dispatch_uncertain is True


def test_post_send_connection_loss_is_uncertain(short_socket_path):
    request, _capability, _edge_key = _signed_request()
    path = short_socket_path
    thread = _server(path, lambda _request: b"", close_without_response=True)
    client = DiscordEdgeClient(
        path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )
    client.connect()
    with pytest.raises(DiscordEdgeClientError) as caught:
        client.execute(request.to_message())
    thread.join(2)
    assert caught.value.dispatch_uncertain is True


def test_execute_requires_explicit_preconnect_before_writer_claim(tmp_path):
    request, _capability, _edge_key = _signed_request()
    client = DiscordEdgeClient(
        tmp_path / "absent.sock",
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )
    with pytest.raises(DiscordEdgeClientError) as caught:
        client.execute(request.to_message())
    assert caught.value.code == "discord_edge_not_preconnected"
    assert caught.value.dispatch_uncertain is False


def test_unauthorized_edge_peer_is_rejected_before_request(short_socket_path):
    path = short_socket_path
    ready = threading.Event()

    def serve() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        listener.listen(1)
        ready.set()
        conn, _ = listener.accept()
        conn.close()
        listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(2)
    client = DiscordEdgeClient(
        path,
        server_authorizer=_Authorizer(False),
        server_peer_getter=_peer,
    )
    with pytest.raises(DiscordEdgeClientError) as caught:
        client.connect()
    thread.join(2)
    assert caught.value.code == "discord_edge_server_unauthorized"


def test_reconcile_recovers_expired_original_request_without_preconnect(
    short_socket_path,
):
    request, capability, edge_key = _signed_request()
    expired_message = request.to_message()
    expired_message["deadline_unix_ms"] = 1
    expired_request = parse_request_for_reconciliation(expired_message)
    query = _reconciliation_query(expired_request)
    observed = []
    thread = _reconciliation_server(
        short_socket_path,
        lambda parsed: (
            observed.append(parsed),
            _reconciliation_response(expired_request, capability, edge_key),
        )[1],
    )
    client = DiscordEdgeClient(
        short_socket_path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )

    result = client.reconcile(query.to_message())
    client.close()
    thread.join(2)

    assert observed == [query]
    assert result.request.deadline_unix_ms == 1
    assert result.request.to_message() == expired_request.to_message()
    assert result.state == "verified"
    assert result.replayed is True


def test_reconcile_binding_mismatch_fails_without_dispatch_uncertainty(
    short_socket_path,
):
    request, capability, edge_key = _signed_request()
    query = _reconciliation_query(request)

    def mismatched_response(_query):
        response = _reconciliation_response(request, capability, edge_key)
        response["request"]["target"]["channel_id"] = "999999999999999999"
        return response

    thread = _reconciliation_server(short_socket_path, mismatched_response)
    client = DiscordEdgeClient(
        short_socket_path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )
    with pytest.raises(DiscordEdgeClientError) as caught:
        client.reconcile(query)
    thread.join(2)

    assert caught.value.code == "discord_edge_reconciliation_failed"
    assert caught.value.dispatch_uncertain is False


def test_reconcile_not_available_is_explicit_and_never_blocked(
    short_socket_path,
):
    request, _capability, _edge_key = _signed_request()
    query = _reconciliation_query(request)
    thread = _reconciliation_server(
        short_socket_path,
        lambda _query: {
            "protocol": RECONCILIATION_RESPONSE_VERSION,
            "error": RECONCILIATION_NOT_AVAILABLE_ERROR,
        },
    )
    client = DiscordEdgeClient(
        short_socket_path,
        server_authorizer=_Authorizer(),
        server_peer_getter=_peer,
    )

    with pytest.raises(DiscordEdgeClientError) as caught:
        client.reconcile(query)
    thread.join(2)

    assert caught.value.code == "discord_edge_reconciliation_not_available"
    assert caught.value.dispatch_uncertain is False
    assert "blocked" not in caught.value.code
